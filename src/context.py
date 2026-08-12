"""Pregame schedule and travel context for the v4 model.

Every value in this module is available from the published schedule before kickoff.
The builder deliberately does not use realized weather: an archived observation is
not the forecast a bettor or coach had before the game.  Weather can be added later
through a timestamped forecast feed without weakening the temporal contract here.

Distances are great-circle distances, not invented road/flight itineraries.  A
shortest-path algorithm cannot recover actual team travel without route or stay-over
data; on a complete graph with great-circle edge weights, Dijkstra collapses to the
direct edge by the triangle inequality.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config import ROOT


CONTEXT_FEATURES = [
    "rest_days_diff",
    "short_rest_diff",
    "log_travel_miles_diff",
    "timezone_shift_diff",
    "eastward_shift_diff",
    "prior_21d_travel_diff",
    "road_streak_diff",
]


def _haversine_miles(lat1, lon1, lat2, lon2) -> float:
    if any(v is None or not np.isfinite(float(v)) for v in (lat1, lon1, lat2, lon2)):
        return 0.0
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp = p2 - p1
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 3958.7613 * 2 * math.asin(math.sqrt(min(1.0, a)))


def _parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _location(row: dict | None) -> dict:
    row = row or {}
    loc = row.get("location") or row
    return {
        "id": loc.get("id"), "latitude": loc.get("latitude"),
        "longitude": loc.get("longitude"), "timezone": loc.get("timezone") or "UTC",
    }


def _utc_offset_hours(tz_name: str, kickoff: datetime) -> float:
    try:
        offset = kickoff.astimezone(ZoneInfo(tz_name)).utcoffset()
        return float(offset.total_seconds() / 3600.0) if offset is not None else 0.0
    except (KeyError, ValueError):
        return 0.0


def _raw(root: Path, name: str):
    path = root / "data" / "raw" / name
    return json.loads(path.read_text()) if path.exists() else []


def _team_locations(root: Path, years) -> dict[int, dict[str, dict]]:
    out = {}
    for year in years:
        out[int(year)] = {r["school"]: _location(r) for r in _raw(root, f"teams_{year}.json")}
    return out


def _venue_locations(root: Path) -> dict[int, dict]:
    return {int(r["id"]): _location(r) for r in _raw(root, "venues.json") if r.get("id") is not None}


def _team_game_state(team: str, kickoff: datetime, venue: dict, home: dict,
                     previous: list[dict]) -> dict[str, float]:
    if previous:
        rest = (kickoff - previous[-1]["kickoff"]).total_seconds() / 86400.0
        rest = float(np.clip(rest, 3.0, 21.0))
    else:
        rest = 7.0

    miles = _haversine_miles(home.get("latitude"), home.get("longitude"),
                             venue.get("latitude"), venue.get("longitude"))
    home_offset = _utc_offset_hours(home.get("timezone", "UTC"), kickoff)
    venue_offset = _utc_offset_hours(venue.get("timezone", "UTC"), kickoff)
    shift = venue_offset - home_offset
    cutoff = kickoff - timedelta(days=21)
    prior_miles = sum(r["trip_miles"] for r in previous if r["kickoff"] >= cutoff)

    streak = 0
    for r in reversed(previous):
        if r["at_home_campus"]:
            break
        streak += 1
    return {
        "rest_days": rest,
        "short_rest": float(rest < 6.5),
        "log_travel_miles": math.log1p(miles),
        "timezone_shift": abs(shift),
        "eastward_shift": max(0.0, shift),
        "prior_21d_travel": math.log1p(prior_miles),
        "road_streak": float(streak),
        "trip_miles": miles,
    }


def build_context(years, root: Path = ROOT) -> pd.DataFrame:
    """Return one directional context row per completed raw game.

    The key is ``(season, week, home_team, away_team)``. Feature signs are home minus
    away, so swapping the complete matchup negates the vector.
    """
    years = [int(y) for y in years]
    teams_by_year = _team_locations(root, years)
    venues = _venue_locations(root)
    rows = []
    for year in years:
        games = [g for g in _raw(root, f"games_{year}.json")
                 if g.get("completed") and g.get("startDate")]
        games.sort(key=lambda g: (_parse_date(g["startDate"]), g.get("id", 0)))
        history: dict[str, list[dict]] = defaultdict(list)
        team_locs = teams_by_year.get(year, {})
        for g in games:
            home_team, away_team = g.get("homeTeam"), g.get("awayTeam")
            if not home_team or not away_team:
                continue
            kickoff = _parse_date(g["startDate"])
            home_loc, away_loc = team_locs.get(home_team, {}), team_locs.get(away_team, {})
            venue = venues.get(int(g["venueId"])) if g.get("venueId") is not None else None
            if not venue:
                venue = home_loc if not g.get("neutralSite", False) else {}
            hs = _team_game_state(home_team, kickoff, venue, home_loc, history[home_team])
            aws = _team_game_state(away_team, kickoff, venue, away_loc, history[away_team])
            row = {"season": year, "week": g.get("week"), "home_team": home_team,
                   "away_team": away_team, "kickoff": kickoff.isoformat(),
                   "venue_id": g.get("venueId"), "neutral_site": bool(g.get("neutralSite", False))}
            for name in CONTEXT_FEATURES:
                stem = name.removesuffix("_diff")
                row[name] = hs[stem] - aws[stem]
            rows.append(row)

            for team, state, loc in ((home_team, hs, home_loc), (away_team, aws, away_loc)):
                at_home = (not g.get("neutralSite", False) and venue.get("id") is not None
                           and venue.get("id") == loc.get("id"))
                history[team].append({"kickoff": kickoff, "trip_miles": state["trip_miles"],
                                      "at_home_campus": bool(at_home)})
    return pd.DataFrame(rows)


def attach_context(predictions: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "week", "home_team", "away_team"]
    keep = keys + [c for c in CONTEXT_FEATURES if c in context]
    return predictions.merge(context[keep].drop_duplicates(keys), on=keys, how="left",
                             validate="one_to_one")


class OffsetLogit:
    """Ridge logistic correction with the base model logit used as a fixed offset."""

    def __init__(self, feature_names=None, penalty=20.0):
        self.feature_names = list(feature_names or CONTEXT_FEATURES)
        self.penalty = float(penalty)
        self.scale = None
        self.coef = None

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def fit(self, frame: pd.DataFrame, base_col="p_dynamic", target_col="y"):
        X = frame[self.feature_names].fillna(0.0).to_numpy(float)
        y = frame[target_col].to_numpy(float)
        offset = self._logit(frame[base_col])
        self.scale = np.std(X, axis=0, ddof=0)
        self.scale = np.where(np.isfinite(self.scale) & (self.scale > 1e-8), self.scale, 1.0)
        Xs = X / self.scale
        beta = np.zeros(X.shape[1], float)
        eye = np.eye(X.shape[1])
        for _ in range(50):
            z = np.clip(offset + Xs @ beta, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-z))
            w = np.maximum(p * (1 - p), 1e-6)
            grad = Xs.T @ (p - y) + self.penalty * beta
            hess = Xs.T @ (Xs * w[:, None]) + self.penalty * eye
            step = np.linalg.solve(hess, grad)
            beta -= step
            if np.max(np.abs(step)) < 1e-8:
                break
        self.coef = beta / self.scale
        return self

    def predict(self, frame: pd.DataFrame, base_col="p_dynamic") -> np.ndarray:
        X = frame[self.feature_names].fillna(0.0).to_numpy(float)
        z = np.clip(self._logit(frame[base_col]) + X @ self.coef, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))

    def payload(self):
        return {"architecture": "ridge_logistic_context_with_v4_logit_offset",
                "feature_names": self.feature_names, "penalty": self.penalty,
                "coef": dict(zip(self.feature_names, self.coef.tolist()))}
