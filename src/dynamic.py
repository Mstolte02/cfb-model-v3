"""Pregame-only weekly rating updates for the v4 team model.

The preseason model supplies the prior.  Completed games then move a compact team
rating on the natural-logit scale.  An entire week's slate is predicted before any
result from that week is applied, so API/file order can never create look-ahead.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import v4 as V4


def _sigmoid(z):
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0))))


def update_delta(k, margin, rating_gap, actual, expected):
    """Symmetric MOV-aware update in natural-logit units."""
    mov = np.log(abs(float(margin)) + 1.0)
    damping = 2.2 / (abs(float(rating_gap)) * .35 + 2.2)
    return float(k * mov * damping * (float(actual) - float(expected)))


@dataclass
class WeeklyRatingState:
    season: int
    ratings: dict[str, float]
    dynamic_k: float = .15
    dynamic_blend: float = .75
    model_version: str = "4.0"
    processed_games: list[str] = field(default_factory=list)

    @classmethod
    def initialize(cls, model: V4.ReciprocalTeamModel, frame: pd.DataFrame,
                   season: int, dynamic_k=.15, dynamic_blend=.75):
        ratings = {team: model.team_logit_strength(frame, team)
                   for team in frame.index}
        return cls(season=int(season), ratings=ratings,
                   dynamic_k=float(dynamic_k), dynamic_blend=float(dynamic_blend),
                   model_version=model.version)

    @staticmethod
    def game_key(season, week, home, away):
        return f"{int(season)}|{week}|{home}|{away}"

    def predict(self, model: V4.ReciprocalTeamModel, frame: pd.DataFrame,
                home: str, away: str, neutral_site=False):
        if home not in self.ratings or away not in self.ratings:
            raise KeyError(f"unrated team in matchup: {home} vs {away}")
        is_home = 0.0 if neutral_site else 1.0
        x = V4.matchup_vector(frame, home, away, model.feature_names)
        static = model.win_prob(x, is_home)
        gap = self.ratings[home] - self.ratings[away] + model.hfa_coef * is_home
        dynamic = _sigmoid(gap)
        blended = (1.0 - self.dynamic_blend) * static + self.dynamic_blend * dynamic
        return {"p_home": float(blended), "p_static": float(static),
                "p_dynamic": float(dynamic),
                "pred_margin": model.pred_margin(x, is_home),
                "rating_gap": float(gap)}

    def update_week(self, model: V4.ReciprocalTeamModel, frame: pd.DataFrame,
                    games: pd.DataFrame):
        """Predict then update one slate; all predictions use start-of-week state."""
        predictions, changes, keys = [], {}, []
        seen = set(self.processed_games)
        for g in games.itertuples(index=False):
            home, away = g.home_team, g.away_team
            key = self.game_key(self.season, g.week, home, away)
            if key in seen:
                continue
            neutral = bool(getattr(g, "neutral_site", False))
            pred = self.predict(model, frame, home, away, neutral)
            hp, ap = float(g.home_points), float(g.away_points)
            actual = 1.0 if hp > ap else (0.0 if hp < ap else .5)
            delta = update_delta(self.dynamic_k, hp - ap, pred["rating_gap"],
                                 actual, pred["p_dynamic"])
            changes[home] = changes.get(home, 0.0) + delta
            changes[away] = changes.get(away, 0.0) - delta
            predictions.append({"season": self.season, "week": g.week,
                                "home_team": home, "away_team": away,
                                "neutral_site": neutral, "y": actual,
                                "margin": hp - ap, **pred})
            keys.append(key)
        for team, delta in changes.items():
            self.ratings[team] += delta
        self.processed_games.extend(keys)
        return pd.DataFrame(predictions)

    def replay(self, model: V4.ReciprocalTeamModel, frame: pd.DataFrame,
               games: pd.DataFrame):
        rows = []
        ordered = games.sort_values(["week"], kind="stable")
        for _, slate in ordered.groupby("week", sort=True, dropna=False):
            result = self.update_week(model, frame, slate)
            if len(result):
                rows.append(result)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def save(self, path):
        payload = {"season": self.season, "ratings": self.ratings,
                   "dynamic_k": self.dynamic_k,
                   "dynamic_blend": self.dynamic_blend,
                   "model_version": self.model_version,
                   "processed_games": self.processed_games}
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def weekly_replay(model: V4.ReciprocalTeamModel, frame: pd.DataFrame, part,
                  k=.15, blend=.75):
    """Array-compatible strict historical replay used by the selection harness."""
    X, y, hf, margins, meta = part
    state = WeeklyRatingState.initialize(model, frame, season=0,
                                         dynamic_k=k, dynamic_blend=blend)
    order = meta.assign(_row=np.arange(len(meta))).sort_values(["week", "_row"])
    out = np.zeros(len(y)); dynamic = np.zeros(len(y))
    for _, slate in order.groupby("week", sort=True, dropna=False):
        changes = {}
        for _, r in slate.iterrows():
            i, home, away = int(r["_row"]), r["home_team"], r["away_team"]
            is_home = hf[i]
            gap = state.ratings[home] - state.ratings[away] + model.hfa_coef * is_home
            pdyn = _sigmoid(gap)
            pstatic = model.win_prob(X[i], is_home)
            out[i] = (1 - blend) * pstatic + blend * pdyn
            dynamic[i] = pdyn
            delta = update_delta(k, margins[i], gap, y[i], pdyn)
            changes[home] = changes.get(home, 0.0) + delta
            changes[away] = changes.get(away, 0.0) - delta
        for team, delta in changes.items():
            state.ratings[team] += delta
    return out, dynamic


def current_power_ratings(model: V4.ReciprocalTeamModel, frame: pd.DataFrame,
                          state: WeeklyRatingState) -> pd.DataFrame:
    """Round-robin ratings using the selected preseason/dynamic blend."""
    preseason = V4.power_ratings(model, frame).set_index("team")
    teams = list(frame.index)
    mean_rating = float(np.mean([state.ratings[t] for t in teams]))
    rows = []
    for team in teams:
        dynamic_power = float(np.mean([
            _sigmoid(state.ratings[team] - state.ratings[opp])
            for opp in teams if opp != team
        ]))
        dynamic_average = _sigmoid(state.ratings[team] - mean_rating)
        static_power = float(preseason.at[team, "power"])
        static_average = float(preseason.at[team, "vs_average"])
        weight = state.dynamic_blend
        rows.append({"team": team,
                     "power": (1 - weight) * static_power + weight * dynamic_power,
                     "vs_average": ((1 - weight) * static_average +
                                    weight * dynamic_average),
                     "preseason_power": static_power,
                     "dynamic_power": dynamic_power,
                     "dynamic_rating": state.ratings[team]})
    out = pd.DataFrame(rows).sort_values(["power", "team"],
                                         ascending=[False, True]).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out
