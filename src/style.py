"""Game control: which style a team wins in, and whether it can impose that style.

Two style dimensions are measured per game, both from cached CFBD data:

* **pace** - game-clock seconds per offensive play for the whole game, from the
  drive summaries (`src.tempo.build_team_game_metrics`, valid drives only);
* **pass environment** - the neutral-situation pass rate of each offense, from the
  play-by-play: downs 1-2, quarters 1-3, score within 14. No fitted expectation is
  involved, so nothing about later games can leak into the definition.

For each team, from its previous ``WINDOW`` games only (start-of-week snapshots, so
a slate never sees itself), three things per dimension:

* **preference** - the style it plays at: its own offensive pace, its own pass rate;
* **control** - how far the realized game moved toward its preference rather than
  the opponent's, on [-1, 1] (the pace version is the existing `tempo` measure);
* **fit** - a shrunk slope of its margin *versus expectation* on the game's style.
  Expectation is CFBD's own pregame Elo, so the residual is independent of every
  model in this repository and cannot carry this model's errors back into it.

A matchup then gets an expected style - each side's preference weighted by its
control - and a style edge, ``fit_home * (E - mean) - fit_away * (E - mean)``, in
points. Every matchup column is antisymmetric in the two teams.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import ndtri

from config import DATA_RAW, GAME_YEARS
from src import tempo

KEYS = ["season", "week", "home_team", "away_team"]
WINDOW = 16
FIT_SHRINK = 6.0          # prior games' worth of zero slope
ELO_HFA = 55.0            # CFBD's own home-field Elo points
ELO_SIGMA = 17.0          # points per probit unit, the model's margin scale
PACE_MEAN, PASS_MEAN = 26.5, .50

COLUMNS = [
    "pace_pref_diff", "pace_fit_edge", "pace_control_diff",
    "pass_pref_diff", "pass_fit_edge", "pass_control_diff", "pass_expected_x_fit",
    "style_edge", "expected_pace_z",
]


def _neutral_pass_rates(year: int) -> dict[tuple[int, str], tuple[float, int]]:
    plays = pd.read_csv(DATA_RAW / f"plays_slim_{year}.csv.gz",
                        usecols=["game_id", "offense", "period", "down",
                                 "score_diff", "kind"])
    keep = (plays.kind.isin(["pass", "rush"]) & plays.down.isin([1, 2]) &
            plays.period.between(1, 3) & (plays.score_diff.abs() <= 14))
    plays = plays.loc[keep]
    grouped = plays.groupby(["game_id", "offense"]).kind
    rate = grouped.apply(lambda k: float((k == "pass").mean()))
    count = grouped.size()
    return {(int(g), t): (float(rate[(g, t)]), int(count[(g, t)]))
            for g, t in rate.index}


def _elo_residuals(year: int) -> dict[int, dict]:
    games = json.loads((DATA_RAW / f"games_{year}.json").read_text())
    out = {}
    for g in games:
        if not g.get("completed") or g.get("homePoints") is None:
            continue
        eh, ea = g.get("homePregameElo"), g.get("awayPregameElo")
        if eh is None or ea is None:
            continue
        hfa = 0.0 if g.get("neutralSite") else ELO_HFA
        p = 1.0 / (1.0 + 10 ** (-(eh - ea + hfa) / 400.0))
        expected = ELO_SIGMA * float(ndtri(min(max(p, .01), .99)))
        margin = float(g["homePoints"]) - float(g["awayPoints"])
        out[int(g["id"])] = {"week": int(g.get("week") or 0), "season_type":
                             g.get("seasonType", "regular"),
                             "home": g["homeTeam"], "away": g["awayTeam"],
                             "residual": margin - expected}
    return out


def _slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return 0.0
    x = np.asarray(xs) - float(np.mean(xs))
    return float(np.dot(x, ys) / (np.dot(x, x) + FIT_SHRINK * float(np.var(xs) or 1.0)))


def _control(realized: float, own: float, other: float) -> float:
    gap = abs(own - other)
    if not np.isfinite(realized) or gap < 1e-9:
        return 0.0
    return float(np.clip((abs(realized - other) - abs(realized - own)) / gap, -1, 1))


def _snapshot(history: list[dict]) -> dict:
    rows = history[-WINDOW:]

    def mean(name, default):
        v = [r[name] for r in rows if np.isfinite(r.get(name, np.nan))]
        return float(np.mean(v)) if v else default

    pace_rows = [r for r in rows if np.isfinite(r.get("game_spp", np.nan))]
    pass_rows = [r for r in rows if np.isfinite(r.get("pass_env", np.nan))]
    return {
        "pace_pref": mean("own_spp", PACE_MEAN),
        "pace_control": mean("pace_control", 0.0),
        "pace_fit": _slope([r["game_spp"] for r in pace_rows],
                           [r["residual"] for r in pace_rows]),
        "pass_pref": mean("own_pass", PASS_MEAN),
        "pass_control": mean("pass_control", 0.0),
        "pass_fit": _slope([r["pass_env"] for r in pass_rows],
                           [r["residual"] for r in pass_rows]),
    }


def _weights(control_h: float, control_a: float) -> float:
    """Home share of the realized style, from the two sides' control records."""
    return 1.0 / (1.0 + math.exp(-2.0 * (control_h - control_a)))


def _matchup(key: tuple, h: dict, a: dict) -> dict:
    w = _weights(h["pace_control"], a["pace_control"])
    pace_e = w * h["pace_pref"] + (1 - w) * a["pace_pref"]
    wp = _weights(h["pass_control"], a["pass_control"])
    pass_e = wp * h["pass_pref"] + (1 - wp) * a["pass_pref"]
    pace_edge = (h["pace_fit"] - a["pace_fit"]) * (pace_e - PACE_MEAN)
    pass_edge = (h["pass_fit"] - a["pass_fit"]) * (pass_e - PASS_MEAN)
    return {
        **dict(zip(KEYS, key)),
        "pace_pref_diff": h["pace_pref"] - a["pace_pref"],
        "pace_fit_edge": pace_edge,
        "pace_control_diff": h["pace_control"] - a["pace_control"],
        "pass_pref_diff": h["pass_pref"] - a["pass_pref"],
        "pass_fit_edge": pass_edge,
        "pass_control_diff": h["pass_control"] - a["pass_control"],
        # Symmetric expected style times an antisymmetric fit difference stays
        # antisymmetric, so the column negates when the teams swap.
        "pass_expected_x_fit": (pass_e - PASS_MEAN) * (h["pass_fit"] - a["pass_fit"]),
        "style_edge": pace_edge + pass_edge,
        # Symmetric; only ever enters multiplied by an antisymmetric column.
        "expected_pace_z": (PACE_MEAN - pace_e) / 1.5,
    }


def build_style_features(years=GAME_YEARS) -> pd.DataFrame:
    """Start-of-week style snapshots for every regular-season FBS game with data."""
    metrics = tempo.build_team_game_metrics(years)
    games_meta = {}
    for year in years:
        rates = _neutral_pass_rates(year)
        for gid, row in _elo_residuals(year).items():
            if row["season_type"] != "regular":
                continue
            key = (int(year), row["week"], row["home"], row["away"])
            games_meta[key] = {**row, "id": gid,
                               "pass_h": rates.get((gid, row["home"]), (np.nan, 0)),
                               "pass_a": rates.get((gid, row["away"]), (np.nan, 0))}
    history = defaultdict(list)
    by_week = defaultdict(list)
    for key in games_meta:
        by_week[(key[0], key[1])].append(key)
    rows = []
    for season_week in sorted(by_week):
        pending = []
        for key in sorted(by_week[season_week], key=lambda k: (k[2], k[3])):
            _, _, home, away = key
            h, a = _snapshot(history[home]), _snapshot(history[away])
            rows.append(_matchup(key, h, a))
            meta = games_meta[key]
            drive = metrics.get(key, {"home": {}, "away": {}})
            spp = drive["home"].get("game_spp", np.nan)
            ph = meta["pass_h"][0] if meta["pass_h"][1] >= 10 else np.nan
            pa = meta["pass_a"][0] if meta["pass_a"][1] >= 10 else np.nan
            env = np.nanmean([ph, pa]) if np.isfinite(ph) or np.isfinite(pa) else np.nan
            pace_c = _control(spp, h["pace_pref"], a["pace_pref"])
            pass_c = _control(env, h["pass_pref"], a["pass_pref"])
            res = meta["residual"]
            pending.append((home, {"game_spp": spp, "own_spp": drive["home"].get(
                "off_spp", np.nan), "pace_control": pace_c, "pass_env": env,
                "own_pass": ph, "pass_control": pass_c, "residual": res}))
            pending.append((away, {"game_spp": spp, "own_spp": drive["away"].get(
                "off_spp", np.nan), "pace_control": -pace_c, "pass_env": env,
                "own_pass": pa, "pass_control": -pass_c, "residual": -res}))
        for team, game in pending:
            history[team].append(game)
    return pd.DataFrame(rows).drop_duplicates(KEYS)
