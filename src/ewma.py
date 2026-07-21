"""Recency-weighted (EWMA) team ratings from game-level advanced stats.

Motivation: a team isn't the same in September and December. The season-aggregate
O/D ratings weight every game equally; an EWMA weights recent games more, so the
prior-season rating reflects how a team was *finishing* rather than its average.

build_od_by_year(years, halflife) -> {year: DataFrame[O, D] indexed by team},
standardized within the season's FBS teams. halflife is in GAMES; halflife=inf
recovers the flat (equal-weight) season average.

rolling_series(team, year, halflife) -> per-game cumulative EWMA overall rating,
used to plot ratings over time and the season-boundary regression.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import load

OFF = ["off_ppa", "off_pass_ppa", "off_rush_ppa", "off_success_rate", "off_explosiveness"]
DEF = ["def_ppa", "def_success_rate", "def_explosiveness"]  # lower = better -> negated


def _z(s):
    s = s.astype(float)
    return (s - s.mean()) / (s.std(ddof=0) or 1.0)


def _game_composites(year: int) -> pd.DataFrame:
    """Per team-game offensive/defensive composite, z-scored over FBS team-games."""
    g = load.game_advanced(year)
    fbs = set(load.team_stats(year)["team"])
    g = g[g["team"].isin(fbs)].dropna(subset=OFF + DEF).copy()
    for c in OFF + DEF:
        g[c] = _z(g[c])
    g["O_game"] = g[OFF].mean(axis=1)
    g["D_game"] = -g[DEF].mean(axis=1)          # negate so higher = better defense
    # order games within a team's season
    g["order"] = g["season_type"].map({"regular": 0, "postseason": 1}).fillna(0) * 100 \
        + g["week"].fillna(0)
    return g.sort_values(["team", "order"])


def _ewma_weights(n: int, halflife: float) -> np.ndarray:
    """Weights for n games ordered oldest->newest; newest weight 1."""
    if not np.isfinite(halflife):
        return np.ones(n)
    decay = 0.5 ** (1.0 / halflife)
    ages = np.arange(n - 1, -1, -1)             # oldest has largest age
    return decay ** ages


def build_od_by_year(years, halflife=float("inf")) -> dict:
    out = {}
    for yr in years:
        g = _game_composites(yr)
        rows = []
        for team, grp in g.groupby("team", sort=False):
            w = _ewma_weights(len(grp), halflife)
            O = np.average(grp["O_game"].values, weights=w)
            D = np.average(grp["D_game"].values, weights=w)
            rows.append({"team": team, "O": O, "D": D})
        df = pd.DataFrame(rows).set_index("team")
        df["O"] = _z(df["O"]); df["D"] = _z(df["D"])   # standardize across teams
        out[yr] = df
    return out


def rolling_series(year: int, halflife: float, teams=None) -> pd.DataFrame:
    """Per-game cumulative EWMA 'overall' rating (O + D) for plotting over time."""
    g = _game_composites(year)
    if teams is not None:
        g = g[g["team"].isin(teams)]
    out = []
    for team, grp in g.groupby("team", sort=False):
        grp = grp.reset_index(drop=True)
        for i in range(len(grp)):
            w = _ewma_weights(i + 1, halflife)
            o = np.average(grp["O_game"].values[:i + 1], weights=w)
            d = np.average(grp["D_game"].values[:i + 1], weights=w)
            out.append({"team": team, "season": year, "game_no": i + 1,
                        "week": grp["week"].iloc[i], "rating": o + d})
    return pd.DataFrame(out)
