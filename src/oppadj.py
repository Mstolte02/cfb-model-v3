"""Opponent (strength-of-schedule) adjustment of the O/D composites — the doc's
"Adj_" stats. Beating weak defenses should count less than beating strong ones.

Iterative SRS-style fixed point on the standardized composites:
    O_adj[t] = O_raw[t] + alpha * mean_over_opp( D_adj[opp] )
    D_adj[t] = D_raw[t] + alpha * mean_over_opp( O_adj[opp] )
i.e. a team that faced tough defenses has its offense revised UP, and vice versa.
alpha=0 recovers raw composites; alpha=1 is full SRS. Result is re-standardized so
it drops into the same pipeline via the od_by_year override.

Uses season-level composites + the schedule (who played whom) — no game-level
splits needed, so it works for every stat including the TruMedia adds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.matchup import od_ratings


def build_schedule(games_df: pd.DataFrame) -> dict:
    """{team: [opponents...]} for a season (both directions)."""
    sched = {}
    for _, g in games_df.iterrows():
        h, a = g["home_team"], g["away_team"]
        sched.setdefault(h, []).append(a)
        sched.setdefault(a, []).append(h)
    return sched


def adjust(std_year: pd.DataFrame, schedule: dict, alpha=1.0, iters=25) -> pd.DataFrame:
    O0, D0 = od_ratings(std_year)
    O, D = O0.copy(), D0.copy()
    teams = O.index
    tset = set(teams)
    for _ in range(iters):
        mean_oppD = pd.Series(
            {t: np.mean([D[o] for o in schedule.get(t, []) if o in tset] or [0.0])
             for t in teams})
        mean_oppO = pd.Series(
            {t: np.mean([O[o] for o in schedule.get(t, []) if o in tset] or [0.0])
             for t in teams})
        O = O0 + alpha * mean_oppD
        D = D0 + alpha * mean_oppO
    O = (O - O.mean()) / (O.std(ddof=0) or 1.0)
    D = (D - D.mean()) / (D.std(ddof=0) or 1.0)
    return pd.DataFrame({"O": O, "D": D})


def build_od_by_year(std_by_year: dict, games_by_year: dict, alpha=1.0) -> dict:
    """{year: opponent-adjusted DataFrame[O, D]} for use as od_by_year."""
    out = {}
    for y, std in std_by_year.items():
        if y in games_by_year:
            out[y] = adjust(std, build_schedule(games_by_year[y]), alpha)
    return out
