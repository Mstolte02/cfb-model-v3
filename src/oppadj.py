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


# --- per-stat adjustment ------------------------------------------------------
# adjust() corrects the COMPOSITE against the opposing composite, which charges a
# team's red-zone defence for the pass rush of the offences it faced. Three features
# come in true pairs - the same event scored from both sides - so they can be
# corrected against their own counterpart instead:
#
#     off_havoc         <-> def_havoc        disruption allowed / generated
#     off_rz_td         <-> def_rz_td        red-zone TD scored / allowed
#     off_press_allowed <-> def_press        pressure allowed / generated
#
# The other four have no exact counterpart in the feature set (there is no
# def_success_rate, and def_ppa is all plays against an offensive RUSH ppa), so they
# keep the composite correction. EXACT_PAIRS is the defensible set; LOOSE_PAIRS adds
# the two approximate rush/efficiency matches, which is a hypothesis to be tested
# rather than an obvious improvement - see scripts/per_stat_oppadj.py.
EXACT_PAIRS = [("off_havoc", "def_havoc"),
               ("off_rz_td", "def_rz_td"),
               ("off_press_allowed", "def_press")]
LOOSE_PAIRS = EXACT_PAIRS + [("off_rush_ppa", "def_line_yds"),
                             ("off_success_rate", "def_ppa")]


def _opp_mean(vals: pd.Series, schedule: dict, teams, tset) -> pd.Series:
    return pd.Series(
        {t: np.mean([vals[o] for o in schedule.get(t, []) if o in tset] or [0.0])
         for t in teams})


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std(ddof=0) or 1.0)


def adjust_per_stat(std_year: pd.DataFrame, schedule: dict, alpha=1.0, iters=25,
                    pairs=None) -> pd.DataFrame:
    """Opponent-adjust each PAIRED stat against its own counterpart, then rebuild
    the composites. Unpaired stats fall back to the composite correction.

    Same fixed point as adjust(), run once per pair instead of once per composite.
    Every stat is already sign-flipped so higher == better for the team it belongs
    to, so the update reads identically to the composite one: a team that faced
    strong opposing pass rushes has its pressure-allowed number revised UP.
    """
    from src.matchup import OFF_STATS, DEF_STATS
    pairs = EXACT_PAIRS if pairs is None else pairs
    teams = std_year.index
    tset = set(teams)
    out = std_year.copy()

    paired = {c for p in pairs for c in p}
    for off_c, def_c in pairs:
        if off_c not in out.columns or def_c not in out.columns:
            continue
        X0, Y0 = _z(out[off_c]), _z(out[def_c])
        X, Y = X0.copy(), Y0.copy()
        for _ in range(iters):
            mX = _opp_mean(Y, schedule, teams, tset)
            mY = _opp_mean(X, schedule, teams, tset)
            X, Y = X0 + alpha * mX, Y0 + alpha * mY
        out[off_c], out[def_c] = _z(X), _z(Y)

    # Unpaired stats still need SOME correction, and the only counterpart available
    # is the opposing composite - which is exactly what adjust() does for everything.
    rest_off = [c for c in OFF_STATS if c not in paired and c in out.columns]
    rest_def = [c for c in DEF_STATS if c not in paired and c in out.columns]
    if rest_off or rest_def:
        comp = adjust(std_year, schedule, alpha, iters)
        for c in rest_off:
            out[c] = _z(_z(std_year[c]) + alpha * _opp_mean(comp["D"], schedule, teams, tset))
        for c in rest_def:
            out[c] = _z(_z(std_year[c]) + alpha * _opp_mean(comp["O"], schedule, teams, tset))

    O = _z(out[[c for c in OFF_STATS if c in out.columns]].mean(axis=1))
    D = _z(out[[c for c in DEF_STATS if c in out.columns]].mean(axis=1))
    return pd.DataFrame({"O": O, "D": D})


def build_od_by_year_per_stat(std_by_year: dict, games_by_year: dict, alpha=1.0,
                              pairs=None) -> dict:
    out = {}
    for y, std in std_by_year.items():
        if y in games_by_year:
            out[y] = adjust_per_stat(std, build_schedule(games_by_year[y]),
                                     alpha, pairs=pairs)
    return out
