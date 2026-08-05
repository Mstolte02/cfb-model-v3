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


def opponent_matrix(teams, schedule):
    """Row-stochastic A where A[i, j] is j's share of i's schedule."""
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    A = np.zeros((n, n))
    for t in teams:
        opps = [o for o in schedule.get(t, []) if o in idx]
        if not opps:
            continue
        wgt = 1.0 / len(opps)
        for o in opps:
            A[idx[t], idx[o]] += wgt
    return A


def solve_srs(O0, D0, schedule, alpha):
    """Solve the SRS fixed point EXACTLY, on the complement of the constant vector.

    The fixed point is x = x0 + alpha * M x with M = [[0, A], [A, 0]] and A the
    row-stochastic opponent-averaging matrix. It used to be run as 25 rounds of
    substitution, which is a power iteration on M, and A is row-stochastic - so its
    largest eigenvalue is exactly 1, on the all-ones vector, and the spectral radius
    of alpha*M is exactly alpha.

    AT THE SHIPPED alpha = 1.0 THAT ITERATION DOES NOT CONVERGE, it drifts along the
    constants forever, and past 1.0 it diverges geometrically. The final
    re-standardization hid it, which is why the tuning curve showed Brier falling to
    1.0 and then "off a cliff" - the cliff was the solver, not football, and the
    sweep could not have found an optimum above 1.0 even if one existed.

    The constant direction is also the one direction that CANNOT MATTER: adding the
    same number to every team's rating changes nothing, and the caller z-scores the
    result anyway. So it is projected out and the remaining system is solved directly.
    (I - alpha*P*M*P) is nonsingular on that subspace for any alpha whose
    second-largest |eigenvalue| stays below 1/alpha, which is a real condition on the
    schedule graph rather than an artifact of how many rounds were run.
    """
    teams = list(O0.index)
    n = len(teams)
    A = opponent_matrix(teams, schedule)

    M = np.zeros((2 * n, 2 * n))
    M[:n, n:] = A          # offence is revised by the defences it faced
    M[n:, :n] = A          # and defence by the offences
    x0 = np.concatenate([O0.to_numpy(float), D0.to_numpy(float)])

    # centering projector, applied per block: the constant direction is unidentified
    P = np.eye(2 * n)
    P[:n, :n] -= 1.0 / n
    P[n:, n:] -= 1.0 / n

    T = alpha * (P @ M @ P)
    x = np.linalg.solve(np.eye(2 * n) - T, P @ x0)
    return pd.Series(x[:n], index=teams), pd.Series(x[n:], index=teams)


def adjust(std_year: pd.DataFrame, schedule: dict, alpha=1.0, iters=25) -> pd.DataFrame:
    """Opponent-adjusted O/D composites. `iters` is accepted and ignored - the system
    is solved rather than iterated; see solve_srs."""
    O0, D0 = od_ratings(std_year)
    O, D = solve_srs(O0, D0, schedule, alpha)
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
        X, Y = solve_srs(X0, Y0, schedule, alpha)   # exact, as in adjust()
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
