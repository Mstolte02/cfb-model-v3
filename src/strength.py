"""Talent layer = regression to a talent-informed mean (methodology §B/§C).

Idea: a team's raw prior-season stats are noisy. We shrink each standardized
stat toward what the team's TALENT predicts:

    adjusted_z[f] = (1 - lam) * prior_z[f]  +  lam * (b[f] * talent_z)

where:
  - prior_z[f]  is the team's standardized stat f from the PRIOR season,
  - talent_z    is the team's standardized talent for the season being entered
                (recruiting composite, known preseason -> leakage-free),
  - b[f]        is the pooled slope of prior_z[f] on talent_z, fit on TRAIN
                seasons only (so high-talent stats anchor high, etc.),
  - lam         is the shrinkage weight (0 = raw stats, 1 = pure talent).

For features uncorrelated with talent, b[f] ~ 0, so the term shrinks toward the
grand mean (0) -- i.e. plain regression to the mean. This is what pulls G5
over-performers (raw-stats outliers) back down and lifts blue-bloods who
under-performed their roster.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import FEATURE_COLS, standardize


def standardize_talent(talent_df: pd.DataFrame) -> pd.Series:
    """z-score talent within a season, indexed by team."""
    t = talent_df.set_index("team")["talent"].astype(float)
    mu, sd = t.mean(), (t.std(ddof=0) or 1.0)
    return (t - mu) / sd


def load_seasons(load, stat_years, talent_years):
    """Return {year: std_stats_df} and {year: talent_z_series}."""
    std_by_year = {y: standardize(load.team_stats(y))[0] for y in stat_years}
    talent_by_year = {y: standardize_talent(load.talent(y)) for y in talent_years}
    return std_by_year, talent_by_year


def fit_talent_coeffs(std_by_year, talent_by_year, train_game_years) -> dict:
    """Pooled per-feature slope of prior-season stat_z on entering-year talent_z.

    For a game in year N: x = talent_z(N), y = stat_z(N-1), paired by team.
    Fit only on train_game_years to keep the held-out season clean.
    """
    coeffs = {}
    for f in FEATURE_COLS:
        xs, ys = [], []
        for N in train_game_years:
            prior = N - 1
            if prior not in std_by_year or N not in talent_by_year:
                continue
            sp = std_by_year[prior][f]
            tz = talent_by_year[N]
            common = sp.index.intersection(tz.index)
            xs.append(tz.loc[common].values)
            ys.append(sp.loc[common].values)
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        # Least-squares slope through standardized vars (intercept ~ 0).
        coeffs[f] = float((x @ y) / (x @ x)) if (x @ x) else 0.0
    return coeffs


def assemble(games_by_year, strength_fn, game_years) -> dict:
    """Build {game_year: (X, y, home_flag)} matchup rows.

    strength_fn(N) -> standardized strength DataFrame (indexed by team) for the
    teams entering season N. Returns None to skip a year.
    """
    from src.features import build_matchup_rows
    parts = {}
    for N in game_years:
        strength = strength_fn(N)
        if strength is None:
            continue
        parts[N] = build_matchup_rows(strength, games_by_year[N])
    return parts


def entering_strength(game_year, std_by_year, talent_by_year, coeffs, lam,
                      use_talent=True) -> pd.DataFrame:
    """Standardized strength vector per team ENTERING `game_year`.

    Without talent: just the prior season's standardized stats.
    With talent: prior stats shrunk toward the talent-implied level.
    """
    prior = game_year - 1
    base = std_by_year[prior].copy()
    if not use_talent or lam == 0:
        return base

    tz = talent_by_year.get(game_year)
    if tz is None:
        return base
    out = base.copy()
    for f in FEATURE_COLS:
        talent_term = coeffs[f] * tz  # Series indexed by team
        aligned = talent_term.reindex(out.index)
        # Teams missing talent keep their raw stat (no shrink).
        blended = (1 - lam) * out[f] + lam * aligned
        out[f] = blended.where(aligned.notna(), out[f])
    return out
