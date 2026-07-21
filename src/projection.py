"""Calibrated preseason projection = "a more accurate mean" (doc §C, improved).

Rather than regressing prior stats toward raw talent, we fit a per-stat
projection of each team's standardized stat for the season being ENTERED:

    stat_z(N)[f] ~ b0 + b1*prior_stat_z(N-1)[f]
                      + b2*talent_z(N)
                      + b3*returning_z(N)
                      + b4*pythag_z(N-1)

All predictors are preseason-available for season N (no leakage). Coefficients
are fit on TRAIN game-years only. The fitted value is the team's projected
strength entering season N; we re-standardize the projection within the season
so it feeds the win model on the same scale the model was trained on.

Pythagorean win expectation (points-based, luck-adjusted) is a cleaner
team-quality signal than raw W-L:  PF^x / (PF^x + PA^x).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import PYTHAG_EXP
from src.features import FEATURE_COLS

PRED_NAMES = ["prior_stat", "talent", "returning", "pythag"]


def _z(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    return (s - s.mean()) / (s.std(ddof=0) or 1.0)


def pythagorean(games_df: pd.DataFrame, exp: float = PYTHAG_EXP) -> pd.Series:
    """Per-team Pythagorean win expectation from a season's game scores."""
    pf, pa = {}, {}
    for _, g in games_df.iterrows():
        h, a, hp, ap = g["home_team"], g["away_team"], g["home_points"], g["away_points"]
        pf[h] = pf.get(h, 0) + hp; pa[h] = pa.get(h, 0) + ap
        pf[a] = pf.get(a, 0) + ap; pa[a] = pa.get(a, 0) + hp
    rows = {}
    for t in pf:
        f, a = pf[t] ** exp, pa[t] ** exp
        rows[t] = f / (f + a) if (f + a) else 0.5
    return pd.Series(rows, name="pythag")


def build_pythag(games_by_year: dict, exp: float = PYTHAG_EXP) -> dict:
    """{year: standardized Pythagorean win-expectation Series}."""
    return {y: _z(pythagorean(g, exp)) for y, g in games_by_year.items()}


def _predictor_frame(N, std_by_year, talent_by_year, returning_by_year,
                     pythag_by_year, feat):
    """Assemble aligned predictors for season N (entering)."""
    prior = N - 1
    if prior not in std_by_year or N not in talent_by_year:
        return None
    if N not in returning_by_year or prior not in pythag_by_year:
        return None
    df = pd.DataFrame({
        "prior_stat": std_by_year[prior][feat],
        "talent": talent_by_year[N],
        "returning": returning_by_year[N],
        "pythag": pythag_by_year[prior],
    })
    return df


def fit_projection(train_game_years, std_by_year, talent_by_year,
                   returning_by_year, pythag_by_year) -> dict:
    """Per-feature OLS coefficients [intercept, prior_stat, talent, returning,
    pythag], fit pooled over train game-years (target = actual stat_z(N))."""
    coeffs = {}
    for f in FEATURE_COLS:
        rows, tgt = [], []
        for N in train_game_years:
            preds = _predictor_frame(N, std_by_year, talent_by_year,
                                     returning_by_year, pythag_by_year, f)
            if preds is None or N not in std_by_year:
                continue
            target = std_by_year[N][f]
            j = preds.join(target.rename("tgt"), how="inner").dropna()
            if len(j):
                rows.append(j[PRED_NAMES].values)
                tgt.append(j["tgt"].values)
        X = np.vstack(rows); y = np.concatenate(tgt)
        Xb = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        coeffs[f] = beta
    return coeffs


def project(N, std_by_year, talent_by_year, returning_by_year, pythag_by_year,
            coeffs) -> pd.DataFrame:
    """Projected, re-standardized strength vectors for teams entering season N."""
    cols = {}
    for f in FEATURE_COLS:
        preds = _predictor_frame(N, std_by_year, talent_by_year,
                                 returning_by_year, pythag_by_year, f)
        if preds is None:
            # Fall back to prior stats if a predictor block is missing.
            cols[f] = std_by_year[N - 1][f]
            continue
        b = coeffs[f]
        p = preds.dropna()
        pred = b[0] + p[PRED_NAMES].values @ b[1:]
        cols[f] = pd.Series(pred, index=p.index)
    out = pd.DataFrame(cols)
    # Re-standardize within the projected set so the win model sees a stable scale.
    for f in FEATURE_COLS:
        out[f] = _z(out[f])
    return out.dropna()
