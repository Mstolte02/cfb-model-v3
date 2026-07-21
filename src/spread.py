"""Margin / spread model (methodology §5.2) — the payoff of the matchup structure.

Predicts each side's expected points as offense-vs-opponent-defense, then derives
spread and total:

    points_scored ~ a + b1*O_scorer + b2*D_opponent + b3*home    (Ridge)

Each game gives two training rows (home scoring, away scoring), sharing one set of
coefficients (offense helps you score; opponent defense suppresses you; b2<0).
For a matchup we predict both sides:
    margin = pred_home_pts - pred_away_pts      (spread)
    total  = pred_home_pts + pred_away_pts      (over/under)

A win probability falls out via a normal model on the margin: P(home) =
Phi(margin / sigma), with sigma = the train residual SD of game margins.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

POINTS_COLS = ["O_scorer", "D_opponent", "home"]


def build_points_rows(frame: pd.DataFrame, games_df: pd.DataFrame):
    """Two rows per game (each team's scoring), from the entering-season frame."""
    teams = set(frame.index)
    X, y = [], []
    for _, g in games_df.iterrows():
        h, a = g["home_team"], g["away_team"]
        if h not in teams or a not in teams:
            continue
        if pd.isna(g["home_points"]) or pd.isna(g["away_points"]):
            continue
        neutral = bool(g.get("neutral_site", False))
        fh, fa = frame.loc[h], frame.loc[a]
        X.append([fh.O, fa.D, 0.0 if neutral else 1.0]); y.append(g["home_points"])
        X.append([fa.O, fh.D, 0.0]);                      y.append(g["away_points"])
    return np.array(X, float), np.array(y, float)


def fit(X, y, alpha=None):
    """Ridge on points; alpha chosen by CV (neg MAE) if not given."""
    if alpha is None:
        grid = [0.5, 1.0, 3.0, 10.0, 30.0]
        alpha = max(grid, key=lambda a: cross_val_score(
            Ridge(alpha=a), X, y, cv=5, scoring="neg_mean_absolute_error").mean())
    model = Ridge(alpha=alpha).fit(X, y)
    return model, float(alpha)


def predict_side(model, frame, scorer, opponent, scorer_home, neutral=False):
    fs, fo = frame.loc[scorer], frame.loc[opponent]
    h = 0.0 if (neutral or not scorer_home) else 1.0
    return float(model.predict([[fs.O, fo.D, h]])[0])


def game(model, frame, home, away, neutral=False) -> dict:
    hp = predict_side(model, frame, home, away, True, neutral)
    ap = predict_side(model, frame, away, home, False, neutral)
    return {"home_pts": hp, "away_pts": ap,
            "margin": hp - ap, "total": hp + ap,
            "spread_home": -(hp - ap)}  # betting convention: favorite is negative


def margin_sigma(model, frame, games_df) -> float:
    """Residual SD of predicted vs actual game margin (for win-prob conversion)."""
    res = []
    teams = set(frame.index)
    for _, g in games_df.iterrows():
        h, a = g["home_team"], g["away_team"]
        if h not in teams or a not in teams or pd.isna(g["home_points"]):
            continue
        pred = game(model, frame, h, a, bool(g.get("neutral_site", False)))["margin"]
        res.append((g["home_points"] - g["away_points"]) - pred)
    return float(np.std(res)) if res else 14.0


def win_prob_from_margin(margin: float, sigma: float) -> float:
    return float(norm.cdf(margin / sigma))
