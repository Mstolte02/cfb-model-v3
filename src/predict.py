"""Predictions from a trained model (methodology §3).

- matchup(): calibrated P(team A beats team B), symmetric, with optional home field.
- power_ratings(): round-robin "strength relative to the field". Every team plays
  every other on a NEUTRAL field (is_home=0), so the rating isn't contaminated by
  home-field (the original notebook set every team to home). This is the
  deterministic expectation the doc's Monte Carlo (§3) approximates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import FEATURE_COLS
from src.model import CFBModel


def matchup(model: CFBModel, std_team: pd.DataFrame, team_a: str, team_b: str,
            home: str | None = None) -> dict:
    """home: team_a, team_b, or None (neutral)."""
    va = std_team.loc[team_a, FEATURE_COLS].values.astype(float)
    vb = std_team.loc[team_b, FEATURE_COLS].values.astype(float)
    is_home = 1.0 if home == team_a else (-1.0 if home == team_b else 0.0)
    p = model.win_prob(va - vb, is_home=is_home)
    return {"team_a": team_a, "team_b": team_b, "home": home,
            "p_a_wins": round(p, 4), "p_b_wins": round(1 - p, 4)}


def power_ratings(model: CFBModel, std_team: pd.DataFrame) -> pd.DataFrame:
    """Average neutral-field win prob vs every other team -> field strength."""
    teams = list(std_team.index)
    vecs = {t: std_team.loc[t, FEATURE_COLS].values.astype(float) for t in teams}
    rows = []
    for t in teams:
        probs = [model.win_prob(vecs[t] - vecs[o], is_home=0.0)
                 for o in teams if o != t]
        rows.append({"team": t, "power": float(np.mean(probs)),
                     "vs_average": model.strength_vs_average(vecs[t])})
    out = pd.DataFrame(rows).sort_values("power", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    return out
