"""Feature engineering (methodology §4.1), forward-looking and leakage-free.

- standardize(): z-score each feature within a season and flip defensive signs
  so larger == stronger for every feature. Returns the scaler so the projection
  season can be transformed identically.
- build_matchup_rows(): for each game in season N, build the differential
  feature vector from each team's season (N-1) standardized stats. An "average
  opponent on a neutral field" is the zero vector, so a team's own standardized
  vector already expresses its win prob vs average (doc §3 / §4.3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import FEATURES

FEATURE_COLS = list(FEATURES)


def standardize(team_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Z-score within season; flip defensive features. Indexed by team."""
    out = team_df[["team"]].copy()
    params = {}
    for feat, meta in FEATURES.items():
        col = team_df[feat].astype(float) if feat in team_df else pd.Series(np.nan, index=team_df.index)
        mu, sd = col.mean(), (col.std(ddof=0) or 1.0)
        z = (col - mu) / sd
        if not meta["higher_is_better"]:
            z = -z
        # Missing (e.g. TruMedia before 2021) -> neutral 0 after standardizing.
        out[feat] = z.fillna(0.0).values
        params[feat] = {"mean": mu, "std": sd, "flip": not meta["higher_is_better"]}
    return out.set_index("team"), params


def apply_scaler(team_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = team_df[["team"]].copy()
    for feat, p in params.items():
        z = (team_df[feat].astype(float) - p["mean"]) / (p["std"] or 1.0)
        if p["flip"]:
            z = -z
        out[feat] = z.values
    return out.set_index("team")


def build_matchup_rows(
    std_prior: pd.DataFrame, games_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X, y, is_home) for games, using PRIOR-season standardized team vectors.

    One row per game, from the home team's perspective:
        X       = home_vec - away_vec     (both from season N-1)
        y       = 1 if home team won
        is_home = 0 at neutral sites, else 1  (lets the model learn HFA, §3)
    """
    teams = set(std_prior.index)
    X, y, home_flag = [], [], []
    for _, g in games_df.iterrows():
        h, a = g["home_team"], g["away_team"]
        if h not in teams or a not in teams:
            continue  # team with no prior-year stats (FCS, dropped-for-missing)
        if g["home_points"] == g["away_points"]:
            continue  # ties ~ nonexistent in CFB
        diff = (std_prior.loc[h, FEATURE_COLS].values
                - std_prior.loc[a, FEATURE_COLS].values).astype(float)
        X.append(diff)
        y.append(1 if g["home_points"] > g["away_points"] else 0)
        home_flag.append(0 if g.get("neutral_site", False) else 1)
    return np.array(X), np.array(y), np.array(home_flag)
