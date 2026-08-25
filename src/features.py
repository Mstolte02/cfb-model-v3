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
from sklearn.linear_model import Ridge

from config import FEATURES

FEATURE_COLS = list(FEATURES)
COACH_FEATURES = [
    "hc_prior_effect", "hc_prior_offense", "hc_prior_defense", "hc_has_prior",
    "hc_tenure_year", "hc_first_year", "hc_effect_delta",
    "hc_effect_x_talent", "hc_effect_x_returning",
    "hc_first_year_x_talent", "hc_first_year_x_returning",
]


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


def _shrunk_crossed_effects(history: pd.DataFrame, outcome: str,
                            coach_lambda: float = 4.0,
                            program_lambda: float = 4.0) -> dict[int, float]:
    """Deterministic empirical-Bayes approximation for an expanding fold.

    Existing preseason observables are residualized in the same training fit. The
    program and coach intercepts are then alternately updated with fixed partial-
    pooling penalties; no target-season row enters.
    """
    if history.empty:
        return {}
    xcols = ["prior_offense", "prior_defense", "talent", "returning"]
    x = history[xcols].to_numpy(float)
    y = history[outcome].to_numpy(float)
    ridge = Ridge(alpha=1.0).fit(x, y)
    residual = y - ridge.predict(x)
    work = history[["team_id", "coach_id"]].copy()
    work["residual"] = residual
    program, coach = {}, {}
    for _ in range(30):
        work["coach_component"] = work.coach_id.map(coach).fillna(0.0)
        centered = work.residual - work.coach_component
        counts = work.groupby("team_id").size()
        means = centered.groupby(work.team_id).mean()
        program = (means * counts / (counts + program_lambda)).to_dict()
        work["program_component"] = work.team_id.map(program).fillna(0.0)
        centered = work.residual - work.program_component
        counts = work.groupby("coach_id").size()
        means = centered.groupby(work.coach_id).mean()
        coach = (means * counts / (counts + coach_lambda)).to_dict()
    return {int(key): float(value) for key, value in coach.items()}


def leakage_safe_coach_features(outcomes: pd.DataFrame,
                                assignments: pd.DataFrame,
                                target_years) -> dict[int, pd.DataFrame]:
    """Build season-N coach features from completed team seasons no later than N-1."""
    result = {}
    for season in sorted(int(value) for value in target_years):
        history = outcomes[outcomes.season < season]
        effects = {
            "rating_overall": _shrunk_crossed_effects(history, "rating_overall"),
            "rating_offense": _shrunk_crossed_effects(history, "rating_offense"),
            "rating_defense": _shrunk_crossed_effects(history, "rating_defense"),
        }
        current = assignments[assignments.season == season].copy()
        current = current.set_index("team")
        coach_ids = current.coach_id
        prior_ids = current.prior_coach_id
        current["hc_prior_effect"] = coach_ids.map(effects["rating_overall"]).fillna(0.0)
        current["hc_prior_offense"] = coach_ids.map(effects["rating_offense"]).fillna(0.0)
        current["hc_prior_defense"] = coach_ids.map(effects["rating_defense"]).fillna(0.0)
        current["hc_has_prior"] = coach_ids.isin(effects["rating_overall"]).astype(float)
        outgoing = prior_ids.map(effects["rating_overall"]).fillna(0.0)
        current["hc_effect_delta"] = np.where(
            current.hc_change.astype(bool), current.hc_prior_effect - outgoing, 0.0)
        current["hc_tenure_year"] = current.hc_tenure_year.astype(float)
        current["hc_first_year"] = current.hc_first_year.astype(float)
        # Products are completed only after joining to the model frame, because
        # talent and returning production are season-N preseason inputs.
        for column in ("hc_effect_x_talent", "hc_effect_x_returning",
                       "hc_first_year_x_talent", "hc_first_year_x_returning"):
            current[column] = 0.0
        current.attrs["max_outcome_season"] = (
            int(history.season.max()) if not history.empty else None)
        result[season] = current
    return result


def attach_coach_features(frame: pd.DataFrame, coach: pd.DataFrame) -> pd.DataFrame:
    """Join and complete team-level interactions while preserving neutral defaults."""
    out = frame.join(coach[COACH_FEATURES], how="left")
    out[COACH_FEATURES] = out[COACH_FEATURES].fillna(0.0)
    out["hc_effect_x_talent"] = out.hc_prior_effect * out.talent
    out["hc_effect_x_returning"] = out.hc_prior_effect * out.returning
    out["hc_first_year_x_talent"] = out.hc_first_year * out.talent
    out["hc_first_year_x_returning"] = out.hc_first_year * out.returning
    return out
