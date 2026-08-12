"""Temporally clean, reciprocal team model used by the v4 production pipeline.

The old model's strongest backtested input used season-N participation and snaps to
predict season N.  V4 has a stricter contract: historical features for N may depend on
data timestamped no later than the preseason of N.  PFF and WAR therefore enter as
completed N-1 TEAM summaries; entering recruiting talent and returning production are
the only N-labelled historical inputs.

The game score is antisymmetric on a neutral field.  Every team has one feature vector
``f(team)`` and every game uses ``f(home) - f(away)``.  Both the win logit and predicted
margin have no intercept; home field is a separate indicator.  Consequently:

    P(A beats B, neutral) + P(B beats A, neutral) == 1
    margin(A, B, neutral) == -margin(B, A, neutral)

This is an invariant, not an empirical hope.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from config import ARTIFACTS
from src.matchup import OFF_STATS, DEF_STATS


CORE_FEATURES = ["O", "D", "talent", "returning", "pff_lag", "war_lag"]
TEAM_FEATURES = [*OFF_STATS, *DEF_STATS, "talent", "returning", "pff_lag", "war_lag"]
MATCHUP_PAIRS = {
    "match_success": ("off_success_rate", "def_ppa"),
    "match_rush": ("off_rush_ppa", "def_line_yds"),
    "match_havoc": ("off_havoc", "def_havoc"),
    "match_red_zone": ("off_rz_td", "def_rz_td"),
    "match_pressure": ("off_press_allowed", "def_press"),
}
INTERACTION_FEATURES = list(MATCHUP_PAIRS)


def _sigmoid(z):
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def build_frame(year: int, std_by_year: dict, talent_by_year: dict,
                returning_by_year: dict, od_by_year: dict,
                pff_lag_by_year: dict | None = None,
                war_lag_by_year: dict | None = None,
                granular: bool = False) -> pd.DataFrame | None:
    """Team features entering ``year``, using only prior/preseason information."""
    prior = year - 1
    if (prior not in std_by_year or prior not in od_by_year or
            year not in talent_by_year or year not in returning_by_year):
        return None
    std, od = std_by_year[prior], od_by_year[prior]
    core = pd.DataFrame({
        "O": od.O,
        "D": od.D,
        "talent": talent_by_year[year],
        "returning": returning_by_year[year],
    }).dropna()

    # A missing licensed player source is unknown, not evidence of weak talent.
    # Zero is the within-season mean after standardization; coverage is recorded in
    # attrs for diagnostics instead of made into a football coefficient.
    pff = (pff_lag_by_year or {}).get(year, pd.Series(dtype=float))
    war = (war_lag_by_year or {}).get(year, pd.Series(dtype=float))
    core["pff_lag"] = pff.reindex(core.index).fillna(0.0)
    core["war_lag"] = war.reindex(core.index).fillna(0.0)
    core.attrs["pff_coverage"] = float(pff.reindex(core.index).notna().mean())
    core.attrs["war_coverage"] = float(war.reindex(core.index).notna().mean())

    if not granular:
        return core
    raw = std[[c for c in [*OFF_STATS, *DEF_STATS] if c in std.columns]]
    return core.join(raw, how="inner")


def _row_matchup_vector(fh: pd.Series, fa: pd.Series,
                        feature_names: list[str]) -> np.ndarray:
    """Build a matchup vector from two team rows."""
    values = []
    for name in feature_names:
        if name in MATCHUP_PAIRS:
            off, deff = MATCHUP_PAIRS[name]
            edge_ha = float(fh[off] - fa[deff])
            edge_ah = float(fa[off] - fh[deff])
            # Odd nonlinear contrast: exceptional mismatches can have more than a
            # linear effect, while swapping teams negates the term exactly.
            values.append(edge_ha * abs(edge_ha) - edge_ah * abs(edge_ah))
        else:
            values.append(float(fh[name] - fa[name]))
    return np.asarray(values, float)


def matchup_vector(frame: pd.DataFrame, home: str, away: str,
                   feature_names: list[str]) -> np.ndarray:
    """Antisymmetric feature vector: swapping teams negates every value."""
    return _row_matchup_vector(frame.loc[home], frame.loc[away], feature_names)


def average_matchup_vector(frame: pd.DataFrame, team: str,
                           feature_names: list[str]) -> np.ndarray:
    """Team's neutral matchup vector against a league-average feature row."""
    average = frame.select_dtypes(include=[np.number]).mean(axis=0)
    return _row_matchup_vector(frame.loc[team], average, feature_names)


def build_year(frame: pd.DataFrame, games_df: pd.DataFrame,
               feature_names: list[str]):
    """Build a season design plus aligned game metadata."""
    teams = set(frame.index)
    X, y, home, margin, rows = [], [], [], [], []
    for idx, g in games_df.iterrows():
        h, a = g["home_team"], g["away_team"]
        if h not in teams or a not in teams:
            continue
        hp, ap = g["home_points"], g["away_points"]
        if pd.isna(hp) or pd.isna(ap) or hp == ap:
            continue
        neutral = bool(g.get("neutral_site", False))
        X.append(matchup_vector(frame, h, a, feature_names))
        y.append(int(hp > ap))
        home.append(0.0 if neutral else 1.0)
        margin.append(float(hp - ap))
        rows.append({"source_index": int(idx), "week": g.get("week"),
                     "home_team": h, "away_team": a, "neutral_site": neutral})
    return (np.asarray(X, float), np.asarray(y, int), np.asarray(home, float),
            np.asarray(margin, float), pd.DataFrame(rows))


def assemble(years, frames, games_by_year, feature_names):
    return {y: build_year(frames[y], games_by_year[y], feature_names)
            for y in years if y in frames and frames[y] is not None}


@dataclass
class ReciprocalTeamModel:
    feature_names: list[str]
    coef: np.ndarray
    hfa_coef: float
    margin_coef: np.ndarray
    margin_hfa: float
    margin_sigma: float
    C: float = 0.1
    alpha: float = 10.0
    ensemble_weight: float = 0.5
    probability_scale: float = 1.0
    version: str = "4.1"

    def raw_logit(self, x_diff, is_home=0.0):
        return float(self.coef @ np.asarray(x_diff) + self.hfa_coef * is_home)

    def pred_margin(self, x_diff, is_home=0.0):
        return float(self.margin_coef @ np.asarray(x_diff) + self.margin_hfa * is_home)

    def win_prob(self, x_diff, is_home=0.0):
        p_logit = float(_sigmoid(self.raw_logit(x_diff, is_home)))
        p_margin = _norm_cdf(self.pred_margin(x_diff, is_home) / self.margin_sigma)
        p = self.ensemble_weight * p_logit + (1 - self.ensemble_weight) * p_margin
        # Temperature-only calibration preserves P(-x)=1-P(x).
        p = float(np.clip(p, 1e-8, 1 - 1e-8))
        return float(_sigmoid(self.probability_scale * math.log(p / (1 - p))))

    def team_logit_strength(self, frame: pd.DataFrame, team: str):
        return self.raw_logit(average_matchup_vector(frame, team,
                                                     self.feature_names), 0.0)

    def save(self, path=ARTIFACTS / "model_v4.json"):
        payload = {
            "version": self.version, "architecture": "reciprocal_team_difference",
            "temporal_contract": "N uses N-1 performance/player summaries plus preseason-N inputs",
            "feature_names": self.feature_names, "coef": self.coef.tolist(),
            "hfa_coef": self.hfa_coef, "margin_coef": self.margin_coef.tolist(),
            "margin_hfa": self.margin_hfa, "margin_sigma": self.margin_sigma,
            "C": self.C, "alpha": self.alpha,
            "ensemble_weight": self.ensemble_weight,
            "probability_scale": self.probability_scale,
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path=ARTIFACTS / "model_v4.json"):
        d = json.loads(Path(path).read_text())
        return cls(feature_names=d["feature_names"], coef=np.asarray(d["coef"]),
                   hfa_coef=d["hfa_coef"], margin_coef=np.asarray(d["margin_coef"]),
                   margin_hfa=d["margin_hfa"], margin_sigma=d["margin_sigma"],
                   C=d.get("C", .1), alpha=d.get("alpha", 10.0),
                   ensemble_weight=d.get("ensemble_weight", .5),
                   probability_scale=d.get("probability_scale", 1.0),
                   version=d.get("version", "4.1"))


def fit(X, y, home_flag, margins, feature_names, *, C=.1, alpha=10.0,
        ensemble_weight=.5, probability_scale=1.0):
    """Fit reciprocal logistic and margin models without an intercept."""
    # Scale but never center antisymmetric differences. Centering would inject a
    # hidden orientation intercept and break P(-x)=1-P(x). Coefficients are converted
    # back to raw-feature units before serialization.
    scale = np.std(X, axis=0, ddof=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    design = np.column_stack([X / scale, np.asarray(home_flag, float)])
    clf = LogisticRegression(C=C, fit_intercept=False, max_iter=5000).fit(design, y)
    ridge = Ridge(alpha=alpha, fit_intercept=False).fit(design, margins)
    residual = np.asarray(margins) - ridge.predict(design)
    sigma = float(np.sqrt(np.mean(residual ** 2))) or 1.0
    n = X.shape[1]
    return ReciprocalTeamModel(
        feature_names=list(feature_names), coef=clf.coef_[0, :n] / scale,
        hfa_coef=float(clf.coef_[0, n]), margin_coef=ridge.coef_[:n] / scale,
        margin_hfa=float(ridge.coef_[n]), margin_sigma=sigma, C=float(C),
        alpha=float(alpha), ensemble_weight=float(ensemble_weight),
        probability_scale=float(probability_scale))


def predict(model: ReciprocalTeamModel, part):
    X, y, home, margins = part[:4]
    p = np.asarray([model.win_prob(x, h) for x, h in zip(X, home)])
    pm = np.asarray([model.pred_margin(x, h) for x, h in zip(X, home)])
    return p, pm


def power_ratings(model: ReciprocalTeamModel, frame: pd.DataFrame) -> pd.DataFrame:
    """Round-robin ratings with exact neutral-site complementarity."""
    teams, rows = list(frame.index), []
    for team in teams:
        probs = [model.win_prob(matchup_vector(frame, team, opp, model.feature_names), 0.0)
                 for opp in teams if opp != team]
        score = model.team_logit_strength(frame, team)
        average_x = average_matchup_vector(frame, team, model.feature_names)
        rows.append({"team": team, "power": float(np.mean(probs)),
                     "vs_average": model.win_prob(average_x, 0.0),
                     "logit_strength": score})
    out = pd.DataFrame(rows).sort_values(["power", "team"], ascending=[False, True])
    out = out.reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def assert_reciprocal(model: ReciprocalTeamModel, frame: pd.DataFrame,
                      atol=1e-12):
    teams = list(frame.index)[:min(40, len(frame))]
    for i, a in enumerate(teams):
        for b in teams[i + 1:]:
            x = matchup_vector(frame, a, b, model.feature_names)
            xr = matchup_vector(frame, b, a, model.feature_names)
            if not np.allclose(xr, -x, atol=atol):
                raise AssertionError("matchup vector is not antisymmetric")
            if abs(model.win_prob(x, 0) + model.win_prob(xr, 0) - 1) > atol:
                raise AssertionError("neutral probabilities are not complementary")
            if abs(model.pred_margin(x, 0) + model.pred_margin(xr, 0)) > atol:
                raise AssertionError("neutral margins are not opposites")
    return True
