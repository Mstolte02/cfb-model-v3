"""Win-probability model: L2 logistic blended with a ridge margin model.

v2 changes (validated by LOSO in scripts/loso_experiments*.py):
  1. Isotonic calibration DROPPED — the linear logistic is already calibrated;
     the isotonic map only added noise (LOSO: 0.2052 -> 0.2048 Brier without it).
  2. A ridge regression on game MARGIN (same features) is fit alongside and its
     implied probability Phi(margin/sigma) is blended with the logistic:
         p = ENSEMBLE_W * p_logistic + (1 - ENSEMBLE_W) * p_margin
     Margin carries information the binary outcome discards (LOSO: 0.2048 ->
     0.2044 Brier, log-loss 0.5930 -> 0.5919, accuracy 67.4 -> 67.7%).

A home-field indicator is appended to the design matrix so HFA is learned, not
hard-coded.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score

from config import ARTIFACTS, C_GRID, ENSEMBLE_W
from src.features import FEATURE_COLS


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class CFBModel:
    coef: np.ndarray
    hfa_coef: float
    intercept: float
    C: float
    # Ridge margin model (same features + home flag), for the probability blend
    # and for spread-style margin predictions.
    margin_coef: np.ndarray = None
    margin_hfa: float = 0.0
    margin_intercept: float = 0.0
    margin_sigma: float = 14.0
    ens_w: float = ENSEMBLE_W
    feature_names: list = None  # names of the columns in `coef`

    def _raw_prob(self, x_diff: np.ndarray, is_home: float) -> float:
        z = self.intercept + self.coef @ x_diff + self.hfa_coef * is_home
        return 1.0 / (1.0 + np.exp(-z))

    def pred_margin(self, x_diff: np.ndarray, is_home: float = 0.0) -> float:
        """Expected margin (team - opp points) from the ridge margin model."""
        return float(self.margin_intercept + self.margin_coef @ x_diff
                     + self.margin_hfa * is_home)

    def win_prob(self, x_diff: np.ndarray, is_home: float = 0.0) -> float:
        """P(team wins): logistic blended with the margin-implied probability."""
        p = self._raw_prob(x_diff, is_home)
        if self.margin_coef is not None:
            p_mg = _norm_cdf(self.pred_margin(x_diff, is_home) / self.margin_sigma)
            p = self.ens_w * p + (1.0 - self.ens_w) * p_mg
        return float(p)

    def strength_vs_average(self, team_vec: np.ndarray) -> float:
        """Doc §3: win prob vs an average opponent (zero vector) on neutral field."""
        return self.win_prob(team_vec, is_home=0.0)

    def save(self, path=ARTIFACTS / "model.json") -> None:
        path.write_text(json.dumps({
            "features": self.feature_names or FEATURE_COLS,
            "coef": self.coef.tolist(),
            "hfa_coef": self.hfa_coef,
            "intercept": self.intercept,
            "C": self.C,
            "margin_coef": (self.margin_coef.tolist()
                            if self.margin_coef is not None else None),
            "margin_hfa": self.margin_hfa,
            "margin_intercept": self.margin_intercept,
            "margin_sigma": self.margin_sigma,
            "ens_w": self.ens_w,
        }, indent=2))

    @classmethod
    def load(cls, path=ARTIFACTS / "model.json") -> "CFBModel":
        d = json.loads(path.read_text())
        mc = d.get("margin_coef")
        return cls(np.array(d["coef"]), d["hfa_coef"], d["intercept"], d["C"],
                   margin_coef=None if mc is None else np.array(mc),
                   margin_hfa=d.get("margin_hfa", 0.0),
                   margin_intercept=d.get("margin_intercept", 0.0),
                   margin_sigma=d.get("margin_sigma", 14.0),
                   ens_w=d.get("ens_w", ENSEMBLE_W),
                   feature_names=d.get("features"))


def _design(X, home_flag):
    return np.column_stack([X, home_flag.astype(float)])


def train(X, y, home_flag, C=None, feature_names=None,
          margins=None) -> tuple["CFBModel", LogisticRegression]:
    """Fit L2 logistic (C by CV on Brier); if `margins` (home - away points per
    game) is given, also fit the ridge margin model for the probability blend."""
    design = _design(X, home_flag)

    if C is None:
        scores = {
            c: cross_val_score(
                LogisticRegression(C=c, max_iter=2000),  # L2 is the default penalty
                design, y, cv=5, scoring="neg_brier_score",
            ).mean()
            for c in C_GRID
        }
        C = max(scores, key=scores.get)

    clf = LogisticRegression(C=C, max_iter=2000).fit(design, y)  # default = L2/Ridge

    n_feat = X.shape[1]
    margin_kw = {}
    if margins is not None:
        alpha_grid = [0.5, 1.0, 3.0, 10.0, 30.0]
        alpha = max(alpha_grid, key=lambda a: cross_val_score(
            Ridge(alpha=a), design, margins, cv=5,
            scoring="neg_mean_absolute_error").mean())
        rg = Ridge(alpha=alpha).fit(design, margins)
        margin_kw = dict(
            margin_coef=rg.coef_[:n_feat],
            margin_hfa=float(rg.coef_[n_feat]),
            margin_intercept=float(rg.intercept_),
            margin_sigma=float(np.std(margins - rg.predict(design))),
        )

    model = CFBModel(
        coef=clf.coef_[0][:n_feat],
        hfa_coef=float(clf.coef_[0][n_feat]),
        intercept=float(clf.intercept_[0]),
        C=float(C),
        feature_names=feature_names,
        **margin_kw,
    )
    return model, clf


def evaluate(model: "CFBModel", X, y, home_flag, margins=None) -> dict:
    """Out-of-sample Brier, log-loss, accuracy + a 10-bin reliability curve."""
    p = np.array([model.win_prob(X[i], home_flag[i]) for i in range(len(y))])
    p = np.clip(p, 1e-6, 1 - 1e-6)
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    acc = float(np.mean((p > 0.5) == y))

    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    reliability = []
    for b in range(10):
        m = idx == b
        if m.sum():
            reliability.append({
                "bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
                "predicted": round(float(p[m].mean()), 3),
                "actual": round(float(y[m].mean()), 3),
                "n": int(m.sum()),
            })
    return {"brier": round(brier, 4), "log_loss": round(logloss, 4),
            "accuracy": round(acc, 4), "n_games": int(len(y)),
            "reliability": reliability}
