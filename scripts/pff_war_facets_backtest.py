"""Do API-only PFF player facets improve forward team signal?

This is a promotion gate, not the production WAR build.  It constructs player
values with the same candidate grammar as ``war_model/candidates.py``, aggregates
them to team-season features, and predicts next-season adjusted win percentage in
expanding folds.  Existing artifacts are never overwritten.

Only if the extended set improves out of sample should the expensive WAR weighting
and player-projection pipeline be rebuilt with these facets.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from config import ARTIFACTS, ROOT


WAR_MODEL = ROOT / "war_model"
if str(WAR_MODEL) not in sys.path:
    sys.path.insert(0, str(WAR_MODEL))
# The backtest evaluates every API family by design.  Production rebuilds remain
# unchanged unless their caller makes the same explicit opt-in.
os.environ.setdefault("PFF_API_WAR_REPORTS", "all")
import candidates as C  # noqa: E402


OUT = ARTIFACTS / "pff_war_facets_backtest.json"
API_PREFIXES = ("pblk__", "rblk__", "prsh__", "rdef__", "cov__")
ARM_LABELS = {
    "pblk__": "pass_blocking", "rblk__": "run_blocking",
    "prsh__": "pass_rush", "rdef__": "run_defense", "cov__": "coverage",
}
ALPHAS = np.array([1.0, 10.0, 30.0, 100.0, 300.0])


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame,
                 features: list[str]) -> tuple[np.ndarray, float]:
    model = RidgeCV(alphas=ALPHAS).fit(train[features], train.target)
    return model.predict(test[features]), float(model.alpha_)


def _metrics(actual, predicted) -> dict:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return {
        "n": int(len(actual)),
        "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "mae": float(np.mean(np.abs(predicted - actual))),
        "correlation": float(np.corrcoef(actual, predicted)[0, 1]),
    }


def _team_cluster_bootstrap(frame: pd.DataFrame, left: str, right: str,
                            draws: int = 5000, seed: int = 20260922) -> dict:
    data = frame.copy()
    data["loss_difference"] = ((data[left] - data.target) ** 2 -
                               (data[right] - data.target) ** 2)
    blocks = [group.loss_difference.to_numpy(float)
              for _, group in data.groupby("team", sort=True)]
    observed = float(data.loss_difference.mean())
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for index in range(draws):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        samples[index] = np.concatenate([blocks[i] for i in chosen]).mean()
    return {
        "estimand": f"MSE({left}) - MSE({right})",
        "difference": observed,
        "ci95": [float(value) for value in np.quantile(samples, [.025, .975])],
        "probability_left_better": float(np.mean(samples < 0)),
        "clusters": int(len(blocks)), "draws": draws, "seed": seed,
    }


def main() -> None:
    catalogue = C.build_catalogue()
    base_cat = [row for row in catalogue
                if not row[1].startswith(API_PREFIXES)]
    api_cat = [row for row in catalogue if row[1].startswith(API_PREFIXES)]
    if not api_cat:
        raise SystemExit(
            "API position reports are incomplete; run sync_pff_api with the five "
            "WAR position reports for every non-2020 season first")

    print(f"catalogue: {len(base_cat)} existing + {len(api_cat)} API-only")
    players = C.load_players()
    print(f"player-team-seasons: {len(players):,}")
    values = C.facet_values(players, catalogue=base_cat + api_cat, verbose=False)
    _, matrix, live = C.team_matrix(values)
    base = [row[0] for row in base_cat if row[0] in live]
    api = [row[0] for row in api_cat if row[0] in live]
    api_arms = {
        label: [row[0] for row in api_cat
                if row[0] in live and row[1].startswith(prefix)]
        for prefix, label in ARM_LABELS.items()
    }
    print(f"live features: {len(base)} existing + {len(api)} API-only")

    features = matrix.reset_index()
    records = pd.read_csv(WAR_MODEL / "records.csv")
    target = records[["season", "team", "adj_win_pct"]].copy()
    target["season"] -= 1
    target = target.rename(columns={"adj_win_pct": "target"})
    data = features.merge(target, on=["season", "team"], how="inner")
    data[[*base, *api]] = data[[*base, *api]].fillna(0.0)

    folds, rows = [], []
    seasons = sorted(data.season.unique())
    for test_season in seasons[3:]:
        train = data[data.season < test_season]
        test = data[data.season == test_season]
        if len(train) < 200 or len(test) < 50:
            continue
        pred_base, alpha_base = _fit_predict(train, test, base)
        arm_predictions = {}
        arm_alphas = {}
        for label, columns in {"all_api": api, **api_arms}.items():
            arm_predictions[label], arm_alphas[label] = _fit_predict(
                train, test, [*base, *columns])
        pred_api, alpha_api = arm_predictions["all_api"], arm_alphas["all_api"]
        base_metrics = _metrics(test.target, pred_base)
        api_metrics = _metrics(test.target, pred_api)
        folds.append({
            "feature_season": int(test_season),
            "target_season": int(test_season + 1),
            "existing": base_metrics, "with_api": api_metrics,
            "rmse_difference": api_metrics["rmse"] - base_metrics["rmse"],
            "alpha_existing": alpha_base, "alpha_with_api": alpha_api,
            "api_arms": {label: _metrics(test.target, prediction)
                         for label, prediction in arm_predictions.items()},
            "api_arm_alphas": arm_alphas,
        })
        row = pd.DataFrame({
            "feature_season": test_season, "target": test.target.to_numpy(),
            "team": test.team.to_numpy(),
            "existing": pred_base, "with_api": pred_api,
        })
        for label, prediction in arm_predictions.items():
            row[label] = prediction
        rows.append(row)
        print(f"{test_season}->{test_season + 1}: "
              f"existing {base_metrics['rmse']:.4f}/{base_metrics['correlation']:.3f}  "
              f"+API {api_metrics['rmse']:.4f}/{api_metrics['correlation']:.3f}")

    pred = pd.concat(rows, ignore_index=True)
    result = {
        "contract": "season N player facets predict season N+1 adjusted win pct",
        "existing_features": base,
        "api_features": api,
        "folds": folds,
        "pooled_existing": _metrics(pred.target, pred.existing),
        "pooled_with_api": _metrics(pred.target, pred.with_api),
        "pooled_api_arms": {label: _metrics(pred.target, pred[label])
                            for label in ["all_api", *api_arms]},
        "team_cluster_bootstrap": _team_cluster_bootstrap(
            pred, "with_api", "existing"),
    }
    result["pooled_rmse_difference"] = (
        result["pooled_with_api"]["rmse"] - result["pooled_existing"]["rmse"])
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nexisting: {result['pooled_existing']}")
    print(f"with API: {result['pooled_with_api']}")
    for label, metrics in result["pooled_api_arms"].items():
        print(f"{label:13s}: {metrics}")
    print(f"bootstrap: {result['team_cluster_bootstrap']}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
