"""Test recursive result processing beside the locked historical weekly v4.

Run ``scripts.v4_backtest`` first. This script reads its exact outer-fold
predictions, constructs recursive features only from earlier weeks in the same
season, and fits a ridge-logistic correction with the v4 logit held as an offset.
Thus the test asks whether the new signal adds information after the shipping model,
not whether it can replace or recalibrate that model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import ARTIFACTS
from scripts.idempotence_backtest import fixed_point, paired_week_bootstrap
from scripts.train import load_bundle
from scripts.v4_backtest import metric
from src.context import OffsetLogit


PRED_PATH = ARTIFACTS / "v4_backtest_predictions.csv"
BASELINE_PATH = ARTIFACTS / "v4_backtest.json"
OUT_JSON = ARTIFACTS / "idempotence_v4_backtest.json"
OUT_CSV = ARTIFACTS / "idempotence_v4_backtest_predictions.csv"
KEYS = ["season", "week", "home_team", "away_team"]
MIN_FEATURE_GAMES = 3
PENALTY = 20.0


def _values(fp, values):
    return dict(zip(fp.teams, values))


def build_features(pred: pd.DataFrame,
                   games_by_year: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """One feature row per locked v4 prediction, frozen at start of week."""
    rows = []
    for season, season_pred in pred.groupby("season", sort=True):
        season = int(season)
        teams = sorted(set(season_pred.home_team) | set(season_pred.away_team))
        members = set(teams)
        games = games_by_year[season]
        games = games[
            games.home_team.isin(members) & games.away_team.isin(members)
        ].copy()
        history = games.iloc[0:0]
        for week, slate in season_pred.groupby("week", sort=True):
            fp = fixed_point(history, teams)
            first = _values(fp, fp.first)
            second = _values(fp, fp.second - fp.first)
            recursive = _values(fp, fp.fixed - fp.first)
            agrees = np.sign(fp.first) == np.sign(fp.fixed)
            stable = _values(fp, np.where(
                agrees,
                np.sign(fp.fixed) * np.minimum(np.abs(fp.first), np.abs(fp.fixed)),
                0.0))
            counts = _values(fp, fp.games)
            for game in slate.itertuples(index=False):
                home, away = game.home_team, game.away_team
                enough = min(counts[home], counts[away]) >= MIN_FEATURE_GAMES
                rows.append({
                    "season": season, "week": week,
                    "home_team": home, "away_team": away,
                    "second_pass_correction_gap": (
                        second[home] - second[away] if enough else 0.0),
                    "recursive_correction_gap": (
                        recursive[home] - recursive[away] if enough else 0.0),
                    "stable_strength_gap": (
                        stable[home] - stable[away] if enough else 0.0),
                    "feature_available": bool(enough),
                })
            completed = games[games.week == week]
            history = pd.concat([history, completed], ignore_index=True)
    return pd.DataFrame(rows)


def score_family(data: pd.DataFrame, name: str, features: list[str]):
    folds, pieces = [], []
    seasons = sorted(int(s) for s in data.season.unique())
    for test_year in seasons[1:]:
        train = data[data.season < test_year]
        test = data[data.season == test_year].copy()
        model = OffsetLogit(features, penalty=PENALTY).fit(train)
        test["p_adjusted"] = model.predict(test)
        change = (metric(test.y, test.p_adjusted)["brier"] -
                  metric(test.y, test.p_dynamic)["brier"])
        folds.append({
            "season": test_year, "n": int(len(test)),
            "brier_change": float(change),
            "coefficients": model.payload()["coef"],
        })
        test["family"] = name
        pieces.append(test)
    out = pd.concat(pieces, ignore_index=True)
    change = (metric(out.y, out.p_adjusted)["brier"] -
              metric(out.y, out.p_dynamic)["brier"])
    return {
        "name": name, "features": features, "n": int(len(out)),
        "pooled_brier_change": float(change), "folds": folds,
        "paired_week_bootstrap": paired_week_bootstrap(
            out, left="p_adjusted", right="p_dynamic"),
    }, out


def main():
    pred = pd.read_csv(PRED_PATH)
    baseline = json.loads(BASELINE_PATH.read_text())
    _, _, _, games, _ = load_bundle()
    features = build_features(pred, games)
    data = pred.merge(features, on=KEYS, how="left", validate="one_to_one")
    feature_names = ["second_pass_correction_gap", "recursive_correction_gap",
                     "stable_strength_gap"]
    if data[feature_names].isna().any().any():
        raise RuntimeError("recursive feature join missed locked v4 games")

    families = {
        "second_pass_correction": ["second_pass_correction_gap"],
        "recursive_correction": ["recursive_correction_gap"],
        "stable_strength": ["stable_strength_gap"],
        "recursive_plus_stable": ["recursive_correction_gap",
                                   "stable_strength_gap"],
    }
    results, predictions = [], []
    for name, columns in families.items():
        result, frame = score_family(data, name, columns)
        results.append(result)
        predictions.append(frame[[*KEYS, "family", "y", "p_dynamic",
                                  "p_adjusted", "feature_available"]])
        print(f"{name:<30} Brier {result['pooled_brier_change']:+.6f}")

    payload = {
        "contract": ("locked outer-fold weekly v4; season N overlay fit only on "
                     "earlier outer-fold predictions; features use prior weeks"),
        "baseline_pooled_dynamic": baseline["pooled_dynamic"],
        "baseline_same_game_benchmark": baseline["same_game_benchmark"],
        "baseline_folds": [{"season": f["season"], "selected": f["selected"],
                            "features": f["features"],
                            "dynamic_k": f["dynamic_k"],
                            "dynamic_blend": f["dynamic_blend"]}
                           for f in baseline["folds"]],
        "overlay_test_seasons": sorted(int(s) for s in data.season.unique())[1:],
        "penalty": PENALTY,
        "families": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    pd.concat(predictions, ignore_index=True).to_csv(OUT_CSV, index=False)
    print(f"baseline dynamic Brier: {baseline['pooled_dynamic']['brier']:.6f}")
    print(f"-> {OUT_JSON}\n-> {OUT_CSV}")


if __name__ == "__main__":
    main()
