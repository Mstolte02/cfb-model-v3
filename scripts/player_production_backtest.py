"""Forward game-prediction ablation for projected CFB player production.

The model fitting, tuning, forward-selection rule, weekly replay, and metrics are all
imported from ``scripts.v4_backtest``. This script supplies only the new precomputed,
cross-fitted team features and research candidates.

Run: ``python -m scripts.player_production_backtest``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import ARTIFACTS, GAME_YEARS, OPP_ADJ_ALPHA
from scripts import v4_backtest as BT
from scripts.train import load_bundle
from src import oppadj as OA
from src import v4 as V4
from src.data import pff, player_production, war


OUT_JSON = ARTIFACTS / "player_production_backtest.json"
OUT_CSV = ARTIFACTS / "player_production_backtest_predictions.csv"

CORE = ["O", "D", "talent", "returning"]
CANDIDATES = {
    "clean_core": CORE,
    "war_projected": [*CORE, "war_projected"],
    "production_direct": [*CORE, "player_prod_off", "player_prod_def"],
    "production_war": [*CORE, "player_prod_war"],
    "war_plus_direct": [*CORE, "war_projected", "player_prod_off",
                        "player_prod_def"],
    "war_plus_production_war": [*CORE, "war_projected", "player_prod_war"],
    "production_only": ["player_prod_off", "player_prod_def"],
}


def _bootstrap_delta(frame: pd.DataFrame, left: str, right: str,
                     draws: int = 5000, seed: int = 20260825):
    d = frame.dropna(subset=[left, right]).copy()
    d["loss_diff"] = (d[left] - d.y) ** 2 - (d[right] - d.y) ** 2
    blocks = [g.loss_diff.to_numpy(float) for _, g in
              d.groupby(["season", "week"], sort=True, dropna=False)]
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for index in range(draws):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        samples[index] = np.concatenate([blocks[i] for i in selected]).mean()
    return {
        "estimand": f"Brier({left}) - Brier({right})",
        "difference": float(d.loss_diff.mean()),
        "ci95": [float(value) for value in np.quantile(samples, [.025, .975])],
        "probability_left_better": float(np.mean(samples < 0)),
        "n_games": int(len(d)), "blocks": int(len(blocks)),
        "draws": draws, "seed": seed,
    }


def build_frames():
    std, talent, returning, games, _ = load_bundle()
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    pff_lag = pff.build_lagged_team_talent()
    indices = {year: series.index for year, series in talent.items()}
    war_lag = war.lagged_team_talent(indices)
    projected_war = war.projected_team_talent(indices)
    production = player_production.by_year(indices)

    frames = {}
    coverage = {}
    for year in GAME_YEARS:
        frame = V4.build_frame(year, std, talent, returning, od, pff_lag,
                               war_lag, granular=True)
        if frame is None:
            continue
        wp = projected_war.get(year, pd.Series(dtype=float)).reindex(frame.index)
        frame["war_projected"] = wp.fillna(0.0)
        pc = production.get(year, pd.DataFrame(
            columns=player_production.FEATURES, index=frame.index)).reindex(frame.index)
        coverage[str(year)] = {
            column: float(pc[column].notna().mean())
            for column in player_production.FEATURES
        }
        for column in player_production.FEATURES:
            frame[column] = pc[column].fillna(0.0)
        frames[year] = frame
    return frames, games, coverage


def main():
    frames, games, coverage = build_frames()
    parts = {name: V4.assemble(GAME_YEARS, frames, games, columns)
             for name, columns in CANDIDATES.items()}
    folds, prediction_rows = [], []
    for test in [2022, 2023, 2024, 2025]:
        pool = [year for year in GAME_YEARS if year < test]
        selected, selection_scores = BT.choose_candidate(parts, pool, CANDIDATES)
        fold = {"season": test, "selected": selected,
                "selection_scores": selection_scores, "candidates": {}}
        meta = None
        for name, columns in CANDIDATES.items():
            knobs, tuning = BT.tune(parts[name], pool, columns)
            k, blend, dynamic_tuning = BT.tune_dynamic(
                parts[name], frames, pool, columns, knobs)
            model, static_p, margin_p = BT.fit_predict(
                parts[name], pool, test, columns, knobs)
            dynamic_p, _ = BT.dynamic_predictions(
                model, frames[test], parts[name][test], k, blend)
            fold["candidates"][name] = {
                "features": columns, "knobs": knobs,
                "dynamic_k": k, "dynamic_blend": blend,
                "static": BT.metric(parts[name][test][1], static_p,
                                    parts[name][test][3], margin_p),
                "dynamic": BT.metric(parts[name][test][1], dynamic_p),
                "tuning": tuning, "dynamic_tuning": dynamic_tuning,
            }
            if meta is None:
                meta = parts[name][test][4].copy()
                meta["season"] = test
                meta["y"] = parts[name][test][1]
            meta[f"p_static_{name}"] = static_p
            meta[f"p_dynamic_{name}"] = dynamic_p
        folds.append(fold)
        prediction_rows.append(meta)
        chosen = fold["candidates"][selected]
        print(f"{test}: selected {selected:<23} static "
              f"{chosen['static']['brier']:.4f} dynamic "
              f"{chosen['dynamic']['brier']:.4f}")

    predictions = pd.concat(prediction_rows, ignore_index=True)
    pooled = {}
    for name in CANDIDATES:
        pooled[name] = {
            "static": BT.metric(predictions.y, predictions[f"p_static_{name}"]),
            "dynamic": BT.metric(predictions.y, predictions[f"p_dynamic_{name}"]),
        }
    comparisons = {}
    for name in CANDIDATES:
        if name == "clean_core":
            continue
        comparisons[name] = {
            "static": _bootstrap_delta(
                predictions, f"p_static_{name}", "p_static_clean_core"),
            "dynamic": _bootstrap_delta(
                predictions, f"p_dynamic_{name}", "p_dynamic_clean_core"),
        }
    incremental = {}
    for name in ("production_direct", "production_war", "war_plus_direct",
                 "war_plus_production_war"):
        incremental[name] = {
            "static": _bootstrap_delta(
                predictions, f"p_static_{name}", "p_static_war_projected"),
            "dynamic": _bootstrap_delta(
                predictions, f"p_dynamic_{name}", "p_dynamic_war_projected"),
        }

    result = {
        "contract": (
            "strict expanding game folds; player production forecasts are themselves "
            "cross-fitted and use only completed seasons <= N-1"
        ),
        "selection_rule": {
            "reference": "clean_core",
            "minimum_extension_brier_gain": BT.SELECTION_MIN_GAIN,
        },
        "incremental_adoption_rule": {
            "reference": "war_projected",
            "reason": "this is the player feature already selected by production v4",
            "minimum_brier_gain": BT.SELECTION_MIN_GAIN,
            "must_clear_in_consecutive_selection_windows": True,
        },
        "candidates": CANDIDATES, "coverage": coverage,
        "folds": folds, "pooled": pooled,
        "paired_season_week_bootstrap_vs_clean_core": comparisons,
        "paired_season_week_bootstrap_vs_production_war_baseline": incremental,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    predictions.to_csv(OUT_CSV, index=False)
    print(f"-> {OUT_JSON}\n-> {OUT_CSV}")


if __name__ == "__main__":
    main()
