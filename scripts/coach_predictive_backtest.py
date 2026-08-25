"""Leakage-safe coach mean and uncertainty-channel experiments.

Coach effects are rebuilt for each season from completed outcomes through N-1. The
same v4 expanding selection, tuning, weekly replay, and metrics are reused.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import ARTIFACTS, GAME_YEARS, OPP_ADJ_ALPHA, UNCERTAINTY_LAMBDA
from scripts import v4_backtest as BT
from scripts.coach_effects import decomposition_frame
from scripts.train import load_bundle
from src import features as F
from src import matchup as MU
from src import oppadj as OA
from src import v4 as V4
from src.data import coaches, pff, war


OUT_JSON = ARTIFACTS / "coach_predictive_backtest.json"
OUT_CSV = ARTIFACTS / "coach_predictive_backtest_predictions.csv"
CORE = ["O", "D", "talent", "returning"]
MEAN = ["hc_prior_effect", "hc_has_prior", "hc_tenure_year", "hc_first_year",
        "hc_effect_delta"]
UNITS = ["hc_prior_offense", "hc_prior_defense", "hc_has_prior",
         "hc_tenure_year", "hc_first_year", "hc_effect_delta"]
INTERACTIONS = [*UNITS, "hc_effect_x_talent", "hc_effect_x_returning",
                "hc_first_year_x_talent", "hc_first_year_x_returning"]
SPECS = {
    "clean_core": ("base", CORE),
    "coach_mean": ("base", [*CORE, *MEAN]),
    "coach_units": ("base", [*CORE, *UNITS]),
    "coach_interactions": ("base", [*CORE, *INTERACTIONS]),
    "flat_uncertainty": ("uncertainty_0", CORE),
    "coach_uncertainty_10": ("uncertainty_10", CORE),
    "coach_uncertainty_20": ("uncertainty_20", CORE),
    "coach_uncertainty_30": ("uncertainty_30", CORE),
}


def _corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else float("nan")


def _coach_adjusted(frame: pd.DataFrame, first_year: pd.Series,
                    b_o: float, b_d: float, gamma: float) -> pd.DataFrame:
    """Mean-preserving heterogeneous shrink: first-year staffs regress more."""
    out = frame.copy()
    flag = first_year.reindex(out.index).fillna(0.0).astype(float)
    # Effective lambdas average approximately the shipping lambda. lam=1 in the
    # algebra below avoids team_frame's u clipping from erasing the upward shock.
    effective = (UNCERTAINTY_LAMBDA + gamma * (flag - flag.mean())).clip(0, 1)
    out["O"] = (1 - effective) * out.O + effective * b_o * out.talent
    out["D"] = (1 - effective) * out.D + effective * b_d * out.talent
    out.attrs["effective_lambda_mean"] = float(effective.mean())
    out.attrs["effective_lambda_first_year"] = (
        float(effective[flag == 1].mean()) if (flag == 1).any() else None)
    return out


def build_frames():
    std, talent, returning, games, _ = load_bundle()
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    pff_lag = pff.build_lagged_team_talent()
    indices = {year: series.index for year, series in talent.items()}
    war_lag = war.lagged_team_talent(indices)
    projected_war = war.projected_team_talent(indices)
    outcome = decomposition_frame()
    assignment = coaches.preseason_head_coaches()
    by_year = F.leakage_safe_coach_features(outcome, assignment, GAME_YEARS)

    frame_sets = {"base": {}}
    for gamma in (0.0, .10, .20, .30):
        frame_sets[f"uncertainty_{int(gamma * 100)}"] = {}
    temporal = {}
    for year in GAME_YEARS:
        frame = V4.build_frame(year, std, talent, returning, od, pff_lag,
                               war_lag, granular=True)
        if frame is None:
            continue
        frame = F.attach_coach_features(frame, by_year[year])
        wp = projected_war.get(year, pd.Series(dtype=float)).reindex(frame.index)
        frame["war_projected"] = wp.fillna(0.0)
        frame_sets["base"][year] = frame
        temporal[str(year)] = {
            "max_outcome_season": by_year[year].attrs["max_outcome_season"],
            "preseason_assignment_coverage": float(
                assignment[assignment.season == year].resolved_preseason.mean()),
        }
        train_years = [value for value in GAME_YEARS if value < year]
        if train_years:
            b_o, b_d = MU.fit_talent_od_slopes(train_years, std, talent,
                                                od_by_year=od)
        else:
            b_o = b_d = 0.0
        for gamma in (0.0, .10, .20, .30):
            key = f"uncertainty_{int(gamma * 100)}"
            frame_sets[key][year] = _coach_adjusted(
                frame, frame.hc_first_year, b_o, b_d, gamma)
    return frame_sets, games, od, temporal


def _bootstrap(frame: pd.DataFrame, left: str, right: str, draws=5000,
               seed=20260825):
    d = frame.dropna(subset=[left, right]).copy()
    d["delta"] = (d[left] - d.y) ** 2 - (d[right] - d.y) ** 2
    blocks = [g.delta.to_numpy(float) for _, g in
              d.groupby(["season", "week"], sort=True)]
    rng = np.random.default_rng(seed)
    samples = np.asarray([
        np.concatenate([blocks[index] for index in
                        rng.integers(0, len(blocks), len(blocks))]).mean()
        for _ in range(draws)])
    return {"difference": float(d.delta.mean()),
            "ci95": [float(value) for value in np.quantile(samples, [.025, .975])],
            "probability_left_better": float(np.mean(samples < 0)),
            "n_games": int(len(d)), "blocks": int(len(blocks))}


def _rating_scores(frame_sets, od):
    result = {}
    for key, frames in frame_sets.items():
        rows = []
        for year, frame in frames.items():
            if year not in od:
                continue
            common = frame.index.intersection(od[year].index)
            row = {"season": year, "n": int(len(common))}
            for side in ("O", "D"):
                expected = frame.loc[common, side].to_numpy(float)
                realized = od[year].loc[common, side].to_numpy(float)
                row[f"{side}_r"] = _corr(expected, realized)
                row[f"{side}_rmse"] = float(np.sqrt(np.mean((expected-realized)**2)))
            row["mean_r"] = (row["O_r"] + row["D_r"]) / 2
            rows.append(row)
        result[key] = {"folds": rows,
                       "mean_r": float(np.mean([row["mean_r"] for row in rows])),
                       "mean_rmse": float(np.mean([(row["O_rmse"] + row["D_rmse"])/2
                                                   for row in rows]))}
    return result


def main():
    frame_sets, games, od, temporal = build_frames()
    candidate_columns = {name: columns for name, (_, columns) in SPECS.items()}
    parts = {name: V4.assemble(GAME_YEARS, frame_sets[frame_key], games, columns)
             for name, (frame_key, columns) in SPECS.items()}
    folds, prediction_rows = [], []
    for test in (2022, 2023, 2024, 2025):
        pool = [year for year in GAME_YEARS if year < test]
        selected, selection_scores = BT.choose_candidate(parts, pool,
                                                          candidate_columns)
        fold = {"season": test, "selected": selected,
                "selection_scores": selection_scores, "candidates": {}}
        meta = None
        for name, (frame_key, columns) in SPECS.items():
            knobs, tuning = BT.tune(parts[name], pool, columns)
            frames = frame_sets[frame_key]
            k, blend, dynamic_tuning = BT.tune_dynamic(
                parts[name], frames, pool, columns, knobs)
            model, static, margin = BT.fit_predict(parts[name], pool, test,
                                                   columns, knobs)
            dynamic, _ = BT.dynamic_predictions(model, frames[test],
                                                 parts[name][test], k, blend)
            fold["candidates"][name] = {
                "static": BT.metric(parts[name][test][1], static,
                                    parts[name][test][3], margin),
                "dynamic": BT.metric(parts[name][test][1], dynamic),
                "knobs": knobs, "dynamic_k": k, "dynamic_blend": blend,
                "tuning": tuning, "dynamic_tuning": dynamic_tuning,
            }
            if meta is None:
                meta = parts[name][test][4].copy()
                meta["season"], meta["y"] = test, parts[name][test][1]
            meta[f"p_static_{name}"] = static
            meta[f"p_dynamic_{name}"] = dynamic
        folds.append(fold)
        prediction_rows.append(meta)
        print(test, selected, fold["candidates"][selected]["static"]["brier"],
              fold["candidates"][selected]["dynamic"]["brier"])

    predictions = pd.concat(prediction_rows, ignore_index=True)
    pooled, comparisons = {}, {}
    for name in SPECS:
        pooled[name] = {
            "static": BT.metric(predictions.y, predictions[f"p_static_{name}"]),
            "dynamic": BT.metric(predictions.y, predictions[f"p_dynamic_{name}"]),
        }
        if name != "clean_core":
            comparisons[name] = {
                "static": _bootstrap(predictions, f"p_static_{name}",
                                     "p_static_clean_core"),
                "dynamic": _bootstrap(predictions, f"p_dynamic_{name}",
                                      "p_dynamic_clean_core"),
            }

    base = frame_sets["base"]
    training = pd.concat([frame.assign(season=year) for year, frame in base.items()])
    correlations = {
        "hc_first_year_returning": _corr(training.hc_first_year, training.returning),
        "hc_first_year_talent": _corr(training.hc_first_year, training.talent),
    }
    uncertainty_incremental = {}
    for name in ("coach_uncertainty_10", "coach_uncertainty_20",
                 "coach_uncertainty_30"):
        uncertainty_incremental[name] = {
            "static": _bootstrap(predictions, f"p_static_{name}",
                                 "p_static_flat_uncertainty"),
            "dynamic": _bootstrap(predictions, f"p_dynamic_{name}",
                                  "p_dynamic_flat_uncertainty"),
        }
    result = {
        "contract": "season N coach effects use completed outcome seasons <= N-1",
        "preseason_assignment": (
            "retain prior incumbent; exclude post-August hires; zero unresolved "
            "multi-coach rows; never select on season-N games"),
        "selection_rule": {"reference": "clean_core",
                           "minimum_brier_gain": BT.SELECTION_MIN_GAIN},
        "candidate_specs": SPECS, "temporal_audit": temporal,
        "correlations": correlations, "rating_target": _rating_scores(frame_sets, od),
        "folds": folds, "pooled": pooled,
        "paired_season_week_bootstrap": comparisons,
        "coach_uncertainty_incremental_vs_matched_flat": uncertainty_incremental,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    predictions.to_csv(OUT_CSV, index=False)
    print(f"-> {OUT_JSON}\n-> {OUT_CSV}")


if __name__ == "__main__":
    main()
