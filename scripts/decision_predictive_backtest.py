"""Phase 6 predictive test: do play-call decision profiles forecast games?

The descriptive script establishes that pass rate over expected and fourth-down
aggression repeat year to year. Repeatable is not the same as useful. This script
runs them through the same expanding selection, tuning, weekly replay and paired
bootstrap the coach and player-production phases used, against the same clean core
and the same predeclared +.001 Brier bar.

Both halves of the construction are refit inside every fold:

* the league expectation models that define "over expected" see only seasons <= N-1;
* a team's decision profile sees only its own seasons <= N-1.

The coach-carried candidate is the one the team-level lag cannot express. When a
staff changes, the team's own history describes coaches who have left, while the
incoming coach's tendencies travel with them from the previous school.
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
from scripts.coach_predictive_backtest import _bootstrap, _corr
from scripts.train import load_bundle
from src import decisions as D
from src import oppadj as OA
from src import v4 as V4
from src.data import coaches, pff, plays as PLAYS, war


OUT_JSON = ARTIFACTS / "decision_predictive_backtest.json"
OUT_CSV = ARTIFACTS / "decision_predictive_backtest_predictions.csv"
CORE = ["O", "D", "talent", "returning"]
# PROE correlates +.34 with talent, so proe*talent is part quadratic talent. Without
# a curvature control there is no way to tell a decision effect from the model simply
# wanting a nonlinear talent term, and the interaction candidates would take credit
# for both. These columns are the control.
CURVATURE = ["talent_sq", "returning_sq", "talent_x_returning"]
SPECS = {
    "clean_core": CORE,
    "decision_tendency": [*CORE, *D.TENDENCY],
    "decision_full": [*CORE, *D.FULL],
    "decision_coach_carried": [*CORE, *D.TENDENCY, *D.COACH_CARRIED],
    "decision_interactions": [*CORE, *D.TENDENCY, *D.INTERACTIONS],
    "decision_everything": [*CORE, *D.ALL_FEATURES],
    "curvature_core": [*CORE, *CURVATURE],
    "curvature_plus_decisions": [*CORE, *CURVATURE, *D.TENDENCY, *D.INTERACTIONS],
}


def build_frames():
    std, talent, returning, games, _ = load_bundle()
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    pff_lag = pff.build_lagged_team_talent()
    indices = {year: series.index for year, series in talent.items()}
    war_lag = war.lagged_team_talent(indices)
    projected_war = war.projected_team_talent(indices)
    play_frame = PLAYS.load()
    assignment = coaches.preseason_head_coaches()
    profiles = D.lagged_profiles(GAME_YEARS, play_frame, assignment)

    frames, temporal = {}, {}
    for year in GAME_YEARS:
        frame = V4.build_frame(year, std, talent, returning, od, pff_lag,
                               war_lag, granular=True)
        if frame is None:
            continue
        wp = projected_war.get(year, pd.Series(dtype=float)).reindex(frame.index)
        frame["war_projected"] = wp.fillna(0.0)
        profile = profiles[year]
        frame = D.attach_decision_features(frame, profile)
        frame["talent_sq"] = frame.talent ** 2
        frame["returning_sq"] = frame.returning ** 2
        frame["talent_x_returning"] = frame.talent * frame.returning
        frames[year] = frame
        covered = (profile.index.intersection(frame.index)
                   if len(profile) else pd.Index([]))
        temporal[str(year)] = {
            "expectation_seasons": profile.attrs.get("expectation_seasons"),
            "max_play_season": profile.attrs.get("max_play_season"),
            "teams_in_frame": int(len(frame)),
            "teams_with_decision_history": int(len(covered)),
            "decision_coverage": float(len(covered) / len(frame)) if len(frame) else 0.0,
            "coach_history_coverage": (
                float(frame.dec_coach_has_history.mean()) if len(frame) else 0.0),
        }
    return frames, games, od, temporal


def main():
    frames, games, od, temporal = build_frames()
    years = [year for year in GAME_YEARS if year in frames]
    parts = {name: V4.assemble(years, frames, games, columns)
             for name, columns in SPECS.items()}
    folds, prediction_rows = [], []
    for test in (2022, 2023, 2024, 2025):
        pool = [year for year in years if year < test]
        if not pool or test not in frames:
            continue
        selected, selection_scores = BT.choose_candidate(parts, pool, SPECS)
        fold = {"season": test, "selected": selected,
                "selection_scores": selection_scores, "candidates": {}}
        meta = None
        for name, columns in SPECS.items():
            knobs, tuning = BT.tune(parts[name], pool, columns)
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
        print(test, selected,
              round(fold["candidates"][selected]["static"]["brier"], 5),
              round(fold["candidates"][selected]["dynamic"]["brier"], 5), flush=True)

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

    training = pd.concat([frame.assign(season=year)
                          for year, frame in frames.items()])
    correlations = {
        "proe_talent": _corr(training.dec_proe, training.talent),
        "proe_returning": _corr(training.dec_proe, training.returning),
        "fourth_go_talent": _corr(training.dec_fourth_go, training.talent),
        "proe_fourth_go": _corr(training.dec_proe, training.dec_fourth_go),
    }
    result = {
        "contract": ("season N uses league expectation models and team decision "
                     "profiles built only from plays in seasons <= N-1"),
        "selection_rule": {"reference": "clean_core",
                           "minimum_brier_gain": BT.SELECTION_MIN_GAIN},
        "candidate_specs": SPECS,
        "temporal_audit": temporal,
        "feature_correlations": correlations,
        "folds": folds,
        "pooled": pooled,
        "paired_season_week_bootstrap": comparisons,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    predictions.to_csv(OUT_CSV, index=False)

    print("\n--- pooled 2022-25 ---")
    base_static = pooled["clean_core"]["static"]["brier"]
    base_online = pooled["clean_core"]["dynamic"]["brier"]
    for name in SPECS:
        s = pooled[name]["static"]["brier"]
        o = pooled[name]["dynamic"]["brier"]
        print(f"{name:24s} static {s:.5f} ({s - base_static:+.5f})"
              f"   online {o:.5f} ({o - base_online:+.5f})")
    print(f"\n-> {OUT_JSON}\n-> {OUT_CSV}")
    return result


if __name__ == "__main__":
    main()
