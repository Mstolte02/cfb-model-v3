"""Forward test of lagged PFF API team features.

This intentionally leaves the production candidate set untouched.  Each PFF arm
must clear the same 0.001 Brier selection threshold as every other extension, and
the dynamic score updater remains identical to the shipping model.

Run after staging 2020-2025 team tables:
    python -m scripts.sync_pff_api --seasons 2020-2025 --team-stats-only
    python -m scripts.pff_api_backtest
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import ARTIFACTS, GAME_YEARS, OPP_ADJ_ALPHA
from scripts import v4_backtest as BT
from scripts.train import load_bundle
from src import oppadj as OA
from src import talent_sources as TS
from src import v4 as V4
from src.data import pff, pff_team, war


OUT_JSON = ARTIFACTS / "pff_api_backtest.json"
OUT_CSV = ARTIFACTS / "pff_api_backtest_predictions.csv"

BASE = ["O", "D", "talent", "returning"]
CURRENT = ["O", "D", "talent", "returning", "war_projected",
           *TS.GROUPS, *TS.PORTAL_RATED]
CANDIDATES = {
    "clean_core": BASE,
    "current_player_layer": CURRENT,
    "core_pff_outcomes": [*BASE, *pff_team.OUTCOME_FEATURES],
    "core_pff_process": [*BASE, *pff_team.PROCESS_FEATURES],
    "core_pff_style": [*BASE, *pff_team.STYLE_FEATURES],
    "core_pff_all": [*BASE, *pff_team.ALL_EXPERIMENT_FEATURES],
    "current_plus_pff_process": [*CURRENT, *pff_team.PROCESS_FEATURES],
}


def main() -> None:
    std, talent, returning, games, _ = load_bundle()
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    pff_lag = pff.build_lagged_team_talent()
    indices = {year: values.index for year, values in talent.items()}
    war_lag = war.lagged_team_talent(indices)
    war_projected = war.projected_team_talent(indices)
    portal = TS.portal_features(GAME_YEARS)
    groups = TS.group_features(GAME_YEARS)
    team_api = pff_team.build_lagged(GAME_YEARS)

    frames = {}
    for year in GAME_YEARS:
        frame = V4.build_frame(year, std, talent, returning, od, pff_lag,
                               war_lag, granular=True)
        if frame is None:
            continue
        projected = war_projected.get(year, pd.Series(dtype=float)).reindex(frame.index)
        frame["war_projected"] = projected.fillna(0.0)
        TS.attach(frame, portal[year], groups[year])
        frames[year] = pff_team.attach(frame, team_api[year])

    all_parts = {name: V4.assemble(GAME_YEARS, frames, games, columns)
                 for name, columns in CANDIDATES.items()}
    folds, predictions = [], []
    for test in [2022, 2023, 2024, 2025]:
        pool = [year for year in GAME_YEARS if year < test]
        selected, scores = BT.choose_candidate(all_parts, pool, CANDIDATES)
        columns, parts = CANDIDATES[selected], all_parts[selected]
        knobs, tuning = BT.tune(parts, pool, columns)
        k, blend, dynamic_tuning = BT.tune_dynamic(
            parts, frames, pool, columns, knobs)
        model, static, margin = BT.fit_predict(parts, pool, test, columns, knobs)
        dynamic, dynamic_only = BT.dynamic_predictions(
            model, frames[test], parts[test], k, blend)
        fold = {
            "season": test, "selected": selected, "features": columns,
            "candidate_scores": scores, "knobs": knobs,
            "dynamic_k": k, "dynamic_blend": blend,
            "static": BT.metric(parts[test][1], static, parts[test][3], margin),
            "dynamic": BT.metric(parts[test][1], dynamic),
            "tuning": tuning, "dynamic_tuning": dynamic_tuning,
            "pff_team_coverage": frames[test].attrs["pff_team_coverage"],
        }
        folds.append(fold)
        meta = parts[test][4].copy()
        meta["season"] = test
        meta["y"] = parts[test][1]
        meta["p_static"] = static
        meta["p_dynamic"] = dynamic
        meta["p_dynamic_only"] = dynamic_only
        meta["selected"] = selected
        predictions.append(meta)
        print(f"{test}: {selected:<26} static={fold['static']['brier']:.4f} "
              f"dynamic={fold['dynamic']['brier']:.4f} "
              f"coverage={fold['pff_team_coverage']:.1%}")

    pred = pd.concat(predictions, ignore_index=True)
    result = {
        "contract": "season N uses only completed PFF season N-1 team tables",
        "selection_rule": {"reference": "clean_core",
                           "minimum_extension_brier_gain": BT.SELECTION_MIN_GAIN},
        "candidate_features": CANDIDATES,
        "folds": folds,
        "pooled_static": BT.metric(pred.y, pred.p_static),
        "pooled_dynamic": BT.metric(pred.y, pred.p_dynamic),
        "selection_counts": pred.selected.value_counts().to_dict(),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pred.to_csv(OUT_CSV, index=False)
    print(f"\npooled static : {result['pooled_static']}")
    print(f"pooled dynamic: {result['pooled_dynamic']}")
    print(f"-> {OUT_JSON}\n-> {OUT_CSV}")


if __name__ == "__main__":
    main()
