"""How good a game predictor is WAR on its own?

The WAR model is a large piece of work that currently enters the win model as one
input among six, blended at a fraction of one of them. That makes it easy to lose
track of what it is actually worth as a forecast. This runs it as the whole model.

Candidates, all through the same expanding selection, tuning, weekly replay and
paired bootstrap the other phases use:

* `war_only`          - projected roster WAR and nothing else.
* `war_lagged_only`   - last season's realised roster WAR and nothing else.
* `war_pair`          - both WAR columns, still no ratings and no talent.
* `strength_only`     - the O/D power rating alone, as a same-size reference.
* `talent_only`       - recruiting talent alone, the other same-size reference.
* `clean_core`        - the four-column baseline every phase reports against.
* `core_minus_war`    - the core, to show what WAR adds on top rather than alone.

The single-input candidates are the point: a fair read of WAR needs to know whether
one column of roster quality beats one column of last year's results, not just
whether the full model beats a stripped one.
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
from src.data import pff, war


OUT_JSON = ARTIFACTS / "war_only_backtest.json"
OUT_CSV = ARTIFACTS / "war_only_backtest_predictions.csv"

SPECS = {
    "war_only": ["war_projected"],
    "war_lagged_only": ["war_lag"],
    "war_pair": ["war_projected", "war_lag"],
    "strength_only": ["strength"],
    "talent_only": ["talent"],
    "clean_core": ["O", "D", "talent", "returning"],
    "core_plus_war": ["O", "D", "talent", "returning", "war_projected"],
}


def _bootstrap(frame: pd.DataFrame, left: str, right: str, draws=5000, seed=20260825):
    data = frame.dropna(subset=[left, right]).copy()
    data["delta"] = (data[left] - data.y) ** 2 - (data[right] - data.y) ** 2
    blocks = [g.delta.to_numpy(float) for _, g in
              data.groupby(["season", "week"], sort=True)]
    rng = np.random.default_rng(seed)
    samples = np.asarray([
        np.concatenate([blocks[i] for i in
                        rng.integers(0, len(blocks), len(blocks))]).mean()
        for _ in range(draws)])
    return {"difference": float(data.delta.mean()),
            "ci95": [float(v) for v in np.quantile(samples, [.025, .975])],
            "probability_left_better": float(np.mean(samples < 0))}


def build_frames():
    std, talent, ret, games, _ = load_bundle()
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    pff_lag = pff.build_lagged_team_talent()
    indices = {y: s.index for y, s in talent.items()}
    war_lag = war.lagged_team_talent(indices)
    war_projected = war.projected_team_talent(indices)
    frames = {}
    for year in GAME_YEARS:
        frame = V4.build_frame(year, std, talent, ret, od, pff_lag, war_lag,
                               granular=True)
        if frame is None:
            continue
        wp = war_projected.get(year, pd.Series(dtype=float)).reindex(frame.index)
        frame["war_projected"] = wp.fillna(0.0)
        frame["strength"] = frame.O + frame.D
        frames[year] = frame
    return frames, games


def main():
    frames, games = build_frames()
    years = [y for y in GAME_YEARS if y in frames]
    parts = {name: V4.assemble(years, frames, games, cols)
             for name, cols in SPECS.items()}
    folds, rows = [], []
    for test in (2022, 2023, 2024, 2025):
        pool = [y for y in years if y < test]
        if not pool:
            continue
        fold = {"season": test, "candidates": {}}
        meta = None
        for name, cols in SPECS.items():
            knobs, _ = BT.tune(parts[name], pool, cols)
            k, blend, _ = BT.tune_dynamic(parts[name], frames, pool, cols, knobs)
            model, static, margin = BT.fit_predict(parts[name], pool, test, cols,
                                                    knobs)
            dynamic, _ = BT.dynamic_predictions(model, frames[test],
                                                 parts[name][test], k, blend)
            fold["candidates"][name] = {
                "static": BT.metric(parts[name][test][1], static,
                                    parts[name][test][3], margin),
                "dynamic": BT.metric(parts[name][test][1], dynamic)}
            if meta is None:
                meta = parts[name][test][4].copy()
                meta["season"], meta["y"] = test, parts[name][test][1]
            meta[f"p_static_{name}"] = static
            meta[f"p_dynamic_{name}"] = dynamic
        folds.append(fold)
        rows.append(meta)
        print(f"{test} done", flush=True)

    predictions = pd.concat(rows, ignore_index=True)
    pooled = {n: {"static": BT.metric(predictions.y, predictions[f"p_static_{n}"]),
                  "dynamic": BT.metric(predictions.y, predictions[f"p_dynamic_{n}"])}
              for n in SPECS}
    vs_core = {n: {"static": _bootstrap(predictions, f"p_static_{n}",
                                        "p_static_clean_core"),
                   "dynamic": _bootstrap(predictions, f"p_dynamic_{n}",
                                         "p_dynamic_clean_core")}
               for n in SPECS if n != "clean_core"}
    result = {"specs": SPECS, "folds": folds, "pooled": pooled,
              "vs_clean_core": vs_core}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    predictions.to_csv(OUT_CSV, index=False)

    print(f"\n{'candidate':20s} {'inputs':>7} {'static':>9} {'online':>9} "
          f"{'acc':>7}")
    for name in SPECS:
        b = pooled[name]
        print(f"{name:20s} {len(SPECS[name]):>7} {b['static']['brier']:>9.5f} "
              f"{b['dynamic']['brier']:>9.5f} "
              f"{b['static'].get('accuracy', float('nan')):>7.3f}")
    print(f"\n-> {OUT_JSON}")
    return result


if __name__ == "__main__":
    main()
