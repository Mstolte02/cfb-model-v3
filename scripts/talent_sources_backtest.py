"""Are there talent inputs the model is not using?

Team talent today is three axes blended into one number: the CFBD recruiting
composite, PFF roster-aware grades, and roster WAR. Two things CFBD publishes are not
in that blend at all, and both describe roster quality that the existing axes cannot
see:

* **The transfer portal.** A team that lost four rated starters and replaced them with
  three looks identical to the recruiting composite as one that stood still. Portal
  entries carry origin, destination and a recruit rating, so incoming and outgoing
  talent can be priced separately.
* **Recruiting by position group.** The shipping talent figure is one number per team.
  The same composite split by quarterback, line, skill and secondary makes it possible
  to say a roster is stacked in the wrong places.

Both enter as team-level features the reciprocal architecture differences itself, and
both are lagged: season N sees the portal class that arrived for N and recruiting
classes through N, which are preseason facts, never season-N results.

Candidates are additive against the clean core so the question is what each one adds,
and the answer is scored on the same +.001 bar as every other phase.
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
from src import talent_sources as TS
from src import v4 as V4
from src.data import pff, war

# The builders live in src/talent_sources.py so the production trainer attaches the
# same columns this script measured; re-exported here for the experiment scripts
# that already import them from this module.
GROUPS = TS.GROUPS
PORTAL_RATED = TS.PORTAL_RATED
group_features = TS.group_features
portal_features = TS.portal_features


OUT_JSON = ARTIFACTS / "talent_sources_backtest.json"
CORE = ["O", "D", "talent", "returning"]
PORTAL = TS.PORTAL
SPECS = {
    "clean_core": CORE,
    "core_portal": [*CORE, *PORTAL],
    "core_portal_rated": [*CORE, *PORTAL_RATED],
    "core_groups": [*CORE, *GROUPS],
    "core_both": [*CORE, *PORTAL, *GROUPS],
    "core_groups_portal_rated": [*CORE, *GROUPS, *PORTAL_RATED],
}


def build_frames():
    std, talent, ret, games, _ = load_bundle()
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    pff_lag = pff.build_lagged_team_talent()
    indices = {y: s.index for y, s in talent.items()}
    war_lag = war.lagged_team_talent(indices)
    portal = portal_features(GAME_YEARS)
    groups = group_features(GAME_YEARS)

    frames, coverage = {}, {}
    for year in GAME_YEARS:
        frame = V4.build_frame(year, std, talent, ret, od, pff_lag, war_lag,
                               granular=True)
        if frame is None:
            continue
        frames[year] = TS.attach(frame, portal[year], groups[year])
        coverage[str(year)] = {
            "portal": frames[year].attrs["portal_coverage"],
            "groups": frames[year].attrs["groups_coverage"],
        }
    return frames, games, coverage


def _bootstrap(frame, left, right, draws=5000, seed=20260825):
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


def main():
    frames, games, coverage = build_frames()
    years = [y for y in GAME_YEARS if y in frames]
    parts = {n: V4.assemble(years, frames, games, c) for n, c in SPECS.items()}
    rows = []
    for test in (2022, 2023, 2024, 2025):
        pool = [y for y in years if y < test]
        if not pool:
            continue
        meta = None
        for name, cols in SPECS.items():
            knobs, _ = BT.tune(parts[name], pool, cols)
            k, blend, _ = BT.tune_dynamic(parts[name], frames, pool, cols, knobs)
            model, static, margin = BT.fit_predict(parts[name], pool, test, cols,
                                                    knobs)
            dynamic, _ = BT.dynamic_predictions(model, frames[test],
                                                 parts[name][test], k, blend)
            if meta is None:
                meta = parts[name][test][4].copy()
                meta["season"], meta["y"] = test, parts[name][test][1]
            meta[f"p_static_{name}"] = static
            meta[f"p_dynamic_{name}"] = dynamic
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
    result = {"specs": SPECS, "coverage": coverage, "pooled": pooled,
              "vs_clean_core": vs_core,
              "selection_bar": BT.SELECTION_MIN_GAIN}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    base_s = pooled["clean_core"]["static"]["brier"]
    base_o = pooled["clean_core"]["dynamic"]["brier"]
    print(f"\n{'candidate':16s} {'static':>9} {'delta':>9} {'online':>9} {'delta':>9}")
    for name in SPECS:
        s = pooled[name]["static"]["brier"]
        o = pooled[name]["dynamic"]["brier"]
        print(f"{name:16s} {s:>9.5f} {s-base_s:>+9.5f} {o:>9.5f} {o-base_o:>+9.5f}")
    print("\nvs clean core (negative = better):")
    for name, block in vs_core.items():
        st, dy = block["static"], block["dynamic"]
        print(f"  {name:14s} static {st['difference']:+.5f} "
              f"[{st['ci95'][0]:+.5f},{st['ci95'][1]:+.5f}]   "
              f"online {dy['difference']:+.5f} "
              f"[{dy['ci95'][0]:+.5f},{dy['ci95'][1]:+.5f}]")
    print(f"\n-> {OUT_JSON}")
    return result


if __name__ == "__main__":
    main()
