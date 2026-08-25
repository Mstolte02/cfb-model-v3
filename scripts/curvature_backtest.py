"""Phase 7: is the talent-to-strength relationship nonlinear, and does the shipping
model already handle it?

`decision_predictive_backtest` found that three curvature terms - talent squared,
returning squared, and their product - beat the four-column clean core by .00449
static, 4.5x the adoption bar. That result arrived post-hoc as a control in a
different experiment and is not evidence of anything until it survives the objection
that kills most post-hoc findings:

    it beat a stunted baseline.

`clean_core` is four columns. The production backtest chooses among ten candidates,
several of which carry granular per-stat inputs, lagged player talent and matchup
interactions. Those are already nonlinear functions of roster quality. If curvature
only helps the four-column core, the finding is "the four-column core is
underspecified", which the production selector routes around by not picking it.

**Predeclared rule, fixed before running.** Every curvature variant is compared to
its OWN parent family, not to clean core. Curvature is adopted only if it improves
the family the production selector actually picks, by at least SELECTION_MIN_GAIN,
with a 95% paired bootstrap interval excluding zero, on BOTH the static and online
metrics. Beating clean core alone is explicitly not sufficient and is reported as a
reproduction of the prior result rather than as support.

Seven parents x two arms is fourteen comparisons, so the multiplicity that inflated
the original finding is reported rather than ignored: the number of tests, and a
Bonferroni reading of the headline interval.
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


OUT_JSON = ARTIFACTS / "curvature_backtest.json"
OUT_CSV = ARTIFACTS / "curvature_backtest_predictions.csv"

CURVATURE = ["talent_sq", "returning_sq", "talent_x_returning"]
# Talent alone, without the returning terms: separates "talent is nonlinear" from
# "some second-order term helps", which the three-term block cannot distinguish.
TALENT_ONLY = ["talent_sq"]

PARENTS = {
    "clean_core": ["O", "D", "talent", "returning"],
    "core_pff_lag": ["O", "D", "talent", "returning", "pff_lag"],
    "core_war_lag": ["O", "D", "talent", "returning", "war_lag"],
    # The production selector picks this one in 2024 and 2025, so the predeclared
    # rule is not testable without it.
    "core_war_projected": ["O", "D", "talent", "returning", "war_projected"],
    "core_players_lag": V4.CORE_FEATURES,
    "core_matchups": ["O", "D", "talent", "returning", *V4.INTERACTION_FEATURES],
    "granular_clean": [*V4.OFF_STATS, *V4.DEF_STATS, "talent", "returning"],
    "granular_players": V4.TEAM_FEATURES,
}


def _bootstrap(frame: pd.DataFrame, left: str, right: str, draws=5000,
               seed=20260825):
    """Paired season-week block bootstrap on the squared-error difference.

    Same construction the coach and decision phases used. It is inlined rather than
    imported so this script does not depend on those experiments' modules.
    """
    data = frame.dropna(subset=[left, right]).copy()
    data["delta"] = (data[left] - data.y) ** 2 - (data[right] - data.y) ** 2
    blocks = [g.delta.to_numpy(float) for _, g in
              data.groupby(["season", "week"], sort=True)]
    rng = np.random.default_rng(seed)
    samples = np.asarray([
        np.concatenate([blocks[index] for index in
                        rng.integers(0, len(blocks), len(blocks))]).mean()
        for _ in range(draws)])
    return {"difference": float(data.delta.mean()),
            "ci95": [float(value) for value in np.quantile(samples, [.025, .975])],
            "probability_left_better": float(np.mean(samples < 0)),
            "n_games": int(len(data)), "blocks": int(len(blocks))}


def build_specs() -> dict[str, list[str]]:
    specs = dict(PARENTS)
    for name, columns in PARENTS.items():
        specs[f"{name}+curvature"] = [*columns, *CURVATURE]
        specs[f"{name}+talent_sq"] = [*columns, *TALENT_ONLY]
    return specs


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
        # Standardised inputs, so a square is a genuine curvature term rather than a
        # rescaling: talent is already centred within season by the shared scaler.
        frame["talent_sq"] = frame.talent ** 2
        frame["returning_sq"] = frame.returning ** 2
        frame["talent_x_returning"] = frame.talent * frame.returning
        frames[year] = frame
    return frames, games


def main():
    frames, games = build_frames()
    specs = build_specs()
    years = [y for y in GAME_YEARS if y in frames]
    parts = {name: V4.assemble(years, frames, games, columns)
             for name, columns in specs.items()}

    folds, prediction_rows = [], []
    for test in (2022, 2023, 2024, 2025):
        pool = [y for y in years if y < test]
        if not pool:
            continue
        # The production selector runs on the real ten-candidate menu, untouched, so
        # "the family that ships" is decided the way it is actually decided.
        production_parts = {n: V4.assemble(years, frames, games, c)
                            for n, c in BT.CANDIDATES.items()}
        shipped, production_scores = BT.choose_candidate(production_parts, pool)
        selected, selection_scores = BT.choose_candidate(parts, pool, specs)
        fold = {"season": test, "production_selected": shipped,
                "selected_with_curvature_on_menu": selected,
                "production_scores": production_scores,
                "selection_scores": selection_scores, "candidates": {}}
        meta = None
        for name, columns in specs.items():
            knobs, _ = BT.tune(parts[name], pool, columns)
            k, blend, _ = BT.tune_dynamic(parts[name], frames, pool, columns, knobs)
            model, static, margin = BT.fit_predict(parts[name], pool, test,
                                                   columns, knobs)
            dynamic, _ = BT.dynamic_predictions(model, frames[test],
                                                 parts[name][test], k, blend)
            fold["candidates"][name] = {
                "static": BT.metric(parts[name][test][1], static,
                                    parts[name][test][3], margin),
                "dynamic": BT.metric(parts[name][test][1], dynamic),
                "knobs": knobs, "dynamic_k": k, "dynamic_blend": blend,
            }
            if meta is None:
                meta = parts[name][test][4].copy()
                meta["season"], meta["y"] = test, parts[name][test][1]
            meta[f"p_static_{name}"] = static
            meta[f"p_dynamic_{name}"] = dynamic
        folds.append(fold)
        prediction_rows.append(meta)
        print(f"{test}: production picks {shipped}; with curvature on the menu "
              f"{selected}", flush=True)

    predictions = pd.concat(prediction_rows, ignore_index=True)
    pooled = {name: {
        "static": BT.metric(predictions.y, predictions[f"p_static_{name}"]),
        "dynamic": BT.metric(predictions.y, predictions[f"p_dynamic_{name}"]),
    } for name in specs}

    # Each arm against its own parent. This is the predeclared comparison.
    against_parent = {}
    for parent in PARENTS:
        for suffix in ("curvature", "talent_sq"):
            arm = f"{parent}+{suffix}"
            against_parent[arm] = {
                "parent": parent,
                "static": _bootstrap(predictions, f"p_static_{arm}",
                                     f"p_static_{parent}"),
                "dynamic": _bootstrap(predictions, f"p_dynamic_{arm}",
                                      f"p_dynamic_{parent}"),
            }
    n_tests = len(against_parent) * 2

    shipped_families = sorted({fold["production_selected"] for fold in folds})
    verdict = {}
    for family in shipped_families:
        arm = f"{family}+curvature"
        if arm not in against_parent:
            continue
        block = against_parent[arm]
        static, dynamic = block["static"], block["dynamic"]
        verdict[family] = {
            "static_delta": static["difference"],
            "static_ci95": static["ci95"],
            "dynamic_delta": dynamic["difference"],
            "dynamic_ci95": dynamic["ci95"],
            "clears_bar_both_metrics": bool(
                static["difference"] <= -BT.SELECTION_MIN_GAIN and
                dynamic["difference"] <= -BT.SELECTION_MIN_GAIN and
                static["ci95"][1] < 0 and dynamic["ci95"][1] < 0),
        }

    result = {
        "predeclared_rule": (
            "curvature is adopted only if it improves the production-selected "
            "family by >= SELECTION_MIN_GAIN with a 95% interval excluding zero on "
            "both static and online; beating clean_core alone is not sufficient"),
        "minimum_gain": BT.SELECTION_MIN_GAIN,
        "multiplicity": {"comparisons": n_tests,
                         "bonferroni_alpha_for_05": 0.05 / n_tests},
        "specs": specs, "folds": folds, "pooled": pooled,
        "vs_own_parent": against_parent,
        "production_selected_families": shipped_families,
        "verdict_on_shipped_families": verdict,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    predictions.to_csv(OUT_CSV, index=False)

    print("\n--- each arm vs its OWN parent (negative = curvature helps) ---")
    print(f"{'arm':34s} {'static d':>10} {'95% CI':>22} "
          f"{'online d':>10} {'95% CI':>22}")
    for arm, block in against_parent.items():
        s, d = block["static"], block["dynamic"]
        print(f"{arm:34s} {s['difference']:>+10.5f} "
              f"[{s['ci95'][0]:+.5f},{s['ci95'][1]:+.5f}] "
              f"{d['difference']:>+10.5f} [{d['ci95'][0]:+.5f},{d['ci95'][1]:+.5f}]")
    print(f"\nproduction-selected families: {shipped_families}")
    for family, block in verdict.items():
        print(f"  {family}: clears bar on both metrics = "
              f"{block['clears_bar_both_metrics']}")
    print(f"\n-> {OUT_JSON}\n-> {OUT_CSV}")
    return result


if __name__ == "__main__":
    main()
