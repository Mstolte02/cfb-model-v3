"""Does pitting a team's rooms against the opponent's rooms beat one team WAR number?

Three questions, in the order they have to be asked:

1. **Does splitting WAR up help at all?**  The 25 (facet x position) units sum to team
   WAR exactly, and the 11 preseason rooms sum to `war_projected` exactly, so a linear
   family is the same total re-weighted.  If the split carries nothing, no pairing of
   it can.
2. **Which units actually interact?**  Every offence unit is crossed against every
   defence unit - 12x13 realized, 6x5 projected - and ranked by how much the cross
   term alone moves the Brier score.  This is a description of the sample, printed to
   be read, and it never chooses what ships.
3. **Does the pairing beat the plain rating?**  The predeclared pairs go in as the
   nonlinear odd contrast `V4.MATCHUP_PAIRS` already uses, and every family is scored
   on the same expanding replay as the production model, against the same reference,
   under the same 0.001 adoption bar.

The linear cross form needs no test: `b(O_h - D_a) + b(D_h - O_a)` expands to
`b(O_h - O_a) + b(D_h - D_a)`, so a linear "matchup" model is the like-for-like model
with worse notation (`audit/RATING_ARCHITECTURE_EXPERIMENTS.md`).  Only the
nonlinearity is new, and only the nonlinearity is on trial here.

Coverage limits the answer, and saying so is part of the answer.  The realized facet
split needs season N-1, and the WAR build has no 2020, so 2021 carries no units and
the realized families are effectively judged on 2023-25.  The projected rooms cover
2021-25 in full.

Run: python -m scripts.facet_matchup_backtest
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import ARTIFACTS, GAME_YEARS, OPP_ADJ_ALPHA
from scripts.train import load_bundle
from scripts.v4_backtest import (DEFAULTS, SELECTION_MIN_GAIN, fit_predict,
                                 forward_score, metric, paired_week_bootstrap)
from src import facet_matchup as FM
from src import oppadj as OA
from src import v4 as V4
from src.data import pff, war
from src.dynamic import weekly_replay

OUT_JSON = ARTIFACTS / "facet_matchup_backtest.json"
OUT_CSV = ARTIFACTS / "facet_matchup_backtest_predictions.csv"
SCAN_CSV = ARTIFACTS / "facet_matchup_interactions.csv"

TEST_SEASONS = [2022, 2023, 2024, 2025]
# First test season whose training pool contains a live realized unit split.
FAIR_WINDOW_FROM = 2023
CORE = ["O", "D", "talent", "returning"]
REFERENCE = "core_war_projected"
# One weekly-update setting for every family, so the online column compares families
# and not their tuning budgets.  These are the v4 module defaults.
DYNAMIC_K, DYNAMIC_BLEND = .15, .75


def build_frames():
    """The production v4 frames, plus unit and room columns."""
    std, talent, ret, games, _ = load_bundle()
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    pff_lag = pff.build_lagged_team_talent()
    index_by_year = {y: s.index for y, s in talent.items()}
    war_lag = war.lagged_team_talent(index_by_year)
    war_projected = war.projected_team_talent(index_by_year)
    units = FM.lagged_unit_war(index_by_year)
    groups = FM.projected_group_war(index_by_year)

    frames = {}
    for year in GAME_YEARS:
        frame = V4.build_frame(year, std, talent, ret, od, pff_lag, war_lag)
        if frame is None:
            continue
        projected = war_projected.get(year, pd.Series(dtype=float)).reindex(frame.index)
        frame["war_projected"] = projected.fillna(0.0)
        frame.attrs["war_projected_coverage"] = float(projected.notna().mean())
        FM.attach(frame, units.get(year), groups.get(year))
        frames[year] = frame
    return frames, games


def families(unit_scan, group_scan):
    """Every family scored head to head, plus the two references."""
    unit_cols, group_cols = FM.unit_columns(), FM.group_columns()
    unit_pairs, group_pairs = list(FM.UNIT_PAIRS), list(FM.GROUP_PAIRS)
    war_core = [*CORE, "war_projected"]
    return {
        # ------------------------------------------------------------ references
        "clean_core": CORE,
        REFERENCE: war_core,
        # ------------------------------- does splitting the same total help at all?
        "group_linear": [*war_core, *group_cols],
        "unit_linear": [*war_core, *unit_cols],
        "group_linear_no_total": [*CORE, *group_cols],
        "unit_linear_no_total": [*CORE, *unit_cols],
        # ------------------------------------------ predeclared matchup crosses
        "group_cross": [*war_core, *group_pairs],
        "unit_cross": [*war_core, *unit_pairs],
        "group_linear_cross": [*war_core, *group_cols, *group_pairs],
        "unit_linear_cross": [*war_core, *unit_cols, *unit_pairs],
        "both_cross": [*war_core, *group_pairs, *unit_pairs],
        # -------------------- can the matchup stand in for the aggregate entirely?
        "group_cross_no_total": [*CORE, *group_pairs],
        "unit_cross_no_total": [*CORE, *unit_pairs],
        # ---------------------- the whole scan, to bound what any subset could add
        "group_scan_all": [*war_core, *group_scan],
        "unit_scan_all": [*war_core, *unit_scan],
    }


def slice_parts(parts, names, index_of):
    """Column-slice the one wide design into a family's own design."""
    columns = [index_of[n] for n in names]
    return {y: (X[:, columns], y_, h, m, meta)
            for y, (X, y_, h, m, meta) in parts.items()}


def scan_interactions(parts, index_of, pairs, pool, base_names, label):
    """Rank single cross terms by the Brier they save on top of a fixed base.

    Scored on the pooled seasons handed in, refit each time.  In-sample by
    construction and reported as description only: nothing downstream reads it.
    """
    base = slice_parts(parts, base_names, index_of)
    X, y, h, m = (np.vstack([base[s][0] for s in pool]),
                  np.concatenate([base[s][1] for s in pool]),
                  np.concatenate([base[s][2] for s in pool]),
                  np.concatenate([base[s][3] for s in pool]))
    model = V4.fit(X, y, h, m, base_names, **DEFAULTS)
    reference = float(np.mean(
        [(model.win_prob(x, hh) - yy) ** 2 for x, hh, yy in zip(X, h, y)]))

    # The scan names a pair by its two units; the predeclared table names it by the
    # matchup it stands for. Match on the pair itself, not on the label.
    declared = {**FM.UNIT_PAIRS, **FM.GROUP_PAIRS}
    named = {v: k for k, v in declared.items()}

    rows = []
    for name, (off, deff) in pairs.items():
        names = [*base_names, name]
        part = slice_parts(parts, names, index_of)
        Xa = np.vstack([part[s][0] for s in pool])
        fitted = V4.fit(Xa, y, h, m, names, **DEFAULTS)
        brier = float(np.mean(
            [(fitted.win_prob(x, hh) - yy) ** 2 for x, hh, yy in zip(Xa, h, y)]))
        rows.append({"base": label, "pair": name, "offense": off, "defense": deff,
                     "brier": brier, "brier_gain": reference - brier,
                     "coef": float(fitted.coef[-1]),
                     "predeclared": named.get((off, deff), "")})
    out = pd.DataFrame(rows).sort_values("brier_gain", ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    out.attrs["reference_brier"] = reference
    return out


def main():
    started = time.time()
    frames, games = build_frames()

    unit_scan = FM.register_pairs(FM.all_unit_pairs(), FM.UNIT_PREFIX)
    group_scan = FM.register_pairs(FM.all_group_pairs(), FM.GROUP_PREFIX)
    FM.register_unit_pairs()
    FM.register_group_pairs()

    catalogue = families(unit_scan, group_scan)
    all_names = list(dict.fromkeys(
        [n for names in catalogue.values() for n in names]))
    index_of = {name: i for i, name in enumerate(all_names)}
    parts = V4.assemble(GAME_YEARS, frames, games, all_names)
    print(f"assembled {sum(len(p[1]) for p in parts.values())} games x "
          f"{len(all_names)} columns in {time.time() - started:.0f}s")

    # The reciprocity invariant has to hold for the new terms too, or none of the
    # numbers below mean anything.
    probe = frames[max(frames)]
    for name in all_names:
        a, b = list(probe.index)[:2]
        forward = V4.matchup_vector(probe, a, b, [name])
        reverse = V4.matchup_vector(probe, b, a, [name])
        if not np.allclose(forward, -reverse, atol=1e-12):
            raise AssertionError(f"{name} is not antisymmetric")

    # ------------------------------------------------------- interaction scan
    pooled = [s for s in TEST_SEASONS if s in parts]
    unit_table = scan_interactions(parts, index_of, FM.all_unit_pairs(), pooled,
                                   [*CORE, "war_projected"], "unit")
    group_table = scan_interactions(parts, index_of, FM.all_group_pairs(), pooled,
                                    [*CORE, "war_projected"], "group")
    scan = pd.concat([unit_table, group_table], ignore_index=True)
    scan.to_csv(SCAN_CSV, index=False)
    show = ["offense", "defense", "rank", "brier_gain", "coef", "predeclared"]
    fmt = lambda v: f"{v: .5f}"  # noqa: E731
    print(f"\ntop realized unit interactions of {len(unit_table)} "
          f"(in-sample description)")
    print(unit_table.head(12)[show].to_string(index=False, float_format=fmt))
    print("\nwhere the predeclared football pairs actually rank")
    print(unit_table[unit_table.predeclared != ""][show]
          .to_string(index=False, float_format=fmt))
    print(f"\ntop projected room interactions of {len(group_table)}")
    print(group_table.head(8)[show].to_string(index=False, float_format=fmt))
    print("\nwhere the predeclared room pairs actually rank")
    print(group_table[group_table.predeclared != ""][show]
          .to_string(index=False, float_format=fmt))

    # ------------------------------------------------- head-to-head replay
    folds, predictions, coverage = [], [], {}
    for test in TEST_SEASONS:
        pool = [y for y in GAME_YEARS if y < test]
        if test not in parts or not pool:
            continue
        row = {"season": int(test), "train_seasons": pool, "families": {}}
        for name, names in catalogue.items():
            family = slice_parts(parts, names, index_of)
            model, p_static, pred_margin = fit_predict(family, pool, test, names,
                                                       DEFAULTS)
            p_dynamic, _ = weekly_replay(model, frames[test], family[test],
                                         DYNAMIC_K, DYNAMIC_BLEND)
            row["families"][name] = {
                "n_features": len(names),
                "static": metric(family[test][1], p_static, family[test][3],
                                 pred_margin),
                "dynamic": metric(family[test][1], p_dynamic),
                "forward_selection_score": forward_score(family, pool, names,
                                                         DEFAULTS),
            }
            meta = family[test][4].copy()
            meta["season"] = test
            meta["y"] = family[test][1]
            meta["family"] = name
            meta["p_static"] = p_static
            meta["p_dynamic"] = p_dynamic
            predictions.append(meta)
        folds.append(row)
        best = min(row["families"], key=lambda f: row["families"][f]["static"]["brier"])
        print(f"\n{test}: best static = {best}")
        for name in catalogue:
            f = row["families"][name]
            print(f"   {name:<24} static {f['static']['brier']:.5f}  "
                  f"online {f['dynamic']['brier']:.5f}  k={f['n_features']}")
        coverage[str(test)] = {
            "unit": frames[test].attrs.get("fu_coverage"),
            "group": frames[test].attrs.get("gu_coverage"),
            "war_projected": frames[test].attrs.get("war_projected_coverage")}

    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(OUT_CSV, index=False)

    def compare(frame, catalogue):
        """Pooled metrics and the paired difference against the shipping features."""
        reference = frame[frame.family == REFERENCE]
        pooled, paired = {}, {}
        for name in catalogue:
            piece = frame[frame.family == name]
            pooled[name] = {"static": metric(piece.y, piece.p_static),
                            "dynamic": metric(piece.y, piece.p_dynamic)}
            if name == REFERENCE:
                continue
            joined = piece.merge(
                reference[["season", "week", "home_team", "away_team",
                           "p_static", "p_dynamic"]],
                on=["season", "week", "home_team", "away_team"],
                suffixes=("", "_ref"))
            paired[name] = {
                "static": paired_week_bootstrap(joined, left="p_static",
                                                right="p_static_ref"),
                "dynamic": paired_week_bootstrap(joined, left="p_dynamic",
                                                 right="p_dynamic_ref")}
        return pooled, paired

    pooled_rows, bootstraps = compare(pred, catalogue)
    # The realized unit split needs season N-1, and the WAR build has no 2020, so
    # every unit column is zero in 2021 and the unit families are byte-identical to
    # the reference in the 2022 fold. Including that fold does not punish them, it
    # flatters them by averaging in a tie, so the honest window is stated separately.
    fair = pred[pred.season >= FAIR_WINDOW_FROM]
    fair_pooled, fair_bootstraps = compare(fair, catalogue)

    # Two bars, because they answer different questions. Beating `clean_core` only
    # says the family carries the WAR the model already has - every room family sums
    # to `war_projected`, so of course it clears that. The bar that decides anything
    # is beating the feature set that ships today.
    verdicts = {}
    for name in catalogue:
        if name in ("clean_core", REFERENCE):
            continue
        row = {}
        for against in ("clean_core", REFERENCE):
            gains = [f["families"][against]["forward_selection_score"]
                     - f["families"][name]["forward_selection_score"] for f in folds]
            row[against] = {
                "per_window_gain": gains,
                "windows_clearing_bar": sum(1 for g in gains
                                            if g >= SELECTION_MIN_GAIN),
                "adopted": sum(1 for g in gains if g >= SELECTION_MIN_GAIN) >= 2}
        verdicts[name] = row
    adopted = [n for n, r in verdicts.items() if r[REFERENCE]["adopted"]]

    result = {
        "question": "does WAR broken out by facet and position, pitted against the "
                    "opponent's matching unit, beat one team WAR number",
        "contract": "season N reads completed N-1 unit WAR and preseason-N projected "
                    "room WAR; no season-N participation, snaps or results",
        "cross_term": "edge_ha*|edge_ha| - edge_ah*|edge_ah|, antisymmetric by "
                      "construction and asserted per column",
        "linear_cross_note": "a linear offence-minus-opponent-defence term is "
                             "algebraically the like-for-like difference, so only the "
                             "nonlinear contrast is tested",
        "dynamic_settings": {"k": DYNAMIC_K, "blend": DYNAMIC_BLEND,
                             "note": "held fixed across families"},
        "reference_family": REFERENCE,
        "selection_rule": {"reference": "clean_core",
                           "minimum_extension_brier_gain": SELECTION_MIN_GAIN,
                           "windows_required": 2},
        "families": catalogue,
        "unit_definitions": {u: {"side": side, "cells": cells}
                             for u, (side, cells) in FM.UNITS.items()},
        "predeclared_unit_pairs": FM.UNIT_PAIRS,
        "predeclared_group_pairs": FM.GROUP_PAIRS,
        "coverage": coverage,
        "folds": folds,
        "pooled": pooled_rows,
        "paired_week_bootstrap_vs_reference": bootstraps,
        "fair_window": {
            "from_season": FAIR_WINDOW_FROM,
            "why": "the realized unit split is zero in 2021, so the 2022 fold ties "
                   "the unit families to the reference instead of testing them",
            "pooled": fair_pooled,
            "paired_week_bootstrap_vs_reference": fair_bootstraps},
        "forward_selection_verdicts": verdicts,
        "families_clearing_adoption_bar_vs_shipping": adopted,
        "interaction_scan_reference_brier": {
            "unit": unit_table.attrs["reference_brier"],
            "group": group_table.attrs["reference_brier"]},
        "runtime_seconds": round(time.time() - started, 1),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, default=float))

    for title, table, paired in (("all test seasons", pooled_rows, bootstraps),
                                 (f"{FAIR_WINDOW_FROM}-{TEST_SEASONS[-1]}, the "
                                  "window where every family is live",
                                  fair_pooled, fair_bootstraps)):
        print(f"\npooled, {title}")
        for name in sorted(table, key=lambda n: table[n]["static"]["brier"]):
            row = table[name]
            delta = row["static"]["brier"] - table[REFERENCE]["static"]["brier"]
            online = row["dynamic"]["brier"] - table[REFERENCE]["dynamic"]["brier"]
            ci = paired.get(name, {}).get("static", {}).get("ci95")
            band = f"  95% [{ci[0]:+.5f}, {ci[1]:+.5f}]" if ci else ""
            print(f"   {name:<24} static {row['static']['brier']:.5f} ({delta:+.5f})"
                  f"  online {row['dynamic']['brier']:.5f} ({online:+.5f}){band}")
    print(f"\nfamilies beating the shipping features by the adoption bar: "
          f"{adopted or 'none'}")
    print(f"-> {OUT_JSON}\n-> {OUT_CSV}\n-> {SCAN_CSV}")


if __name__ == "__main__":
    main()
