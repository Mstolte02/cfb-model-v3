"""Strict joint-fit test of Four-Pass features in the initial v4 probability.

Unlike four_pass_backtest.py, this is not a correction to an already-produced
probability.  The reciprocal v4 logistic and margin models estimate core team,
fragility/reversibility, and structural-context coefficients together.  Every
rolling game-shape value is frozen before kickoff, and every outer season is fit
and tuned using earlier seasons only.

Run: python -m scripts.four_pass_initial_backtest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import ARTIFACTS, GAME_YEARS
from scripts.four_pass_backtest import KEYS, build_shape_features
from scripts.train_v4 import build_frames, build_inputs
from scripts.v4_backtest import fit_predict, metric, paired_week_bootstrap, tune
from src import v4 as V4


OUT_JSON = ARTIFACTS / "four_pass_initial_backtest.json"
OUT_CSV = ARTIFACTS / "four_pass_initial_backtest_predictions.csv"
BASE_NAMES = ["O", "D", "talent", "returning", "war_projected"]
TEST_YEARS = [2023, 2024, 2025]

FRAGILITY = [
    "frag_giveback_diff", "frag_lead_loss_diff",
    "frag_late_decline_diff", "frag_volatility_diff",
]
REVERSIBILITY = [
    "rev_comeback_gain_diff", "rev_comeback_win_diff", "rev_late_gain_diff",
]
FRAG_REV = ["frag_index_diff", "rev_index_diff", "frag_rev_cross"]
STRUCTURAL = [
    "struct_linear", "struct_nonlinear", "struct_breadth",
    "core_x_struct_support", "core_x_struct_coherence",
]


def _direct_shape_features(d: pd.DataFrame) -> pd.DataFrame:
    """Create antisymmetric team-difference and matchup interaction features."""
    out = d.copy()
    for source, name in (
        ("giveback", "frag_giveback_diff"),
        ("lead_loss", "frag_lead_loss_diff"),
        ("late_decline", "frag_late_decline_diff"),
        ("volatility", "frag_volatility_diff"),
        ("comeback_gain", "rev_comeback_gain_diff"),
        ("comeback_win", "rev_comeback_win_diff"),
        ("late_gain", "rev_late_gain_diff"),
    ):
        out[name] = out[f"home_{source}"] - out[f"away_{source}"]

    def index(side: str, kind: str):
        if kind == "frag":
            return ((out[f"{side}_giveback"] / 14.0) +
                    out[f"{side}_lead_loss"] +
                    (out[f"{side}_late_decline"] / 14.0) +
                    (out[f"{side}_volatility"] / 21.0)) / 4.0
        return ((out[f"{side}_comeback_gain"] / 14.0) +
                out[f"{side}_comeback_win"] +
                (out[f"{side}_late_gain"] / 14.0)) / 3.0

    home_frag, away_frag = index("home", "frag"), index("away", "frag")
    home_rev, away_rev = index("home", "rev"), index("away", "rev")
    out["frag_index_diff"] = home_frag - away_frag
    out["rev_index_diff"] = home_rev - away_rev
    # Positive means the home team is more fragile relative to the away team's
    # comeback profile than vice versa.  Swapping teams negates the feature.
    out["frag_rev_cross"] = home_frag * away_rev - away_frag * home_rev
    return out


def _add_structural_features(d: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in d.itertuples():
        home, away = frame.loc[r.home_team], frame.loc[r.away_team]
        edges = []
        for offense, defense in V4.MATCHUP_PAIRS.values():
            home_edge = float(home[offense]) - float(away[defense])
            away_edge = float(away[offense]) - float(home[defense])
            edges.append(home_edge - away_edge)
        edges = np.asarray(edges, float)
        strength = float(home.O + home.D - away.O - away.D)
        breadth = float(np.sign(edges).mean())
        rows.append({
            "struct_linear": float(edges.mean()),
            "struct_nonlinear": float(np.mean(edges * np.abs(edges))),
            "struct_breadth": breadth,
            # These let structural agreement alter the initial core evaluation;
            # both remain antisymmetric without using a previously fitted p.
            "core_x_struct_support": float(abs(strength) * edges.mean()),
            "core_x_struct_coherence": float(strength * abs(breadth)),
        })
    return pd.concat([d.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def build_augmented_parts(frames, games):
    base = V4.assemble(GAME_YEARS, frames, games, BASE_NAMES)
    shape = build_shape_features()
    augmented, feature_frames = {}, {}
    for year, part in base.items():
        meta = part[4].copy()
        meta.insert(0, "season", year)
        joined = meta.merge(shape, on=KEYS, how="left", sort=False,
                            validate="one_to_one")
        if len(joined) != len(meta) or joined.filter(regex="^home_giveback$").isna().any().any():
            raise RuntimeError(f"game-shape join failed for {year}")
        joined = _direct_shape_features(joined)
        joined = _add_structural_features(joined, frames[year])
        feature_frames[year] = joined
        augmented[year] = part
    return base, feature_frames


def family_parts(base, feature_frames, features):
    out = {}
    for year, part in base.items():
        context = feature_frames[year][features].to_numpy(float)
        if not np.isfinite(context).all():
            raise RuntimeError(f"non-finite initial Four-Pass feature in {year}")
        out[year] = (np.column_stack([part[0], context]), *part[1:])
    return out


def score_family(base, feature_frames, name, features, base_cache):
    names = [*BASE_NAMES, *features]
    parts = family_parts(base, feature_frames, features)
    folds, predictions = [], []
    for test in TEST_YEARS:
        pool = [y for y in GAME_YEARS if y < test]
        if test not in parts or not all(y in parts for y in pool):
            continue
        base_knobs, base_p, base_pm = base_cache[test]
        knobs, tuning = tune(parts, pool, names)
        model, p, pm = fit_predict(parts, pool, test, names, knobs)
        _, paired_p, paired_pm = fit_predict(
            parts, pool, test, names, base_knobs)
        y, margin = parts[test][1], parts[test][3]
        base_metric = metric(y, base_p, margin, base_pm)
        joint_metric = metric(y, p, margin, pm)
        paired_metric = metric(y, paired_p, margin, paired_pm)
        fold = {
            "season": test, "n": int(len(y)), "knobs": knobs,
            "base_knobs": base_knobs, "tuning": tuning,
            "brier_change": joint_metric["brier"] - base_metric["brier"],
            "logloss_change": joint_metric["logloss"] - base_metric["logloss"],
            "margin_mae_change": (joint_metric["margin_mae"] -
                                  base_metric["margin_mae"]),
            "margin_rmse_change": (joint_metric["margin_rmse"] -
                                   base_metric["margin_rmse"]),
            "same_knobs_brier_change": (paired_metric["brier"] -
                                        base_metric["brier"]),
            "same_knobs_margin_mae_change": (paired_metric["margin_mae"] -
                                             base_metric["margin_mae"]),
            "same_knobs_margin_rmse_change": (paired_metric["margin_rmse"] -
                                              base_metric["margin_rmse"]),
            "context_logit_coef": dict(zip(features, model.coef[-len(features):])),
            "context_margin_coef": dict(zip(features,
                                             model.margin_coef[-len(features):])),
        }
        folds.append(fold)
        meta = parts[test][4].copy()
        meta.insert(0, "season", test)
        meta["family"] = name
        meta["y"], meta["margin"] = y, margin
        meta["p_base"], meta["p_joint"] = base_p, p
        meta["p_same_knobs"] = paired_p
        meta["margin_base"], meta["margin_joint"] = base_pm, pm
        meta["margin_same_knobs"] = paired_pm
        predictions.append(meta)

    pred = pd.concat(predictions, ignore_index=True)
    base_metric = metric(pred.y, pred.p_base, pred.margin, pred.margin_base)
    joint_metric = metric(pred.y, pred.p_joint, pred.margin, pred.margin_joint)
    paired_metric = metric(pred.y, pred.p_same_knobs, pred.margin,
                           pred.margin_same_knobs)
    result = {
        "name": name, "features": features, "n": int(len(pred)),
        "pooled_brier_change": joint_metric["brier"] - base_metric["brier"],
        "pooled_logloss_change": joint_metric["logloss"] - base_metric["logloss"],
        "pooled_margin_mae_change": (joint_metric["margin_mae"] -
                                     base_metric["margin_mae"]),
        "pooled_margin_rmse_change": (joint_metric["margin_rmse"] -
                                      base_metric["margin_rmse"]),
        "same_knobs_brier_change": (paired_metric["brier"] -
                                    base_metric["brier"]),
        "same_knobs_margin_mae_change": (paired_metric["margin_mae"] -
                                         base_metric["margin_mae"]),
        "same_knobs_margin_rmse_change": (paired_metric["margin_rmse"] -
                                          base_metric["margin_rmse"]),
        "base": base_metric, "joint": joint_metric,
        "same_knobs": paired_metric, "folds": folds,
        "paired_week_bootstrap": paired_week_bootstrap(
            pred, left="p_joint", right="p_base"),
        "same_knobs_paired_week_bootstrap": paired_week_bootstrap(
            pred, left="p_same_knobs", right="p_base"),
    }
    return result, pred


def main():
    std, talent, returning, games, od, pff, war, _ = build_inputs(
        include_projection=False)
    frames = build_frames(std, talent, returning, od, pff, war, GAME_YEARS)
    base, feature_frames = build_augmented_parts(frames, games)

    # Cache the identical core-only comparator once per outer fold.  Each extended
    # family is then tuned independently, but never sees its test season.
    base_cache = {}
    for test in TEST_YEARS:
        pool = [y for y in GAME_YEARS if y < test]
        knobs, _ = tune(base, pool, BASE_NAMES)
        _, p, pm = fit_predict(base, pool, test, BASE_NAMES, knobs)
        base_cache[test] = (knobs, p, pm)

    families = {
        "fragility_components": FRAGILITY,
        "reversibility_components": REVERSIBILITY,
        "fragility_reversibility_indices": FRAG_REV,
        "structural_initial_weight": STRUCTURAL,
        "all_four_pass_initial": [*FRAG_REV, *STRUCTURAL],
    }
    results, predictions = [], []
    for name, features in families.items():
        result, pred = score_family(base, feature_frames, name, features,
                                    base_cache)
        results.append(result)
        predictions.append(pred)
        print(f"{name:<38} Brier {result['pooled_brier_change']:+.5f}  "
              f"paired {result['same_knobs_brier_change']:+.5f}  "
              f"MAE {result['pooled_margin_mae_change']:+.3f}  "
              f"RMSE {result['pooled_margin_rmse_change']:+.3f}")

    payload = {
        "contract": "joint initial fit; outer season N and all tuning use seasons < N",
        "architecture": "reciprocal core and context coefficients estimated together",
        "base_features": BASE_NAMES, "test_seasons": TEST_YEARS,
        "adoption_threshold": {"pooled_brier_gain": .001,
                               "require_fold_stability": True},
        "families": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    pd.concat(predictions, ignore_index=True).to_csv(OUT_CSV, index=False)
    print(f"-> {OUT_JSON}\n-> {OUT_CSV}")


if __name__ == "__main__":
    main()
