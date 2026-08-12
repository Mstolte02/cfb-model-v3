"""Strict tests of pace, game-script, and quick-pass/pressure matchup features.

Two entry points are scored side by side:

* overlay: a penalized correction to the locked weekly v4 probability/margin;
* initial: coefficients estimated jointly with the clean reciprocal team core.

Drive traits use only completed prior weeks (last 12 games).  Quarterback time to
throw and team pressure/style use the completed N-1 season.  Outer test season N and
all hyperparameter tuning use seasons before N only.

Run: python -m scripts.tempo_style_backtest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import ARTIFACTS, GAME_YEARS
from scripts.four_pass_backtest import score_family as score_overlay
from scripts.four_pass_initial_backtest import (
    family_parts, score_family as score_initial,
)
from scripts.train_v4 import build_frames, build_inputs
from scripts.v4_backtest import (
    fit_predict, metric, paired_week_bootstrap, tune, tune_dynamic,
)
from src import tempo
from src import v4 as V4
from src.dynamic import update_delta


PRED_PATH = ARTIFACTS / "v4_backtest_predictions.csv"
OUT_JSON = ARTIFACTS / "tempo_style_backtest.json"
OUT_OVERLAY_CSV = ARTIFACTS / "tempo_style_overlay_predictions.csv"
OUT_INITIAL_CSV = ARTIFACTS / "tempo_style_initial_predictions.csv"
BASE_NAMES = ["O", "D", "talent", "returning", "war_projected"]

FAMILIES = {
    "pace_identity": tempo.PACE_IDENTITY,
    "scripted_windows": tempo.SCRIPT_WINDOWS,
    "state_and_control": tempo.STATE_CONTROL,
    "pace_matchup_control": tempo.PACE_MATCHUP,
    "quick_pass_pressure": tempo.QUICK_PRESSURE,
    "pace_script_control": [
        *tempo.PACE_IDENTITY, *tempo.SCRIPT_WINDOWS,
        *tempo.STATE_CONTROL, *tempo.PACE_MATCHUP,
    ],
    "all_tempo_style": [
        *tempo.PACE_IDENTITY, *tempo.SCRIPT_WINDOWS,
        *tempo.STATE_CONTROL, *tempo.PACE_MATCHUP, *tempo.QUICK_PRESSURE,
    ],
}


def build_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = pd.read_csv(PRED_PATH)
    drive = tempo.build_rolling_drive_features()
    drive = tempo.attach_quick_pressure(drive)
    data = pred.merge(drive, on=tempo.KEYS, how="left", validate="one_to_one")
    needed = sorted(set(sum(FAMILIES.values(), [])))
    missing = data[needed].isna().any(axis=1)
    if missing.any():
        examples = data.loc[missing, tempo.KEYS].head().to_dict("records")
        raise RuntimeError(f"tempo/style join missed {int(missing.sum())} games: {examples}")
    return data, drive


def build_initial_inputs(features: pd.DataFrame):
    std, talent, returning, games, od, pff, war, _ = build_inputs(
        include_projection=False)
    frames = build_frames(std, talent, returning, od, pff, war, GAME_YEARS)
    base = V4.assemble(GAME_YEARS, frames, games, BASE_NAMES)
    needed = sorted(set(sum(FAMILIES.values(), [])))
    feature_frames = {}
    for year, part in base.items():
        meta = part[4].copy()
        meta.insert(0, "season", year)
        joined = meta.merge(features[[*tempo.KEYS, *needed]], on=tempo.KEYS,
                            how="left", sort=False, validate="one_to_one")
        if len(joined) != len(meta) or joined[needed].isna().any(axis=1).any():
            raise RuntimeError(f"initial tempo/style join failed for {year}")
        feature_frames[year] = joined
    return base, feature_frames, frames


def base_cache(base):
    cache = {}
    for test in [2023, 2024, 2025]:
        pool = [year for year in GAME_YEARS if year < test]
        knobs, _ = tune(base, pool, BASE_NAMES)
        _, p, margin = fit_predict(base, pool, test, BASE_NAMES, knobs)
        cache[test] = (knobs, p, margin)
    return cache


def _sigmoid(value):
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0))))


def context_dynamic_predictions(core_model, context_model, frame, part, k, blend):
    """Blend initial and weekly probabilities with context in both branches.

    The live branch owns persistent team strength.  Only the jointly fitted
    game-context logit is added to that rating gap; core coefficients are not counted
    twice.  Thus a 100% dynamic blend can still use matchup-specific evidence.
    """
    X, y, home_flag, margins, meta = part
    ratings = {team: core_model.team_logit_strength(frame, team)
               for team in frame.index}
    out = np.zeros(len(y)); dynamic = np.zeros(len(y))
    order = meta.assign(_row=np.arange(len(meta))).sort_values(["week", "_row"])
    for _, slate in order.groupby("week", sort=True, dropna=False):
        changes = {}
        for _, row in slate.iterrows():
            i = int(row["_row"])
            home, away = row["home_team"], row["away_team"]
            gap = (ratings[home] - ratings[away] +
                   core_model.hfa_coef * home_flag[i])
            n_context = max(len(context_model.feature_names) - len(BASE_NAMES), 0)
            context_effect = (float(context_model.coef[-n_context:] @ X[i, -n_context:])
                              if n_context else 0.0)
            adjusted_gap = gap + context_effect
            p_dynamic = _sigmoid(adjusted_gap)
            p_initial = context_model.win_prob(X[i], home_flag[i])
            out[i] = (1.0 - blend) * p_initial + blend * p_dynamic
            dynamic[i] = p_dynamic
            delta = update_delta(k, margins[i], adjusted_gap, y[i], p_dynamic)
            changes[home] = changes.get(home, 0.0) + delta
            changes[away] = changes.get(away, 0.0) - delta
        for team, change in changes.items():
            ratings[team] += change
    return out, dynamic


def dynamic_base_cache(base, frames):
    cache = {}
    for test in [2023, 2024, 2025]:
        pool = [year for year in GAME_YEARS if year < test]
        knobs, _ = tune(base, pool, BASE_NAMES)
        model, _, _ = fit_predict(base, pool, test, BASE_NAMES, knobs)
        k, blend, _ = tune_dynamic(base, frames, pool, BASE_NAMES, knobs)
        p, _ = context_dynamic_predictions(model, model, frames[test], base[test],
                                            k, blend)
        cache[test] = {"model": model, "knobs": knobs, "k": k,
                       "blend": blend, "p": p}
    return cache


def tune_context_dynamic(base, parts, frames, pool, names, core_knobs,
                         context_knobs):
    """Select weekly update/blend for context initial probabilities before outer N."""
    grid_k, grid_blend = [.05, .10, .15, .20, .30], [0, .25, .5, .75, 1.0]
    scores = {(k, blend): [] for k in grid_k for blend in grid_blend}
    for i in range(1, len(pool)):
        test, train = pool[i], pool[:i]
        core_model, _, _ = fit_predict(base, train, test, BASE_NAMES, core_knobs)
        context_model, _, _ = fit_predict(parts, train, test, names, context_knobs)
        for k, blend in scores:
            p, _ = context_dynamic_predictions(
                core_model, context_model, frames[test], parts[test], k, blend)
            scores[(k, blend)].extend((p - parts[test][1]) ** 2)
    if not any(scores.values()):
        return .15, .75, {}
    means = {f"{k},{blend}": float(np.mean(values))
             for (k, blend), values in scores.items()}
    best = min(scores, key=lambda pair: np.mean(scores[pair]))
    return best[0], best[1], means


def score_integrated_dynamic(base, feature_frames, frames, name, features,
                             core_cache):
    names = [*BASE_NAMES, *features]
    parts = family_parts(base, feature_frames, features)
    folds, predictions = [], []
    for test in [2023, 2024, 2025]:
        pool = [year for year in GAME_YEARS if year < test]
        core = core_cache[test]
        knobs, _ = tune(parts, pool, names)
        k, blend, dynamic_tuning = tune_context_dynamic(
            base, parts, frames, pool, names, core["knobs"], knobs)
        model, _, _ = fit_predict(parts, pool, test, names, knobs)
        fixed_model, _, _ = fit_predict(parts, pool, test, names, core["knobs"])
        p, _ = context_dynamic_predictions(
            core["model"], model, frames[test], parts[test],
            k, blend)
        fixed, _ = context_dynamic_predictions(
            core["model"], fixed_model, frames[test], parts[test],
            core["k"], core["blend"])
        y = parts[test][1]
        folds.append({
            "season": test, "n": int(len(y)), "knobs": knobs,
            "dynamic_k": k, "dynamic_blend": blend,
            "base_dynamic_k": core["k"],
            "base_dynamic_blend": core["blend"],
            "dynamic_tuning": dynamic_tuning,
            "brier_change": metric(y, p)["brier"] - metric(y, core["p"])["brier"],
            "same_knobs_brier_change": (metric(y, fixed)["brier"] -
                                        metric(y, core["p"])["brier"]),
        })
        meta = parts[test][4].copy()
        meta.insert(0, "season", test)
        meta["family"], meta["y"] = name, y
        meta["p_core_dynamic"] = core["p"]
        meta["p_context_dynamic"] = p
        meta["p_context_same_knobs_dynamic"] = fixed
        predictions.append(meta)
    pred = pd.concat(predictions, ignore_index=True)
    base_metric = metric(pred.y, pred.p_core_dynamic)
    candidate_metric = metric(pred.y, pred.p_context_dynamic)
    fixed_metric = metric(pred.y, pred.p_context_same_knobs_dynamic)
    return {
        "name": name, "features": features, "n": int(len(pred)),
        "pooled_brier_change": candidate_metric["brier"] - base_metric["brier"],
        "same_knobs_brier_change": fixed_metric["brier"] - base_metric["brier"],
        "base": base_metric, "candidate": candidate_metric,
        "same_knobs": fixed_metric, "folds": folds,
        "paired_week_bootstrap": paired_week_bootstrap(
            pred, left="p_context_dynamic", right="p_core_dynamic"),
        "same_knobs_paired_week_bootstrap": paired_week_bootstrap(
            pred, left="p_context_same_knobs_dynamic", right="p_core_dynamic"),
    }, pred


def main():
    data, features = build_data()
    base, feature_frames, frames = build_initial_inputs(features)
    cached_base = base_cache(base)
    cached_dynamic = dynamic_base_cache(base, frames)
    overlay_results, initial_results, integrated_results = [], [], []
    overlay_predictions, initial_predictions, integrated_predictions = [], [], []
    for name, features in FAMILIES.items():
        overlay, op = score_overlay(data, name, features)
        overlay["paired_week_bootstrap"] = paired_week_bootstrap(
            op, left="p_four_pass", right="p_dynamic")
        initial, ip = score_initial(base, feature_frames, name, features,
                                    cached_base)
        integrated, dp = score_integrated_dynamic(
            base, feature_frames, frames, name, features, cached_dynamic)
        overlay_results.append(overlay)
        initial_results.append(initial)
        integrated_results.append(integrated)
        overlay_predictions.append(op)
        initial_predictions.append(ip)
        integrated_predictions.append(dp)
        print(f"{name:<28} overlay {overlay['pooled_brier_change']:+.5f}  "
              f"initial {initial['pooled_brier_change']:+.5f}  "
              f"integrated {integrated['pooled_brier_change']:+.5f}")

    payload = {
        "contract": "start-of-week rolling drives; N-1 PFF/TruMedia; outer N trains/tunes before N",
        "sources": {
            "pace_and_windows": "CFBD /drives regular-season summaries",
            "quick_pass": "N-1 PFF team quarterback avg_time_to_throw",
            "pressure_and_style": "N-1 TruMedia/PFF pressure, blitz, and early pass rates",
        },
        "definitions": {
            "pace": "offensive game-clock seconds per play on valid drives",
            "script": "drives covering approximately the first 15 offensive plays",
            "middle_eight": "drives starting in final 4:00 Q2 or first 4:00 Q3",
            "pace_control": "whether realized game pace was closer to a team's entering preference than its opponent's",
            "fast_slow_fit": "shrunk win rates in <=25 and >=28 second/play games, weighted by expected matchup pace",
        },
        "adoption_threshold": {"pooled_brier_gain": .001,
                               "require_fold_stability": True},
        "overlay_families": overlay_results,
        "initial_families": initial_results,
        "integrated_dynamic_families": integrated_results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    pd.concat(overlay_predictions, ignore_index=True).to_csv(OUT_OVERLAY_CSV,
                                                              index=False)
    pd.concat(initial_predictions, ignore_index=True).to_csv(OUT_INITIAL_CSV,
                                                              index=False)
    pd.concat(integrated_predictions, ignore_index=True).to_csv(
        ARTIFACTS / "tempo_style_integrated_predictions.csv", index=False)
    print(f"-> {OUT_JSON}\n-> {OUT_OVERLAY_CSV}\n-> {OUT_INITIAL_CSV}")


if __name__ == "__main__":
    main()
