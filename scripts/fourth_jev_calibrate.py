"""Paired, leakage-safe calibration and tail evaluation for Fourth & Jev.

The CFB baseline comes from immutable historical state files. Jev rows are inner-
joined on game id, pregame cutoff and target, so a missing answer can never improve a
comparison by silently changing the game sample. Uncertainty resamples season-week
blocks, the temporal unit used by the replay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


TARGETS = ("home_win", "home_by_7_plus", "home_by_14_plus",
           "home_by_21_plus", "away_by_7_plus", "within_3")
BUCKETS = ("away_21_plus", "away_14_20", "away_7_13", "away_4_6",
           "away_1_3", "home_1_3", "home_4_6", "home_7_13",
           "home_14_20", "home_21_plus")


def _p(answer):
    if not isinstance(answer, dict):
        return None
    value = answer.get("noul")
    return float(value) if value is not None else None


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_states(state_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    rows, raw = [], []
    for path in sorted(state_dir.glob("states_*.jsonl")):
        for item in _iter_jsonl(path):
            raw.append(item)
            state = item["state"]
            model = state.get("existing_model", {})
            probs = model.get("baseline_probabilities") or {
                "home_win": model.get("home_win_probability")}
            outcome = item.get("outcome", {})
            game = state.get("game", {})
            for target in TARGETS:
                if probs.get(target) is None or outcome.get(target) is None:
                    continue
                rows.append({
                    "game_id": str(item["game_id"]), "as_of": item["as_of"],
                    "season": int(game.get("season")), "week": int(game.get("week")),
                    "target": target, "p": float(probs[target]),
                    "y": int(bool(outcome[target])), "source": "cfb",
                })
    return pd.DataFrame(rows), raw


def load_jev(path: Path) -> tuple[pd.DataFrame, list[dict]]:
    if not path.exists():
        return pd.DataFrame(), []
    rows, records = [], []
    for record in _iter_jsonl(path):
        records.append(record)
        answers = record.get("answers", {})
        outcome = record.get("metadata", {}).get("outcome", {})
        season = record.get("metadata", {}).get("season")
        for target in TARGETS:
            p, y = _p(answers.get(target)), outcome.get(target)
            if p is not None and y is not None:
                rows.append({
                    "game_id": str(record.get("game_id")),
                    "as_of": record.get("as_of"), "season": season,
                    "target": target, "p": p, "y": int(bool(y)),
                    "source": "jev",
                })
    return pd.DataFrame(rows), records


def _cfb_buckets(model: dict) -> np.ndarray:
    mu, sigma = float(model["predicted_margin"]), float(model["margin_sigma"])
    cuts = [-np.inf, -20.5, -13.5, -6.5, -3.5, -.5,
            .5, 3.5, 6.5, 13.5, 20.5, np.inf]
    mass = np.diff(norm.cdf(cuts, loc=mu, scale=sigma))
    # Exclude the continuous proxy's tie interval (-.5, .5) and condition the ten
    # playable integer-margin buckets back to one.
    playable = np.delete(mass, 5)
    return playable / playable.sum()


def _margin_bucket(margin: float) -> str:
    m = float(margin)
    if m >= 21: return "home_21_plus"
    if m >= 14: return "home_14_20"
    if m >= 7: return "home_7_13"
    if m >= 4: return "home_4_6"
    if m >= 1: return "home_1_3"
    if m <= -21: return "away_21_plus"
    if m <= -14: return "away_14_20"
    if m <= -7: return "away_7_13"
    if m <= -4: return "away_4_6"
    return "away_1_3"


def load_bucket_rows(state_dir: Path, records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfb_rows = []
    for path in sorted(state_dir.glob("states_*.jsonl")):
        for item in _iter_jsonl(path):
            state, outcome = item["state"], item.get("outcome", {})
            model, game = state.get("existing_model", {}), state.get("game", {})
            if model.get("predicted_margin") is None or outcome.get("home_margin") is None:
                continue
            p = _cfb_buckets(model)
            cfb_rows.append({
                "game_id": str(item["game_id"]), "as_of": item["as_of"],
                "season": int(game["season"]), "week": int(game["week"]),
                "actual_bucket": _margin_bucket(outcome["home_margin"]),
                **{f"p_{name}": float(value) for name, value in zip(BUCKETS, p)},
            })
    jev_rows = []
    for record in records:
        probs = (record.get("answers", {}).get("margin_bucket") or {}).get("probabilities")
        outcome = record.get("metadata", {}).get("outcome", {})
        if not isinstance(probs, dict) or not set(BUCKETS).issubset(probs):
            continue
        jev_rows.append({
            "game_id": str(record.get("game_id")), "as_of": record.get("as_of"),
            "season": int(record.get("metadata", {}).get("season")),
            "actual_bucket": _margin_bucket(outcome["home_margin"]),
            **{f"p_{name}": float(probs[name]) for name in BUCKETS},
        })
    return pd.DataFrame(cfb_rows), pd.DataFrame(jev_rows)


def _bucket_scores(frame: pd.DataFrame) -> pd.DataFrame:
    p = frame[[f"p_{name}" for name in BUCKETS]].to_numpy(float)
    p = p / p.sum(axis=1, keepdims=True)
    truth_index = np.array([BUCKETS.index(x) for x in frame.actual_bucket])
    y = np.eye(len(BUCKETS))[truth_index]
    out = frame[["game_id", "as_of", "season"]].copy()
    if "week" in frame:
        out["week"] = frame.week.to_numpy()
    out["brier"] = np.sum((p-y)**2, axis=1)
    out["logloss"] = -np.log(np.clip(p[np.arange(len(p)), truth_index], 1e-12, 1))
    out["rps"] = np.sum((np.cumsum(p, axis=1)[:, :-1]
                         - np.cumsum(y, axis=1)[:, :-1])**2, axis=1) / (len(BUCKETS)-1)
    return out


def bucket_evaluation(cfb: pd.DataFrame, jev: pd.DataFrame,
                      reps: int = 2000, seed: int = 20260927) -> dict:
    keys = ["game_id", "as_of", "season", "actual_bucket"]
    paired = cfb.merge(jev, on=keys, how="inner", suffixes=("_cfb", "_jev"),
                       validate="one_to_one")
    cfb_score = _bucket_scores(cfb)
    jev_score = _bucket_scores(jev)
    score = cfb_score.merge(jev_score, on=["game_id", "as_of", "season"],
                            suffixes=("_cfb", "_jev"), validate="one_to_one")
    blocks = score[["season", "week"]].astype(str).agg("-".join, axis=1).to_numpy()
    unique, rng = np.unique(blocks), np.random.default_rng(seed)
    def profile(frame: pd.DataFrame) -> dict:
        raw = frame[[f"p_{name}" for name in BUCKETS]].to_numpy(float)
        sums = raw.sum(axis=1)
        p = raw / sums[:, None]
        actual = frame.actual_bucket.to_numpy()
        true_index = np.array([BUCKETS.index(x) for x in actual])
        return {
            "mean_probability_sum": float(np.mean(sums)),
            "min_probability_sum": float(np.min(sums)),
            "max_probability_sum": float(np.max(sums)),
            "true_bucket_zero_probability_games": int(np.sum(
                p[np.arange(len(p)), true_index] <= 0)),
            "top1_accuracy": float(np.mean(np.argmax(p, axis=1) == true_index)),
            "mean_max_probability": float(np.mean(np.max(p, axis=1))),
            "by_bucket": {
                name: {"mean_p": float(np.mean(p[:, i])),
                       "event_rate": float(np.mean(actual == name))}
                for i, name in enumerate(BUCKETS)},
        }

    result = {"paired_games": int(len(paired)), "blocks": int(len(unique)),
              "cfb": {}, "jev": {}, "jev_minus_cfb": {}}
    for name in ("brier", "logloss", "rps"):
        a, b = score[f"{name}_cfb"].to_numpy(), score[f"{name}_jev"].to_numpy()
        delta, draws = b-a, []
        for _ in range(reps):
            chosen = rng.choice(unique, len(unique), replace=True)
            idx = np.concatenate([np.flatnonzero(blocks == block) for block in chosen])
            draws.append(float(np.mean(delta[idx])))
        result["cfb"][name] = float(np.mean(a))
        result["jev"][name] = float(np.mean(b))
        result["jev_minus_cfb"][name] = {
            "delta": float(np.mean(delta)),
            "ci95": [float(x) for x in np.quantile(draws, [.025, .975])],
        }
    result["cfb"]["profile"] = profile(cfb)
    result["jev"]["profile"] = profile(jev)
    return result


def blend_holdout(paired: pd.DataFrame, reps: int = 2000,
                  seed: int = 20260927) -> list[dict]:
    """Choose a CFB/Jev probability blend on prior seasons and score the next one."""
    if paired.empty:
        return []
    grid = np.linspace(0, 1, 21)
    out = []
    for target, all_target in paired.groupby("target"):
        for holdout in (2024, 2025):
            dev = all_target[all_target.season < holdout]
            test = all_target[all_target.season == holdout]
            if dev.empty or test.empty:
                continue
            def brier(frame, weight):
                p = (1-weight)*frame.p_cfb.to_numpy(float) + weight*frame.p_jev.to_numpy(float)
                return float(np.mean((p-frame.y_cfb.to_numpy(float))**2))
            scores = [brier(dev, weight) for weight in grid]
            weight = float(grid[int(np.argmin(scores))])
            p = ((1-weight)*test.p_cfb.to_numpy(float)
                 + weight*test.p_jev.to_numpy(float))
            y = test.y_cfb.to_numpy(float)
            cfb_brier = brier(test, 0.0)
            blend_brier = float(np.mean((p-y)**2))
            row_delta = (p-y)**2 - (test.p_cfb.to_numpy(float)-y)**2
            blocks = test[["season", "week"]].astype(str).agg("-".join, axis=1).to_numpy()
            unique = np.unique(blocks)
            rng = np.random.default_rng(seed + holdout)
            draws = []
            for _ in range(reps):
                chosen = rng.choice(unique, len(unique), replace=True)
                idx = np.concatenate([np.flatnonzero(blocks == block) for block in chosen])
                draws.append(float(np.mean(row_delta[idx])))
            out.append({
                "target": target, "holdout_season": holdout,
                "development_seasons": sorted(int(x) for x in dev.season.unique()),
                "selected_jev_weight": weight,
                "development_brier": float(min(scores)),
                "holdout_cfb_brier": cfb_brier,
                "holdout_blend_brier": blend_brier,
                "holdout_blend_minus_cfb_brier": blend_brier-cfb_brier,
                "holdout_blend_minus_cfb_brier_ci95": [
                    float(x) for x in np.quantile(draws, [.025, .975])],
            })
    return out


def _calibration_fit(p: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    if len(np.unique(y)) < 2:
        return None, None
    x = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))

    def loss(beta):
        eta = np.clip(beta[0] + beta[1] * x, -35, 35)
        q = 1 / (1 + np.exp(-eta))
        return -np.sum(y * np.log(np.clip(q, 1e-12, 1))
                       + (1 - y) * np.log(np.clip(1 - q, 1e-12, 1)))

    fitted = minimize(loss, np.array([0.0, 1.0]), method="BFGS")
    return float(fitted.x[0]), float(fitted.x[1])


def reliability(group: pd.DataFrame, bins: int = 10) -> list[dict]:
    z = group.copy()
    z["bin"] = pd.cut(z.p, np.linspace(0, 1, bins + 1), include_lowest=True,
                      labels=False)
    return [{"bin": int(i), "n": int(len(g)), "mean_p": float(g.p.mean()),
             "event_rate": float(g.y.mean())}
            for i, g in z.groupby("bin", observed=True)]


def metric(group: pd.DataFrame) -> dict:
    p = np.clip(group.p.to_numpy(float), 1e-8, 1 - 1e-8)
    y = group.y.to_numpy(float)
    intercept, slope = _calibration_fit(p, y)
    rel = reliability(group)
    ece = sum(cell["n"] * abs(cell["mean_p"] - cell["event_rate"])
              for cell in rel) / len(group)
    return {
        "n": int(len(group)), "brier": float(np.mean((p - y) ** 2)),
        "logloss": float(np.mean(-(y * np.log(p) + (1-y) * np.log(1-p)))),
        "mean_p": float(np.mean(p)), "event_rate": float(np.mean(y)),
        "calibration_bias": float(np.mean(p) - np.mean(y)),
        "calibration_intercept": intercept, "calibration_slope": slope,
        "ece_10": float(ece), "reliability": rel,
    }


def summaries(frame: pd.DataFrame) -> list[dict]:
    out = []
    for (source, target, season), group in frame.groupby(
            ["source", "target", "season"], dropna=False):
        out.append({"source": source, "target": target,
                    "season": int(season), **metric(group)})
    for (source, target), group in frame.groupby(["source", "target"]):
        out.append({"source": source, "target": target, "season": "ALL",
                    **metric(group)})
    return out


def paired_frame(cfb: pd.DataFrame, jev: pd.DataFrame) -> pd.DataFrame:
    if jev.empty:
        return pd.DataFrame()
    keys = ["game_id", "as_of", "season", "target"]
    paired = cfb.merge(jev, on=keys, how="inner", suffixes=("_cfb", "_jev"),
                       validate="one_to_one")
    if not paired.empty and not (paired.y_cfb == paired.y_jev).all():
        raise ValueError("paired state and ledger outcomes disagree")
    return paired


def paired_bootstrap(group: pd.DataFrame, reps: int = 2000,
                     seed: int = 20260927) -> dict:
    if group.empty:
        return {}
    p0 = np.clip(group.p_cfb.to_numpy(float), 1e-8, 1 - 1e-8)
    p1 = np.clip(group.p_jev.to_numpy(float), 1e-8, 1 - 1e-8)
    y = group.y_cfb.to_numpy(float)
    brier = (p1-y)**2 - (p0-y)**2
    logloss = (-(y*np.log(p1)+(1-y)*np.log(1-p1))
               + (y*np.log(p0)+(1-y)*np.log(1-p0)))
    blocks = group[["season", "week"]].astype(str).agg("-".join, axis=1).to_numpy()
    unique = np.unique(blocks)
    rng = np.random.default_rng(seed)
    draws_b, draws_l = [], []
    for _ in range(reps):
        chosen = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(blocks == block) for block in chosen])
        draws_b.append(float(np.mean(brier[idx])))
        draws_l.append(float(np.mean(logloss[idx])))
    return {
        "n": int(len(group)), "blocks": int(len(unique)),
        "jev_minus_cfb_brier": float(np.mean(brier)),
        "jev_minus_cfb_brier_ci95": [float(x) for x in np.quantile(draws_b, [.025, .975])],
        "jev_minus_cfb_logloss": float(np.mean(logloss)),
        "jev_minus_cfb_logloss_ci95": [float(x) for x in np.quantile(draws_l, [.025, .975])],
    }


def paired_summaries(paired: pd.DataFrame) -> list[dict]:
    if paired.empty:
        return []
    out = []
    for (target, season), group in paired.groupby(["target", "season"]):
        out.append({"target": target, "season": int(season), **paired_bootstrap(group)})
    for target, group in paired.groupby("target"):
        out.append({"target": target, "season": "ALL", **paired_bootstrap(group)})
    return out


def coherence(records: list[dict]) -> dict:
    checked = violations = bucket_checked = 0
    bucket_abs_error = {target: [] for target in TARGETS}
    examples = []
    for record in records:
        a = record.get("answers", {})
        p = {target: _p(a.get(target)) for target in TARGETS}
        if all(p[t] is not None for t in TARGETS):
            checked += 1
            bad = []
            if not (p["home_by_21_plus"] <= p["home_by_14_plus"]
                    <= p["home_by_7_plus"] <= p["home_win"]):
                bad.append("home_tail_not_nested")
            if p["away_by_7_plus"] > 1 - p["home_win"]:
                bad.append("away_7_exceeds_away_win")
            if bad:
                violations += 1
                if len(examples) < 10:
                    examples.append({"game_id": record.get("game_id"), "rules": bad})
        buckets = (a.get("margin_bucket") or {}).get("probabilities")
        if isinstance(buckets, dict) and p["home_win"] is not None:
            required = {"home_1_3", "home_4_6", "home_7_13", "home_14_20",
                        "home_21_plus", "away_1_3", "away_4_6", "away_7_13",
                        "away_14_20", "away_21_plus"}
            if required.issubset(buckets):
                bucket_checked += 1
                implied = {
                    "home_win": sum(float(buckets[k]) for k in required if k.startswith("home_")),
                    "home_by_7_plus": sum(float(buckets[k]) for k in
                                          ("home_7_13", "home_14_20", "home_21_plus")),
                    "home_by_14_plus": float(buckets["home_14_20"])+float(buckets["home_21_plus"]),
                    "home_by_21_plus": float(buckets["home_21_plus"]),
                    "away_by_7_plus": sum(float(buckets[k]) for k in
                                          ("away_7_13", "away_14_20", "away_21_plus")),
                    "within_3": float(buckets["home_1_3"])+float(buckets["away_1_3"]),
                }
                for target in TARGETS:
                    bucket_abs_error[target].append(abs(implied[target] - p[target]))
    return {
        "direct_sets_checked": checked, "nesting_violations": violations,
        "nesting_violation_rate": violations / checked if checked else None,
        "examples": examples, "bucket_sets_checked": bucket_checked,
        "bucket_vs_direct_mae": {
            target: (float(np.mean(values)) if values else None)
            for target, values in bucket_abs_error.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=Path,
                    default=Path("artifacts/fourth_jev/football_ledger.jsonl"))
    ap.add_argument("--state-dir", type=Path,
                    default=Path("artifacts/fourth_jev/states"))
    ap.add_argument("--out", type=Path,
                    default=Path("artifacts/fourth_jev/calibration.csv"))
    ap.add_argument("--json-out", type=Path,
                    default=Path("artifacts/fourth_jev/evaluation.json"))
    args = ap.parse_args()

    cfb, _ = load_states(args.state_dir)
    if cfb.empty:
        raise SystemExit(f"no scored historical states found in {args.state_dir}")
    jev, records = load_jev(args.ledger)
    cfb_buckets, jev_buckets = load_bucket_rows(args.state_dir, records)
    paired = paired_frame(cfb, jev)
    scored = cfb if jev.empty else pd.concat([cfb, jev], ignore_index=True)
    rows = summaries(scored)
    flat = pd.DataFrame([{k: v for k, v in row.items() if k != "reliability"}
                         for row in rows])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    flat.to_csv(args.out, index=False)

    paired_games = (int(paired[["game_id", "as_of"]].drop_duplicates().shape[0])
                    if not paired.empty else 0)
    paired_stats = paired_summaries(paired)
    coherence_stats = coherence(records)
    bucket_stats = bucket_evaluation(cfb_buckets, jev_buckets)
    blend_stats = blend_holdout(paired)
    pooled = [x for x in paired_stats if x["season"] == "ALL"]
    complete = paired_games == int(cfb[["game_id", "as_of"]].drop_duplicates().shape[0])
    robust = (len(pooled) == len(TARGETS)
              and all(x["jev_minus_cfb_brier_ci95"][1] < 0 for x in pooled))
    coherent = (coherence_stats["direct_sets_checked"] == paired_games
                and coherence_stats["nesting_violations"] == 0)
    bucket_robust = all(
        value["ci95"][1] < 0
        for value in bucket_stats["jev_minus_cfb"].values())
    blend_robust = any(
        row["holdout_season"] == 2025 and row["selected_jev_weight"] > 0
        and row["holdout_blend_minus_cfb_brier_ci95"][1] < 0
        for row in blend_stats)
    candidate = complete and robust and coherent and bucket_robust and blend_robust
    failures = []
    if paired_games == 0:
        failures.append("No historical Jev forecasts are available on the exact CFB game/cutoff sample.")
    elif not complete:
        failures.append("Historical Jev coverage is incomplete on the exact CFB game/cutoff sample.")
    if not coherent:
        failures.append("Jev direct tail probabilities fail completeness or event-nesting checks.")
    if not robust:
        failures.append("Direct Jev probabilities do not show robust paired Brier improvement for every graded target.")
    if not bucket_robust:
        failures.append("The Jev margin-bucket distribution does not robustly beat the CFB margin baseline.")
    if not blend_robust:
        failures.append("No development-selected Jev blend has a statistically robust 2025 Brier gain.")
    gate_reason = (" ".join(failures) if failures else
                   "Forecast layer passes the calibration gate; market/price validation is still required before betting.")

    evaluation = {
        "contract": {
            "state": "outer-season fit on seasons strictly before target season",
            "weekly_features": "weeks strictly before game week",
            "pairing": "inner join on game_id, as_of, season and target",
            "uncertainty": "2000 bootstrap resamples of season-week blocks",
            "cfb_tail_baseline": "fold-specific Normal margin/probit benchmark with half-point integer boundaries",
        },
        "coverage": {
            "cfb_games": int(cfb[["game_id", "as_of"]].drop_duplicates().shape[0]),
            "jev_games": int(jev[["game_id", "as_of"]].drop_duplicates().shape[0]) if not jev.empty else 0,
            "paired_games": paired_games,
            "seasons": sorted(int(x) for x in cfb.season.unique()),
        },
        "calibration": rows,
        "paired_deltas": paired_stats,
        "development_selected_blend": blend_stats,
        "margin_bucket": bucket_stats,
        "jev_tail_coherence": coherence_stats,
        "signal_gate": {
            "status": "candidate_for_market_validation" if candidate else "withhold",
            "reason": gate_reason,
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(evaluation, indent=2, allow_nan=False),
                             encoding="utf-8")
    print(flat.to_string(index=False))
    print(f"\ncoverage: {evaluation['coverage']}")
    print(f"signal gate: {evaluation['signal_gate']['status']} - {evaluation['signal_gate']['reason']}")
    print(f"-> {args.out}\n-> {args.json_out}")


if __name__ == "__main__":
    main()
