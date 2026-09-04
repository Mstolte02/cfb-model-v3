"""Forward-only test of an evidence-dependent K factor.

The candidate keeps the winning robust-margin score unchanged and lets the
learning rate move smoothly as a team accumulates games:

    K(g) = K_open * end_ratio ** (g / 11)

where ``g`` is the average number of prior games played by the two teams.  An
``end_ratio`` below one decays K; a value above one grows it; one is the nested
constant-K baseline.  Every parameter is selected using earlier seasons only.

Run: python -m scripts.inseason_moving_k_backtest
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.special import expit

from config import ARTIFACTS, GAME_YEARS
from scripts.inseason_evidence_backtest import (
    adjusted_margin_rows,
    public_pgwe_for_part,
    update_score,
)
from scripts.inseason_update_backtest import bootstrap_difference, build_frames
from scripts.train import load_bundle
from scripts.v4_backtest import CANDIDATES, choose_candidate, fit_predict, metric, tune
from src import v4 as V4
from src.data import load


OUT_JSON = ARTIFACTS / "inseason_moving_k_backtest.json"
OUT_CSV = ARTIFACTS / "inseason_moving_k_backtest_predictions.csv"
# Concentrate the sweep around the stable winners from the component backtests.
# This still includes the prior search winners, both directions of movement, and
# the exact constant-K baseline without needlessly re-tuning rejected extremes.
K_GRID = [.15, .20, .25]
BLEND_GRID = [.75, 1.0]
END_RATIO_GRID = [.50, .75, 1.0, 1.33, 2.0]
ACTUAL_WEIGHT_GRID = [.50, .75, 1.0]
RULES = ["raw_margin", "cfbd_adjusted_margin"]


def parameter_grid(rule, moving):
    values = {
        "k": K_GRID,
        "blend": BLEND_GRID,
        "actual_weight": ACTUAL_WEIGHT_GRID if rule == "cfbd_adjusted_margin" else [1.0],
        "gamma": [0.0],
        "end_ratio": END_RATIO_GRID if moving else [1.0],
    }
    keys = list(values)
    return [dict(zip(keys, item)) for item in itertools.product(
        *(values[key] for key in keys))]


def default_parameters(rule):
    return {
        "k": .20,
        "blend": .75,
        "actual_weight": .50 if rule == "cfbd_adjusted_margin" else 1.0,
        "gamma": 0.0,
        "end_ratio": 1.0,
    }


def replay(model, frame, part, rule, params, public_pgwe):
    """Replay one season, updating K from prior games without lookahead."""
    X, y, home_flag, margins, meta = part
    ratings = {team: model.team_logit_strength(frame, team) for team in frame.index}
    games_played = {team: 0 for team in frame.index}
    order = meta.assign(_row=np.arange(len(meta))).sort_values(["week", "_row"])
    pred = np.zeros(len(y))

    for _, slate in order.groupby("week", sort=True, dropna=False):
        rating_changes = {}
        games_this_slate = {}
        for _, row in slate.iterrows():
            i, home, away = int(row._row), row.home_team, row.away_team
            gap = ratings[home] - ratings[away] + model.hfa_coef * home_flag[i]
            dynamic = float(expit(gap))
            static = model.win_prob(X[i], home_flag[i])
            pred[i] = (1.0 - params["blend"]) * static + params["blend"] * dynamic

            # Eleven prior games approximates the end of a 12-game regular
            # season.  Averaging the opponents' counts preserves a symmetric,
            # zero-sum update even when schedules are uneven.
            prior_games = .5 * (games_played.get(home, 0) + games_played.get(away, 0))
            progress = prior_games / 11.0
            moving_k = params["k"] * params["end_ratio"] ** progress
            score = update_score(
                rule, margins[i], margins[i], dynamic, model.margin_sigma,
                {"sigma0": model.margin_sigma, "beta": 0.0}, y[i],
                params["actual_weight"], params["gamma"], public_pgwe[i],
            )
            delta = moving_k * score
            rating_changes[home] = rating_changes.get(home, 0.0) + delta
            rating_changes[away] = rating_changes.get(away, 0.0) - delta
            games_this_slate[home] = games_this_slate.get(home, 0) + 1
            games_this_slate[away] = games_this_slate.get(away, 0) + 1

        for team, change in rating_changes.items():
            ratings[team] += change
        for team, count in games_this_slate.items():
            games_played[team] = games_played.get(team, 0) + count
    return pred


def tune_rule(contexts, rule, moving):
    if not contexts:
        return default_parameters(rule), {}
    scores = {}
    for params in parameter_grid(rule, moving):
        losses = []
        for context in contexts:
            prediction = replay(
                context["model"], context["frame"], context["part"], rule,
                params, context["public_pgwe"],
            )
            losses.extend((prediction - context["part"][1]) ** 2)
        key = ",".join(f"{name}={value}" for name, value in params.items())
        scores[key] = float(np.mean(losses))
    winner = min(scores, key=scores.get)
    return ({item.split("=")[0]: float(item.split("=")[1])
             for item in winner.split(",")}, scores)


def make_context(model, frame, part, postgame_rows):
    public_pgwe, coverage = public_pgwe_for_part(
        postgame_rows, part[4], part[1])
    return {
        "model": model,
        "frame": frame,
        "part": part,
        "public_pgwe": public_pgwe,
        "coverage": coverage,
    }


def main():
    std, talent, ret, games, _ = load_bundle()
    frames = build_frames(std, talent, ret, games)
    all_parts = {name: V4.assemble(GAME_YEARS, frames, games, columns)
                 for name, columns in CANDIDATES.items()}
    postgame = {
        year: adjusted_margin_rows(year, load.games(year))
        for year in [2020, *GAME_YEARS]
    }

    folds, outputs = [], []
    for test in [2022, 2023, 2024, 2025]:
        pool = [year for year in GAME_YEARS if year < test]
        selected, _ = choose_candidate(all_parts, pool)
        names, parts = CANDIDATES[selected], all_parts[selected]
        knobs, _ = tune(parts, pool, names)

        validation = []
        for i in range(1, len(pool)):
            val, train = pool[i], pool[:i]
            val_model, _, _ = fit_predict(parts, train, val, names, knobs)
            validation.append(make_context(
                val_model, frames[val], parts[val], postgame[val]))

        model, _, _ = fit_predict(parts, pool, test, names, knobs)
        test_context = make_context(model, frames[test], parts[test], postgame[test])
        rows = parts[test][4].copy()
        rows["season"], rows["y"] = test, parts[test][1]
        fold = {
            "season": test,
            "selected_team_model": selected,
            "cfbd_pgwe_coverage": float(test_context["coverage"].mean()),
            "rules": {},
        }

        for rule in RULES:
            for moving in [False, True]:
                label = f"{rule}_{'moving_k' if moving else 'constant_k'}"
                params, trace = tune_rule(validation, rule, moving)
                prediction = replay(
                    model, frames[test], parts[test], rule, params,
                    test_context["public_pgwe"],
                )
                rows[label] = prediction
                fold["rules"][label] = {
                    "parameters": params,
                    "metrics": metric(parts[test][1], prediction),
                    "tuning": trace,
                }
        folds.append(fold)
        outputs.append(rows)
        print(f"{test}: " + "  ".join(
            f"{name}={item['metrics']['brier']:.5f}"
            for name, item in fold["rules"].items()), flush=True)

    predictions = pd.concat(outputs, ignore_index=True)
    primary = predictions[predictions.season >= 2023].copy()
    result = {
        "contract": "strict expanding replay; each held-out season uses parameters selected on earlier seasons only",
        "formula": "K(g) = K_open * end_ratio ** (average_prior_games / 11)",
        "primary_window": "2023-2025",
        "n_primary": int(len(primary)),
        "folds": folds,
        "pooled_2023_2025": {},
        "bootstrap_2023_2025": {},
    }
    for rule in RULES:
        constant = f"{rule}_constant_k"
        moving = f"{rule}_moving_k"
        result["pooled_2023_2025"][constant] = metric(primary.y, primary[constant])
        result["pooled_2023_2025"][moving] = metric(primary.y, primary[moving])
        result["bootstrap_2023_2025"][moving] = bootstrap_difference(
            primary, moving, right=constant)

    OUT_JSON.write_text(json.dumps(result, indent=2))
    predictions.to_csv(OUT_CSV, index=False)
    print("\nPrimary 2023-25:")
    for name, item in result["pooled_2023_2025"].items():
        print(f"  {name:<38} Brier={item['brier']:.6f}  logloss={item['logloss']:.6f}")
    for name, item in result["bootstrap_2023_2025"].items():
        lo, hi = item["ci95"]
        print(f"  {name} delta vs constant="
              f"{item['brier_difference_vs_current']:+.6f} "
              f"CI=[{lo:+.6f}, {hi:+.6f}]")
    print(f"-> {OUT_JSON}\n-> {OUT_CSV}")


if __name__ == "__main__":
    main()
