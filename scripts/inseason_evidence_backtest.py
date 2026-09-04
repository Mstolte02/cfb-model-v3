"""Forward test of score reliability and continuous in-season update rules.

This experiment starts from the shipping weekly updater and tests four questions:

* Does a stat-line adjusted margin beat the observed score margin?
* Is residual margin variance a smooth function of pregame certainty?
* Should margin magnitude receive less weight near a 50/50 game?
* Does a continuous pregame-certainty x postgame-surprise interaction help?

All postgame-margin models, heteroskedastic curves, and update hyperparameters use
only seasons earlier than the held-out season.

Run: python -m scripts.inseason_evidence_backtest
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, ndtri
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import ARTIFACTS, GAME_YEARS
from scripts.inseason_update_backtest import bootstrap_difference, build_frames
from scripts.train import load_bundle
from scripts.v4_backtest import (CANDIDATES, choose_candidate, fit_predict,
                                 metric, tune)
from src import v4 as V4
from src.data import cfbd_client, load


OUT_JSON = ARTIFACTS / "inseason_evidence_backtest.json"
OUT_CSV = ARTIFACTS / "inseason_evidence_backtest_predictions.csv"
RULES = ["current", "raw_margin", "adjusted_margin", "cfbd_adjusted_margin",
         "hetero_margin", "sign_margin_interaction", "surprise_overlay",
         "adjusted_hetero", "cfbd_adjusted_hetero", "cfbd_surprise_overlay"]
K_GRID = [.10, .15, .20, .25, .30, .40]
BLEND_GRID = [.50, .75, 1.0]
ADJUSTED_FEATURE_PATHS = {
    "total_ppa": ("totalPPA",), "plays": ("plays",), "drives": ("drives",),
    "success": ("successRate",), "explosiveness": ("explosiveness",),
    "power_success": ("powerSuccess",), "stuff_rate": ("stuffRate",),
    "line_yards": ("lineYards",), "standard_ppa": ("standardDowns", "ppa"),
    "passing_down_ppa": ("passingDowns", "ppa"),
    "rush_ppa": ("rushingPlays", "ppa"),
    "pass_ppa": ("passingPlays", "ppa"),
}
ADJUSTED_FEATURES = list(ADJUSTED_FEATURE_PATHS)


def _nested(mapping, path):
    value = mapping
    for key in path:
        value = (value or {}).get(key)
    return float(value)


def adjusted_margin_rows(year, season_games):
    """Rich postgame efficiency differences, excluding points and score margin."""
    raw = cfbd_client.game_advanced(year)
    lookup = {(g.get("week"), g.get("team"), g.get("opponent")):
              g.get("offense") or {} for g in raw}
    public_games = {(g.get("week"), g.get("homeTeam"), g.get("awayTeam")):
                    g.get("homePostgameWinProbability")
                    for g in cfbd_client.games(year)}
    rows = []
    for game in season_games.itertuples(index=False):
        if game.home_points == game.away_points:
            continue
        home = lookup.get((game.week, game.home_team, game.away_team))
        away = lookup.get((game.week, game.away_team, game.home_team))
        if home is None or away is None:
            continue
        values = {name: _nested(home, path) - _nested(away, path)
                  for name, path in ADJUSTED_FEATURE_PATHS.items()}
        public_pgwe = public_games.get((game.week, game.home_team, game.away_team))
        rows.append({"season": year, "week": game.week,
                     "home_team": game.home_team, "away_team": game.away_team,
                     "y": float(game.home_points > game.away_points),
                     "margin": float(game.home_points - game.away_points),
                     "cfbd_pgwe": public_pgwe, **values})
    return pd.DataFrame(rows)


def fit_adjusted_margin(rows):
    frame = pd.concat(rows, ignore_index=True)
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(
        frame[ADJUSTED_FEATURES], frame.margin)


def adjusted_for_part(model, rows, meta, actual_margin):
    lookup = {(r.week, r.home_team, r.away_team): r
              for r in rows.itertuples(index=False)}
    out = np.asarray(actual_margin, float).copy()
    covered = np.zeros(len(meta), dtype=bool)
    for i, row in enumerate(meta.itertuples(index=False)):
        item = lookup.get((row.week, row.home_team, row.away_team))
        if item is None:
            continue
        x = pd.DataFrame([[getattr(item, c) for c in ADJUSTED_FEATURES]],
                         columns=ADJUSTED_FEATURES)
        out[i] = float(model.predict(x)[0])
        covered[i] = True
    return out, covered


def public_pgwe_for_part(rows, meta, actual):
    lookup = {(r.week, r.home_team, r.away_team): r.cfbd_pgwe
              for r in rows.dropna(subset=["cfbd_pgwe"]).itertuples(index=False)}
    out = np.asarray(actual, float).copy()
    covered = np.zeros(len(meta), dtype=bool)
    for i, row in enumerate(meta.itertuples(index=False)):
        value = lookup.get((row.week, row.home_team, row.away_team))
        if value is not None:
            out[i], covered[i] = float(value), True
    return out, covered


def fit_sigma_curve(model, parts, years):
    """MLE sigma(d)=sigma0*(1+beta*d), d=|P-.5|, on earlier seasons."""
    residuals, distance = [], []
    for year in years:
        X, _, home, margins = parts[year][:4]
        p = np.asarray([model.win_prob(x, h) for x, h in zip(X, home)])
        predicted = np.asarray([model.pred_margin(x, h) for x, h in zip(X, home)])
        residuals.extend(np.asarray(margins) - predicted)
        distance.extend(np.abs(p - .5))
    residuals, distance = np.asarray(residuals), np.asarray(distance)

    def objective(theta):
        sigma0, beta = np.exp(theta[0]), theta[1]
        sigma = sigma0 * (1.0 + beta * distance)
        if np.any(sigma <= 1.0):
            return 1e12
        return float(np.sum(np.log(sigma) + .5 * (residuals / sigma) ** 2))

    start = [np.log(np.std(residuals, ddof=0)), 0.0]
    result = minimize(objective, start, method="L-BFGS-B",
                      bounds=[(np.log(5.0), np.log(40.0)), (-1.8, 4.0)])
    return {"sigma0": float(np.exp(result.x[0])), "beta": float(result.x[1]),
            "n": int(len(residuals)), "converged": bool(result.success)}


def parameter_grid(rule):
    base = {"k": K_GRID, "blend": BLEND_GRID,
            "actual_weight": [1.0], "gamma": [0.0]}
    if rule in {"adjusted_margin", "adjusted_hetero", "cfbd_adjusted_margin",
                "cfbd_adjusted_hetero", "cfbd_surprise_overlay"}:
        base["actual_weight"] = [0.0, .25, .50, .75, 1.0]
    if rule == "sign_margin_interaction":
        base["gamma"] = [.50, 1.0, 2.0, 4.0]
    if rule == "surprise_overlay":
        base["gamma"] = [0.0, .25, .50, 1.0, 1.50]
    if rule == "cfbd_surprise_overlay":
        # The full Cartesian grid is unnecessary after the component sweeps;
        # concentrate around the K/blend/score weights they selected.
        base["k"] = [.15, .20, .25]
        base["blend"] = [.75, 1.0]
        base["actual_weight"] = [.50, .75, 1.0]
        base["gamma"] = [0.0, .25, .50, 1.0]
    keys = list(base)
    return [dict(zip(keys, values)) for values in
            itertools.product(*(base[key] for key in keys))]


def default_parameters(rule):
    out = {"k": .15 if rule == "current" else .20, "blend": .75,
           "actual_weight": 1.0, "gamma": 0.0}
    if rule in {"adjusted_margin", "adjusted_hetero", "cfbd_adjusted_margin",
                "cfbd_adjusted_hetero", "cfbd_surprise_overlay"}:
        out["actual_weight"] = .50
    if rule == "sign_margin_interaction":
        out["gamma"] = 1.0
    if rule == "surprise_overlay":
        out["gamma"] = .50
    return out


def update_score(rule, margin, adjusted_margin, expected, sigma_base,
                 sigma_curve, actual, actual_weight, gamma, public_pgwe):
    distance = abs(float(expected) - .5)
    sigma = float(sigma_base)
    if rule in {"hetero_margin", "adjusted_hetero", "cfbd_adjusted_hetero"}:
        sigma = sigma_curve["sigma0"] * (1.0 + sigma_curve["beta"] * distance)
        sigma = float(np.clip(sigma, 5.0, 45.0))
    expected_margin = sigma * float(ndtri(np.clip(expected, .01, .99)))
    observed = float(margin)
    if rule in {"adjusted_margin", "adjusted_hetero"}:
        observed = (actual_weight * float(margin) +
                    (1.0 - actual_weight) * float(adjusted_margin))
    if rule in {"cfbd_adjusted_margin", "cfbd_adjusted_hetero",
                "cfbd_surprise_overlay"}:
        public_margin = sigma_base * float(ndtri(np.clip(public_pgwe, .01, .99)))
        observed = (actual_weight * float(margin) +
                    (1.0 - actual_weight) * public_margin)
    margin_score = float(np.clip((observed - expected_margin) / sigma, -2.5, 2.5))
    if rule == "sign_margin_interaction":
        # Pure direction at P=.5, smoothly becoming the full margin score as
        # pregame certainty grows.  No threshold or regime switch.
        margin_weight = (2.0 * distance) ** gamma
        sign_score = 2.0 * (float(actual) - float(expected))
        return float((1.0 - margin_weight) * sign_score +
                     margin_weight * margin_score)
    if rule in {"surprise_overlay", "cfbd_surprise_overlay"}:
        # Continuous replacement for an emphatic-upset gate.  It is zero when the
        # expected side wins and grows jointly with pregame certainty and margin
        # surprise when the underdog wins.  The winning margin rule stays intact.
        outcome_sign = 2.0 * float(actual) - 1.0
        favorite_lean = 2.0 * float(expected) - 1.0
        upset_strength = max(0.0, -outcome_sign * favorite_lean)
        margin_strength = min(abs(margin_score) / 2.5, 1.0)
        return float(margin_score + gamma * outcome_sign * upset_strength *
                     margin_strength)
    return margin_score


def replay(model, frame, part, rule, params, adjusted, public_pgwe, sigma_curve,
           return_trace=False):
    X, y, home_flag, margins, meta = part
    ratings = {team: model.team_logit_strength(frame, team) for team in frame.index}
    order = meta.assign(_row=np.arange(len(meta))).sort_values(["week", "_row"])
    pred = np.zeros(len(y)); dynamic_out = np.zeros(len(y))
    expected_margin_out = np.zeros(len(y)); sigma_out = np.zeros(len(y))
    for _, slate in order.groupby("week", sort=True, dropna=False):
        changes = {}
        for _, row in slate.iterrows():
            i, home, away = int(row._row), row.home_team, row.away_team
            gap = ratings[home] - ratings[away] + model.hfa_coef * home_flag[i]
            dynamic = float(expit(gap))
            static = model.win_prob(X[i], home_flag[i])
            pred[i] = (1.0 - params["blend"]) * static + params["blend"] * dynamic
            dynamic_out[i] = dynamic
            if rule == "current":
                mov = np.log(abs(float(margins[i])) + 1.0)
                damping = 2.2 / (2.2 + .35 * abs(gap))
                delta = params["k"] * mov * damping * (float(y[i]) - dynamic)
                sigma = model.margin_sigma
            else:
                sigma = model.margin_sigma
                if rule in {"hetero_margin", "adjusted_hetero",
                            "cfbd_adjusted_hetero"}:
                    sigma = sigma_curve["sigma0"] * (
                        1.0 + sigma_curve["beta"] * abs(dynamic - .5))
                    sigma = float(np.clip(sigma, 5.0, 45.0))
                score = update_score(rule, margins[i], adjusted[i], dynamic,
                                     model.margin_sigma, sigma_curve, y[i],
                                     params["actual_weight"], params["gamma"],
                                     public_pgwe[i])
                delta = params["k"] * score
            expected_margin_out[i] = sigma * ndtri(np.clip(dynamic, .01, .99))
            sigma_out[i] = sigma
            changes[home] = changes.get(home, 0.0) + delta
            changes[away] = changes.get(away, 0.0) - delta
        for team, change in changes.items():
            ratings[team] += change
    if return_trace:
        return pred, dynamic_out, expected_margin_out, sigma_out
    return pred


def tune_rule(contexts, rule):
    if not contexts:
        return default_parameters(rule), {}
    scores = {}
    for params in parameter_grid(rule):
        losses = []
        for context in contexts:
            p = replay(context["model"], context["frame"], context["part"],
                       rule, params, context["adjusted"], context["public_pgwe"],
                       context["sigma_curve"])
            losses.extend((p - context["part"][1]) ** 2)
        key = ",".join(f"{name}={value}" for name, value in params.items())
        scores[key] = float(np.mean(losses))
    winner = min(scores, key=scores.get)
    params = {item.split("=")[0]: float(item.split("=")[1])
              for item in winner.split(",")}
    return params, scores


def calibration_tables(frame):
    data = frame.copy()
    data["residual_margin"] = data.margin - data.current_expected_margin
    edges = np.linspace(0.0, 1.0, 11)
    data["probability_band"] = pd.cut(data.current_dynamic, edges,
                                      include_lowest=True)
    deciles = []
    for band, group in data.groupby("probability_band", observed=True):
        deciles.append({"band": str(band), "n": int(len(group)),
                        "mean_prediction": float(group.current_dynamic.mean()),
                        "actual_win_rate": float(group.y.mean()),
                        "calibration_error": float(group.y.mean() -
                                                   group.current_dynamic.mean()),
                        "residual_margin_mean": float(group.residual_margin.mean()),
                        "residual_margin_sd": float(group.residual_margin.std(ddof=0))})
    distance = abs(data.current_dynamic - .5)
    labels = ["45-55", "35-45 / 55-65", "25-35 / 65-75",
              "15-25 / 75-85", "0-15 / 85-100"]
    certainty = pd.cut(distance, [0, .05, .15, .25, .35, .50], labels=labels,
                       include_lowest=True)
    bands = []
    for band, group in data.groupby(certainty, observed=True):
        bands.append({"band": str(band), "n": int(len(group)),
                      "mean_absolute_certainty": float(abs(group.current_dynamic-.5).mean()),
                      "mean_absolute_calibration_error": float(abs(group.y-group.current_dynamic).mean()),
                      "residual_margin_sd": float(group.residual_margin.std(ddof=0))})
    return deciles, bands


def main():
    std, talent, ret, games, _ = load_bundle()
    frames = build_frames(std, talent, ret, games)
    all_parts = {name: V4.assemble(GAME_YEARS, frames, games, columns)
                 for name, columns in CANDIDATES.items()}
    stats_years = [2020, *GAME_YEARS]
    score_games = {year: load.games(year) for year in stats_years}
    postgame = {year: adjusted_margin_rows(year, score_games[year])
                for year in stats_years}

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
            adjusted_model = fit_adjusted_margin(
                [postgame[y] for y in stats_years if y < val])
            adjusted, _ = adjusted_for_part(adjusted_model, postgame[val],
                                            parts[val][4], parts[val][3])
            public_pgwe, _ = public_pgwe_for_part(postgame[val], parts[val][4],
                                                  parts[val][1])
            validation.append({"model": val_model, "frame": frames[val],
                               "part": parts[val], "adjusted": adjusted,
                               "public_pgwe": public_pgwe,
                               "sigma_curve": fit_sigma_curve(val_model, parts, train)})

        model, _, _ = fit_predict(parts, pool, test, names, knobs)
        adjusted_model = fit_adjusted_margin(
            [postgame[y] for y in stats_years if y < test])
        adjusted, adjusted_coverage = adjusted_for_part(
            adjusted_model, postgame[test], parts[test][4], parts[test][3])
        public_pgwe, public_coverage = public_pgwe_for_part(
            postgame[test], parts[test][4], parts[test][1])
        sigma_curve = fit_sigma_curve(model, parts, pool)
        rows = parts[test][4].copy()
        rows["season"], rows["y"], rows["margin"] = test, parts[test][1], parts[test][3]
        rows["adjusted_margin"] = adjusted
        rows["adjusted_margin_available"] = adjusted_coverage
        rows["cfbd_pgwe"] = public_pgwe
        rows["cfbd_pgwe_available"] = public_coverage
        fold = {"season": test, "selected_team_model": selected,
                "adjusted_margin_coverage": float(adjusted_coverage.mean()),
                "cfbd_pgwe_coverage": float(public_coverage.mean()),
                "sigma_curve": sigma_curve, "rules": {}}
        for rule in RULES:
            params, trace = tune_rule(validation, rule)
            values = replay(model, frames[test], parts[test], rule, params,
                            adjusted, public_pgwe, sigma_curve,
                            return_trace=(rule == "current"))
            if rule == "current":
                pred, dynamic, expected_margin, sigma = values
                rows["current_dynamic"] = dynamic
                rows["current_expected_margin"] = expected_margin
                rows["current_sigma"] = sigma
            else:
                pred = values
            rows[rule] = pred
            fold["rules"][rule] = {"parameters": params,
                                     "metrics": metric(parts[test][1], pred),
                                     "tuning": trace}
        folds.append(fold); outputs.append(rows)
        print(f"{test}: " + "  ".join(
            f"{rule}={fold['rules'][rule]['metrics']['brier']:.5f}"
            for rule in RULES))

    pred = pd.concat(outputs, ignore_index=True)
    primary = pred[pred.season >= 2023].copy()
    deciles, certainty = calibration_tables(primary)
    result = {"contract": "strict expanding replay; all postgame models and parameters use prior seasons",
              "primary_window": "2023-2025", "n_primary": int(len(primary)),
              "adjusted_margin_features": ADJUSTED_FEATURES, "folds": folds,
              "pooled_2023_2025": {}, "bootstrap_2023_2025": {},
              "calibration_by_probability_decile": deciles,
              "residual_variance_by_certainty": certainty}
    for rule in RULES:
        result["pooled_2023_2025"][rule] = metric(primary.y, primary[rule])
        if rule != "current":
            result["bootstrap_2023_2025"][rule] = bootstrap_difference(
                primary, rule, right="current")
    result["adjusted_margin_diagnostic"] = {
        "coverage": float(primary.adjusted_margin_available.mean()),
        "correlation_with_actual_margin": float(primary.adjusted_margin.corr(primary.margin)),
        "mae_vs_actual_margin": float(np.mean(abs(primary.adjusted_margin-primary.margin))),
        "sign_accuracy": float(np.mean((primary.adjusted_margin >= 0) == primary.y))}
    OUT_JSON.write_text(json.dumps(result, indent=2))
    pred.to_csv(OUT_CSV, index=False)
    print("\nPrimary 2023-25:")
    for rule in RULES:
        item = result["pooled_2023_2025"][rule]
        print(f"  {rule:<25} Brier={item['brier']:.5f}  logloss={item['logloss']:.5f}")
    print("Sigma curves:", [(f["season"], round(f["sigma_curve"]["sigma0"], 3),
                              round(f["sigma_curve"]["beta"], 3)) for f in folds])
    print(f"-> {OUT_JSON}\n-> {OUT_CSV}")


if __name__ == "__main__":
    main()
