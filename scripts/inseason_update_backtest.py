"""Compare alternative in-season rating updates under a strict forward replay.

The game model, preseason seed, and weekly prediction contract are unchanged.  Only
the postgame evidence-to-rating update is varied.  Each update's learning rate and
static/dynamic prediction blend are selected on seasons earlier than the held-out
season.  PGWE (postgame win expectancy) is fitted on earlier game-level advanced
stats and is never used until after the game it describes.

Run: python -m scripts.inseason_update_backtest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.special import expit, logit, ndtri
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import ARTIFACTS, GAME_YEARS, OPP_ADJ_ALPHA
from scripts.train import load_bundle
from scripts.v4_backtest import (CANDIDATES, choose_candidate, fit_predict,
                                 metric, tune)
from src import oppadj as OA
from src import talent_sources as TS
from src import v4 as V4
from src.data import load, pff, war


OUT_JSON = ARTIFACTS / "inseason_update_backtest.json"
OUT_CSV = ARTIFACTS / "inseason_update_backtest_predictions.csv"
PGWE_FEATURES = ["off_ppa", "off_success_rate", "off_explosiveness"]
VARIANTS = ["absolute_mov", "no_gap_mov", "winner_signed_mov",
            "margin_residual", "margin_emphatic_gate",
            "margin_residual_uncertainty",
            "pgwe_signed_mov", "pgwe_logit_target"]
K_GRIDS = {
    "absolute_mov": [.05, .10, .15, .20, .30, .40],
    "no_gap_mov": [.05, .10, .15, .20, .30, .40],
    "winner_signed_mov": [.05, .10, .15, .20, .30, .40],
    "margin_residual": [.10, .20, .30, .40, .50, .70],
    "margin_emphatic_gate": [.10, .20, .30, .40, .50, .70],
    "margin_residual_uncertainty": [.10, .20, .30, .40, .50, .70],
    "pgwe_signed_mov": [.05, .10, .15, .20, .30, .40],
    "pgwe_logit_target": [.05, .10, .20, .30, .50, .70],
}
BLEND_GRID = [.50, .75, 1.0]
DEFAULTS = {
    "absolute_mov": (.15, .75), "no_gap_mov": (.15, .75),
    "winner_signed_mov": (.15, .75), "margin_residual": (.30, .75),
    "margin_emphatic_gate": (.30, .75),
    "margin_residual_uncertainty": (.30, .75),
    "pgwe_signed_mov": (.15, .75), "pgwe_logit_target": (.20, .75),
}
GAMMA_GRIDS = {"margin_residual_uncertainty": [0.0, .25, .50, .75]}


def build_frames(std, talent, ret, games):
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    pff_lag = pff.build_lagged_team_talent()
    war_lag = war.lagged_team_talent({y: s.index for y, s in talent.items()})
    war_projected = war.projected_team_talent(
        {y: s.index for y, s in talent.items()})
    portal = TS.portal_features(GAME_YEARS)
    groups = TS.group_features(GAME_YEARS)
    frames = {}
    for year in GAME_YEARS:
        frame = V4.build_frame(year, std, talent, ret, od, pff_lag, war_lag,
                               granular=True)
        if frame is None:
            continue
        wp = war_projected.get(year, pd.Series(dtype=float)).reindex(frame.index)
        frame["war_projected"] = wp.fillna(0.0)
        TS.attach(frame, portal[year], groups[year])
        frame["strength"] = frame.O + frame.D
        frames[year] = frame
    return frames


def pgwe_rows(year, season_games):
    """One symmetric postgame-stat vector per scored FBS-v-FBS game."""
    stats = load.game_advanced(year)
    lookup = {(r.week, r.team, r.opponent): r
              for r in stats.dropna(subset=PGWE_FEATURES).itertuples(index=False)}
    rows = []
    for game in season_games.itertuples(index=False):
        if game.home_points == game.away_points:
            continue
        home = lookup.get((game.week, game.home_team, game.away_team))
        away = lookup.get((game.week, game.away_team, game.home_team))
        if home is None or away is None:
            continue
        values = [float(getattr(home, c)) - float(getattr(away, c))
                  for c in PGWE_FEATURES]
        rows.append({"season": year, "week": game.week,
                     "home_team": game.home_team, "away_team": game.away_team,
                     "y": float(game.home_points > game.away_points),
                     "margin": float(game.home_points - game.away_points),
                     **{name: value for name, value in zip(PGWE_FEATURES, values)}})
    return pd.DataFrame(rows)


def fit_pgwe(train_rows):
    frame = pd.concat(train_rows, ignore_index=True)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0,
                         max_iter=2000)).fit(frame[PGWE_FEATURES], frame.y)


def pgwe_for_part(model, rows, meta, actual):
    lookup = {(r.week, r.home_team, r.away_team): r
              for r in rows.itertuples(index=False)}
    out = np.asarray(actual, float).copy()
    covered = np.zeros(len(meta), dtype=bool)
    for i, r in enumerate(meta.itertuples(index=False)):
        item = lookup.get((r.week, r.home_team, r.away_team))
        if item is None:
            continue
        x = pd.DataFrame([[getattr(item, c) for c in PGWE_FEATURES]],
                         columns=PGWE_FEATURES)
        out[i] = float(model.predict_proba(x)[0, 1])
        covered[i] = True
    return out, covered


def signed_damping(gap, margin):
    """Damp expected favorite wins; amplify surprising underdog wins."""
    winner_gap = np.sign(float(margin)) * float(gap)
    raw = 2.2 / max(.50, 2.2 + .35 * winner_gap)
    return float(np.clip(raw, .50, 3.0))


def delta_for(variant, k, margin, gap, actual, expected, sigma, pgwe):
    mov = np.log(abs(float(margin)) + 1.0)
    if variant == "absolute_mov":
        damping = 2.2 / (2.2 + .35 * abs(float(gap)))
        return float(k * mov * damping * (actual - expected))
    if variant == "no_gap_mov":
        return float(k * mov * (actual - expected))
    if variant == "winner_signed_mov":
        return float(k * mov * signed_damping(gap, margin) * (actual - expected))
    if variant == "margin_emphatic_gate":
        upset = ((actual == 1.0 and expected <= .25) or
                 (actual == 0.0 and expected >= .75))
        if upset and abs(float(margin)) >= 14.0:
            return float(k * mov * signed_damping(gap, margin) *
                         (actual - expected))
        variant = "margin_residual"
    if variant in {"margin_residual", "margin_residual_uncertainty"}:
        expected_margin = float(sigma * ndtri(np.clip(expected, .01, .99)))
        standardized = np.clip((float(margin) - expected_margin) / sigma,
                               -2.5, 2.5)
        return float(k * standardized)
    if variant == "pgwe_signed_mov":
        return float(k * mov * signed_damping(gap, margin) * (pgwe - expected))
    if variant == "pgwe_logit_target":
        target = float(logit(np.clip(pgwe, .02, .98)))
        # Updating both teams by +/-delta moves their gap by 2*delta.
        return float(.5 * k * np.clip(target - gap, -6.0, 6.0))
    raise KeyError(variant)


def replay(model, frame, part, variant, k, blend, pgwe, gamma=0.0):
    X, y, home_flag, margins, meta = part
    ratings = {team: model.team_logit_strength(frame, team) for team in frame.index}
    order = meta.assign(_row=np.arange(len(meta))).sort_values(["week", "_row"])
    out = np.zeros(len(y))
    for _, slate in order.groupby("week", sort=True, dropna=False):
        changes = {}
        for _, row in slate.iterrows():
            i = int(row._row)
            home, away = row.home_team, row.away_team
            gap = ratings[home] - ratings[away] + model.hfa_coef * home_flag[i]
            dynamic = float(expit(gap))
            static = model.win_prob(X[i], home_flag[i])
            out[i] = (1.0 - blend) * static + blend * dynamic
            d = delta_for(variant, k, margins[i], gap, y[i], dynamic,
                          model.margin_sigma, pgwe[i])
            if variant == "margin_residual_uncertainty":
                # Returning production is already standardized within the target
                # season.  Low-continuity teams have a less certain preseason prior
                # and are allowed to learn faster; clipping prevents one bad input
                # from erasing the prior in a single game.
                home_mult = np.clip(np.exp(-gamma * float(frame.at[home, "returning"])),
                                    .50, 2.50)
                away_mult = np.clip(np.exp(-gamma * float(frame.at[away, "returning"])),
                                    .50, 2.50)
            else:
                home_mult = away_mult = 1.0
            changes[home] = changes.get(home, 0.0) + d * home_mult
            changes[away] = changes.get(away, 0.0) - d * away_mult
        for team, change in changes.items():
            ratings[team] += change
    return out


def tune_variant(contexts, variant):
    if not contexts:
        return (*DEFAULTS[variant], 0.0, {})
    scores = {}
    for k in K_GRIDS[variant]:
        for blend in BLEND_GRID:
            for gamma in GAMMA_GRIDS.get(variant, [0.0]):
                losses = []
                for model, frame, part, pgwe in contexts:
                    pred = replay(model, frame, part, variant, k, blend, pgwe,
                                  gamma)
                    losses.extend((pred - part[1]) ** 2)
                scores[(k, blend, gamma)] = float(np.mean(losses))
    best = min(scores, key=scores.get)
    trace = {f"{k},{blend},{gamma}": value
             for (k, blend, gamma), value in scores.items()}
    return best[0], best[1], best[2], trace


def bootstrap_difference(pred, left, right="absolute_mov", draws=5000,
                         seed=20260904):
    loss = ((pred[left] - pred.y) ** 2 - (pred[right] - pred.y) ** 2)
    blocks = [g.to_numpy(float) for _, g in
              pred.assign(_loss=loss).groupby(["season", "week"])._loss]
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for i in range(draws):
        chosen = rng.integers(0, len(blocks), len(blocks))
        samples[i] = np.concatenate([blocks[j] for j in chosen]).mean()
    return {"brier_difference_vs_current": float(loss.mean()),
            "ci95": [float(x) for x in np.quantile(samples, [.025, .975])],
            "probability_better": float(np.mean(samples < 0))}


def mark_next_game_after_emphatic_upset(frame):
    """Flag the upset winner's next appearance (pregame underdog <=25%, MOV>=14)."""
    marked = pd.Series(False, index=frame.index)
    for _, season in frame.groupby("season", sort=True):
        pending = set()
        for _, slate in season.sort_values(["week", "source_index"]).groupby("week"):
            for i, row in slate.iterrows():
                participants = {row.home_team, row.away_team}
                if pending & participants:
                    marked.at[i] = True
                    pending -= participants
            for _, row in slate.iterrows():
                p = float(row.absolute_mov)
                home_upset = row.y == 1 and p <= .25 and row.margin >= 14
                away_upset = row.y == 0 and p >= .75 and row.margin <= -14
                if home_upset:
                    pending.add(row.home_team)
                elif away_upset:
                    pending.add(row.away_team)
    return marked


def main():
    std, talent, ret, games, _ = load_bundle()
    frames = build_frames(std, talent, ret, games)
    all_parts = {name: V4.assemble(GAME_YEARS, frames, games, columns)
                 for name, columns in CANDIDATES.items()}

    # The PGWE model is itself forward-only.  Include 2020 solely as training data
    # for the first historical update season; the team model still starts in 2021.
    stats_years = [2020, *GAME_YEARS]
    game_lookup = {year: load.games(year) for year in stats_years}
    pgwe_data = {year: pgwe_rows(year, game_lookup[year]) for year in stats_years}

    folds, output = [], []
    for test in [2022, 2023, 2024, 2025]:
        pool = [year for year in GAME_YEARS if year < test]
        selected, _ = choose_candidate(all_parts, pool)
        names, parts = CANDIDATES[selected], all_parts[selected]
        knobs, _ = tune(parts, pool, names)

        validation = []
        for i in range(1, len(pool)):
            val, train = pool[i], pool[:i]
            val_model, _, _ = fit_predict(parts, train, val, names, knobs)
            pgwe_model = fit_pgwe([pgwe_data[y] for y in stats_years if y < val])
            pgwe, _ = pgwe_for_part(pgwe_model, pgwe_data[val], parts[val][4],
                                    parts[val][1])
            validation.append((val_model, frames[val], parts[val], pgwe))

        model, _, _ = fit_predict(parts, pool, test, names, knobs)
        pgwe_model = fit_pgwe([pgwe_data[y] for y in stats_years if y < test])
        pgwe, coverage = pgwe_for_part(pgwe_model, pgwe_data[test], parts[test][4],
                                       parts[test][1])
        fold = {"season": test, "selected_team_model": selected,
                "pgwe_coverage": float(coverage.mean()), "variants": {}}
        rows = parts[test][4].copy()
        rows["season"] = test
        rows["y"] = parts[test][1]
        rows["margin"] = parts[test][3]
        rows["pgwe"] = pgwe
        rows["pgwe_available"] = coverage
        for variant in VARIANTS:
            k, blend, gamma, trace = tune_variant(validation, variant)
            pred = replay(model, frames[test], parts[test], variant, k, blend, pgwe,
                          gamma)
            rows[variant] = pred
            fold["variants"][variant] = {
                "k": k, "blend": blend, "uncertainty_gamma": gamma,
                "metrics": metric(parts[test][1], pred),
                "tuning": trace}
        folds.append(fold)
        output.append(rows)
        print(f"{test}: " + "  ".join(
            f"{v}={fold['variants'][v]['metrics']['brier']:.4f}"
            for v in VARIANTS))

    pred = pd.concat(output, ignore_index=True)
    pred["next_after_emphatic_upset"] = mark_next_game_after_emphatic_upset(pred)
    primary = pred[pred.season >= 2023].copy()
    result = {
        "contract": "strict expanding weekly replay; update parameters and PGWE use only prior seasons",
        "pgwe_features": PGWE_FEATURES,
        "primary_window": "2023-2025 (every fold has at least one forward validation season)",
        "folds": folds, "pooled_2022_2025": {}, "pooled_2023_2025": {},
        "bootstrap_2023_2025": {}, "by_period_2023_2025": {},
        "next_game_after_emphatic_upset_2023_2025": {
            "definition": "upset winner's next game; pregame probability <=.25 and win margin >=14",
            "n": int(primary.next_after_emphatic_upset.sum()), "variants": {}}}
    for variant in VARIANTS:
        result["pooled_2022_2025"][variant] = metric(pred.y, pred[variant])
        result["pooled_2023_2025"][variant] = metric(primary.y, primary[variant])
        if variant != "absolute_mov":
            result["bootstrap_2023_2025"][variant] = bootstrap_difference(
                primary, variant)
        shocked = primary[primary.next_after_emphatic_upset]
        result["next_game_after_emphatic_upset_2023_2025"]["variants"][variant] = \
            metric(shocked.y, shocked[variant])
    for label, subset in {
            "weeks_1_4": primary[primary.week <= 4],
            "weeks_5_9": primary[(primary.week >= 5) & (primary.week <= 9)],
            "weeks_10_plus": primary[primary.week >= 10]}.items():
        result["by_period_2023_2025"][label] = {
            variant: metric(subset.y, subset[variant]) for variant in VARIANTS}
    OUT_JSON.write_text(json.dumps(result, indent=2))
    pred.to_csv(OUT_CSV, index=False)
    print("\nPrimary 2023-25 pooled Brier:")
    for variant in VARIANTS:
        item = result["pooled_2023_2025"][variant]
        boot = result["bootstrap_2023_2025"].get(variant)
        suffix = "" if boot is None else f"  diff={boot['brier_difference_vs_current']:+.5f}"
        print(f"  {variant:<22} {item['brier']:.5f}{suffix}")
    print(f"-> {OUT_JSON}\n-> {OUT_CSV}")


if __name__ == "__main__":
    main()
