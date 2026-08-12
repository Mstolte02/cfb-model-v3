"""Strict expanding-window selection and backtest for the reciprocal v4 model.

For outer test season N, candidate features and every numeric knob are selected only
from forward validation folds whose test season is < N.  The historical player inputs
are the leakage-safe N-1 team summaries in src.data.pff/war; target-season participant
rows and snaps never enter.

Run: python -m scripts.v4_backtest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import ARTIFACTS, GAME_YEARS, OPP_ADJ_ALPHA, ROOT
from scripts.train import load_bundle
from src import oppadj as OA
from src import v4 as V4
from src.dynamic import weekly_replay
from src.data import pff, war


OUT_JSON = ARTIFACTS / "v4_backtest.json"
OUT_CSV = ARTIFACTS / "v4_backtest_predictions.csv"

CANDIDATES = {
    "scalar": ["strength"],
    "team_only": ["O", "D"],
    "clean_core": ["O", "D", "talent", "returning"],
    "core_pff_lag": ["O", "D", "talent", "returning", "pff_lag"],
    "core_war_lag": ["O", "D", "talent", "returning", "war_lag"],
    "core_war_projected": ["O", "D", "talent", "returning", "war_projected"],
    "core_players_lag": V4.CORE_FEATURES,
    "granular_clean": [*V4.OFF_STATS, *V4.DEF_STATS, "talent", "returning"],
    "granular_players": V4.TEAM_FEATURES,
    "core_matchups": ["O", "D", "talent", "returning", *V4.INTERACTION_FEATURES],
}
DEFAULTS = {"C": .1, "alpha": 10.0, "ensemble_weight": .5,
            "probability_scale": 1.0}
# Tiny validation wins from a larger feature family are usually selection noise.
# An extension must save at least one Brier point per thousand versus the declared
# clean core before it earns production complexity. This would have rejected the
# historical player/matchup additions whose gains were only 0.0001-0.0003.
SELECTION_MIN_GAIN = .001


def attach_cfbd_elo(pred):
    """Attach CFBD pregame Elo to the exact same game rows as the v4 replay."""
    lookups = {}
    for year in sorted(pred.season.unique()):
        raw = json.loads((ROOT / "data" / "raw" / f"games_{year}.json").read_text())
        lookups[int(year)] = {
            (g.get("week"), g.get("homeTeam"), g.get("awayTeam")):
            (g.get("homePregameElo"), g.get("awayPregameElo"))
            for g in raw
        }
    values = []
    for r in pred.itertuples():
        rh, ra = lookups[int(r.season)].get(
            (r.week, r.home_team, r.away_team), (None, None))
        if rh is None or ra is None:
            values.append(np.nan)
            continue
        hfa = 0.0 if r.neutral_site else 65.0
        values.append(1.0 / (1.0 + 10 ** (-((rh - ra + hfa) / 400.0))))
    out = pred.copy()
    out["p_cfbd_elo"] = values
    return out


def paired_week_bootstrap(df, left="p_dynamic", right="p_cfbd_elo",
                          draws=5000, seed=20260812):
    """Paired block bootstrap of Brier(left)-Brier(right), by season-week."""
    d = df.dropna(subset=[left, right]).copy()
    d["loss_diff"] = (d[left] - d.y) ** 2 - (d[right] - d.y) ** 2
    blocks = [g.loss_diff.to_numpy(float) for _, g in
              d.groupby(["season", "week"], sort=True, dropna=False)]
    observed = float(d.loss_diff.mean())
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for i in range(draws):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        samples[i] = np.concatenate([blocks[j] for j in chosen]).mean()
    low, high = np.quantile(samples, [.025, .975])
    return {"estimand": f"Brier({left}) - Brier({right})",
            "n_games": int(len(d)), "n_season_week_blocks": int(len(blocks)),
            "difference": observed, "ci95": [float(low), float(high)],
            "probability_left_better": float(np.mean(samples < 0)),
            "draws": int(draws), "seed": int(seed)}


def period_metrics(df):
    periods = {"weeks_1_4": df.week <= 4,
               "weeks_5_9": (df.week >= 5) & (df.week <= 9),
               "weeks_10_plus": df.week >= 10}
    out = {}
    for name, mask in periods.items():
        d = df.loc[mask].dropna(subset=["p_cfbd_elo"])
        out[name] = {"n": int(len(d)),
                     "v4_static_brier": float(np.mean((d.p_static - d.y) ** 2)),
                     "v4_dynamic_brier": float(np.mean((d.p_dynamic - d.y) ** 2)),
                     "cfbd_elo_brier": float(np.mean((d.p_cfbd_elo - d.y) ** 2))}
    return out


def metric(y, p, margin=None, pm=None):
    y, p = np.asarray(y), np.clip(np.asarray(p), 1e-8, 1 - 1e-8)
    d = {"n": int(len(y)), "brier": float(np.mean((p - y) ** 2)),
         "logloss": float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p))),
         "accuracy": float(np.mean((p >= .5) == y))}
    if margin is not None:
        e = np.asarray(pm) - np.asarray(margin)
        d.update(margin_mae=float(np.mean(np.abs(e))),
                 margin_rmse=float(np.sqrt(np.mean(e**2))),
                 margin_bias=float(np.mean(e)))
    return d


def stack(parts, years):
    return (np.vstack([parts[y][0] for y in years]),
            np.concatenate([parts[y][1] for y in years]),
            np.concatenate([parts[y][2] for y in years]),
            np.concatenate([parts[y][3] for y in years]))


def fit_predict(parts, train, test, names, knobs):
    X, y, h, m = stack(parts, train)
    mdl = V4.fit(X, y, h, m, names, **knobs)
    p, pm = V4.predict(mdl, parts[test])
    return mdl, p, pm


def forward_score(parts, pool, names, knobs):
    scores = []
    for i in range(1, len(pool)):
        test, train = pool[i], pool[:i]
        if test not in parts or any(y not in parts for y in train):
            continue
        _, p, _ = fit_predict(parts, train, test, names, knobs)
        scores.extend((p - parts[test][1]) ** 2)
    return float(np.mean(scores)) if scores else np.inf


def choose_candidate(all_parts, pool):
    if len(pool) < 2:
        return "clean_core", {k: None for k in CANDIDATES}
    scores = {name: forward_score(all_parts[name], pool, cols, DEFAULTS)
              for name, cols in CANDIDATES.items()}
    best = min(scores, key=scores.get)
    selected = (best if scores["clean_core"] - scores[best] >= SELECTION_MIN_GAIN
                else "clean_core")
    return selected, scores


def tune(parts, pool, names):
    knobs = dict(DEFAULTS)
    if len(pool) < 2:
        return knobs, {}
    trace = {}
    grids = {
        "C": [.03, .1, .3, 1.0],
        "alpha": [1.0, 10.0, 30.0, 100.0],
        "ensemble_weight": [0.0, .25, .5, .75, 1.0],
        "probability_scale": [.7, .8, .9, 1.0, 1.1],
    }
    for key, grid in grids.items():
        scores = {}
        for value in grid:
            trial = dict(knobs, **{key: value})
            scores[str(value)] = forward_score(parts, pool, names, trial)
        best = min(grid, key=lambda v: scores[str(v)])
        knobs[key] = best
        trace[key] = scores
    return knobs, trace


def dynamic_predictions(model, frame, part, k=.15, blend=.75):
    """Pregame weekly updates on the natural-logit rating scale.

    Every game in a week is predicted from the rating at the START of that week; only
    after the full slate is predicted are its deltas applied.  This avoids letting an
    arbitrary file order leak an earlier same-week result into a later same-week game.
    """
    return weekly_replay(model, frame, part, k, blend)


def tune_dynamic(parts, frames, pool, names, knobs):
    if len(pool) < 2:
        return .15, .75, {}
    grid_k, grid_w = [.05, .10, .15, .20, .30], [0, .25, .5, .75, 1.0]
    scores = {(k, w): [] for k in grid_k for w in grid_w}
    for i in range(1, len(pool)):
        test, train = pool[i], pool[:i]
        mdl, _, _ = fit_predict(parts, train, test, names, knobs)
        for k, w in scores:
            p, _ = dynamic_predictions(mdl, frames[test], parts[test], k, w)
            scores[(k, w)].extend((p - parts[test][1]) ** 2)
    means = {f"{k},{w}": float(np.mean(v)) for (k, w), v in scores.items()}
    best = min(scores, key=lambda z: np.mean(scores[z]))
    return best[0], best[1], means


def main():
    std, talent, ret, games, _ = load_bundle()
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    pff_lag = pff.build_lagged_team_talent()
    war_lag = war.lagged_team_talent({y: s.index for y, s in talent.items()})
    war_projected = war.projected_team_talent(
        {y: s.index for y, s in talent.items()})

    frames = {}
    for y in GAME_YEARS:
        fr = V4.build_frame(y, std, talent, ret, od, pff_lag, war_lag, granular=True)
        if fr is not None:
            wp = war_projected.get(y, pd.Series(dtype=float)).reindex(fr.index)
            fr["war_projected"] = wp.fillna(0.0)
            fr.attrs["war_projected_coverage"] = float(wp.notna().mean())
            fr["strength"] = fr.O + fr.D
            frames[y] = fr
    all_parts = {name: V4.assemble(GAME_YEARS, frames, games, cols)
                 for name, cols in CANDIDATES.items()}

    folds, predictions = [], []
    for test in [2022, 2023, 2024, 2025]:
        pool = [y for y in GAME_YEARS if y < test]
        candidate, candidate_scores = choose_candidate(all_parts, pool)
        names, parts = CANDIDATES[candidate], all_parts[candidate]
        knobs, tune_trace = tune(parts, pool, names)
        k, blend, dynamic_trace = tune_dynamic(parts, frames, pool, names, knobs)
        mdl, ps, pm = fit_predict(parts, pool, test, names, knobs)
        p_dyn, p_dyn_only = dynamic_predictions(mdl, frames[test], parts[test], k, blend)
        base = metric(parts[test][1], ps, parts[test][3], pm)
        dyn = metric(parts[test][1], p_dyn)
        fold = {"season": test, "selected": candidate, "features": names,
                "knobs": knobs, "dynamic_k": k, "dynamic_blend": blend,
                "static": base, "dynamic": dyn, "candidate_scores": candidate_scores,
                "tuning": tune_trace, "dynamic_tuning": dynamic_trace}
        folds.append(fold)
        meta = parts[test][4].copy()
        meta["season"] = test; meta["y"] = parts[test][1]
        meta["margin"] = parts[test][3]; meta["p_static"] = ps
        meta["p_dynamic"] = p_dyn; meta["p_dynamic_only"] = p_dyn_only
        meta["pred_margin"] = pm; meta["selected"] = candidate
        predictions.append(meta)
        print(f"{test}: {candidate:<17} {names}  static {base['brier']:.4f}  "
              f"dynamic {dyn['brier']:.4f}  k={k:.2f} blend={blend:.2f}  {knobs}")

    pred = attach_cfbd_elo(pd.concat(predictions, ignore_index=True))
    pooled_static = metric(pred.y, pred.p_static, pred.margin, pred.pred_margin)
    pooled_dynamic = metric(pred.y, pred.p_dynamic)
    aligned = pred.dropna(subset=["p_cfbd_elo"])
    aligned_dynamic = metric(aligned.y, aligned.p_dynamic)
    cfbd_elo = metric(aligned.y, aligned.p_cfbd_elo)
    paired = paired_week_bootstrap(aligned)
    result = {"contract": "strict expanding; historical N uses no season-N player rows/snaps",
              "selection_rule": {"reference": "clean_core",
                                 "minimum_extension_brier_gain": SELECTION_MIN_GAIN},
              "candidate_features": CANDIDATES, "folds": folds,
              "pooled_static": pooled_static, "pooled_dynamic": pooled_dynamic,
              "same_game_benchmark": {"v4_dynamic": aligned_dynamic,
                                      "cfbd_pregame_elo": cfbd_elo,
                                      "paired_week_bootstrap": paired,
                                      "by_period": period_metrics(aligned)},
              "pff_coverage": {str(y): frames[y].attrs.get("pff_coverage") for y in frames},
              "war_coverage": {str(y): frames[y].attrs.get("war_coverage") for y in frames}}
    result["war_projected_coverage"] = {
        str(y): frames[y].attrs.get("war_projected_coverage") for y in frames}
    OUT_JSON.write_text(json.dumps(result, indent=2))
    pred.to_csv(OUT_CSV, index=False)
    print(f"\npooled static : {pooled_static}")
    print(f"pooled dynamic: {pooled_dynamic}")
    print(f"aligned Elo   : {cfbd_elo}")
    print(f"paired diff   : {paired}")
    print(f"-> {OUT_JSON}\n-> {OUT_CSV}")


if __name__ == "__main__":
    main()
