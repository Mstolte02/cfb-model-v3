"""Does the projected margin belong to the in-season ratings or the static model?

The site prints a win probability and a spread side by side and they can name
different winners.  The cause is structural, not a rendering slip: `predict()` in
viz/app.js and `WeeklyRatingState.predict` in src/dynamic.py take the probability from
the static/dynamic blend - and the shipped 2026 state runs `dynamic_blend = 1.0`, so it
is the in-season rating alone - while `pred_margin` comes only from the preseason ridge,
which never sees a result from the current season.  On the 2026 board 58 of 710 unplayed
games have a probability and a spread that disagree about the winner.

The obvious repair is to let the margin follow the probability through the same probit
link the ensemble already uses in reverse (`p_margin = Phi(pred_margin / sigma)`), and
which `update_delta` already uses to turn a probability into an expected margin:

    margin_dynamic = sigma * Phi^-1(p)

That is a proposal, not a result, so this script measures it before anything ships.

CONTRACT.  The replay is the same strict expanding window as scripts.v4_backtest: for
test season N the candidate feature set, every knob, the learning rate and the blend
were selected only on folds ending before N.  Those choices are read back from
artifacts/v4_backtest.json rather than re-tuned, so nothing here can quietly pick a
different model than the one that produced the shipped numbers.  Any free parameter
introduced here (the mix weight, the fitted probit slope) is selected on 2022-24 and
reported once on the untouched 2025 holdout, which is the rule the betting audit uses.

Run:  venv/Scripts/python -m scripts.dynamic_margin_backtest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.special import ndtri

from config import ARTIFACTS, GAME_YEARS, OPP_ADJ_ALPHA, ROOT
from scripts.betting_backtest import select_line, settle_games
from scripts.train import load_bundle
from scripts.v4_backtest import CANDIDATES, fit_predict
from src import oppadj as OA
from src import talent_sources as TS
from src import v4 as V4
from src.data import pff, war
from src.dynamic import weekly_replay

V4_BACKTEST = ARTIFACTS / "v4_backtest.json"
OUT_JSON = ARTIFACTS / "dynamic_margin_backtest.json"
OUT_CSV = ARTIFACTS / "dynamic_margin_backtest_predictions.csv"

SELECTION_SEASONS = [2022, 2023, 2024]
HOLDOUT_SEASON = 2025
# Phi^-1 is unbounded, so a probability of exactly 1 would ask for an infinite spread.
# .001/.999 caps the implied margin at about +-3.1 sigma, near 53 points at the fitted
# sigma - already past any real college scoreline, so the cap binds on nothing real.
P_CLIP = .001
MIX_GRID = [0.0, .25, .50, .75, 1.0]
# The gate the site actually flags spreads at, plus the neighbourhood, because a margin
# change that only helps at one arbitrary threshold has not helped.
SPREAD_GATES = [0.0, 3.0, 6.0, 8.0, 10.0, 14.0]


def probit_margin(p: np.ndarray, sigma: float) -> np.ndarray:
    return sigma * ndtri(np.clip(np.asarray(p, float), P_CLIP, 1 - P_CLIP))


def build_frames():
    """Frames and assembled parts, identical to scripts.v4_backtest.main."""
    std, talent, ret, games, _ = load_bundle()
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    pff_lag = pff.build_lagged_team_talent()
    war_lag = war.lagged_team_talent({y: s.index for y, s in talent.items()})
    war_projected = war.projected_team_talent({y: s.index for y, s in talent.items()})
    portal = TS.portal_features(GAME_YEARS)
    groups = TS.group_features(GAME_YEARS)

    frames = {}
    for y in GAME_YEARS:
        fr = V4.build_frame(y, std, talent, ret, od, pff_lag, war_lag, granular=True)
        if fr is None:
            continue
        wp = war_projected.get(y, pd.Series(dtype=float)).reindex(fr.index)
        fr["war_projected"] = wp.fillna(0.0)
        TS.attach(fr, portal[y], groups[y])
        fr["strength"] = fr.O + fr.D
        frames[y] = fr
    all_parts = {name: V4.assemble(GAME_YEARS, frames, games, cols)
                 for name, cols in CANDIDATES.items()}
    return frames, all_parts


def replay() -> tuple[pd.DataFrame, list[dict]]:
    """One row per graded game, with every margin variant on it."""
    if not V4_BACKTEST.exists():
        raise SystemExit(f"missing {V4_BACKTEST}; run scripts.v4_backtest first")
    recorded = {int(f["season"]): f for f in json.loads(V4_BACKTEST.read_text())["folds"]}
    frames, all_parts = build_frames()

    rows, folds = [], []
    for test in [2022, 2023, 2024, 2025]:
        fold = recorded[test]
        names = CANDIDATES[fold["selected"]]
        parts = all_parts[fold["selected"]]
        pool = [y for y in GAME_YEARS if y < test]
        mdl, p_static, pred_margin = fit_predict(parts, pool, test, names, fold["knobs"])
        p_blended, p_dyn_only = weekly_replay(
            mdl, frames[test], parts[test], fold["dynamic_k"], fold["dynamic_blend"])

        meta = parts[test][4].copy()
        meta["season"] = test
        meta["y"] = parts[test][1]
        meta["margin"] = parts[test][3]
        meta["margin_sigma"] = mdl.margin_sigma
        meta["p_static"] = p_static
        meta["p_blended"] = p_blended
        meta["p_dynamic_only"] = p_dyn_only
        meta["m_static"] = pred_margin
        meta["m_probit_blended"] = probit_margin(p_blended, mdl.margin_sigma)
        meta["m_probit_dynamic"] = probit_margin(p_dyn_only, mdl.margin_sigma)
        # Control arm. The in-season ratings carry information the preseason ridge does
        # not have, so a gain from m_probit_blended could be the new information rather
        # than the probit link. Running the STATIC probability through the same link
        # separates the two: whatever this arm gains is the link alone.
        meta["m_probit_static"] = probit_margin(p_static, mdl.margin_sigma)
        rows.append(meta)
        folds.append({"season": test, "selected": fold["selected"],
                      "knobs": fold["knobs"], "dynamic_k": fold["dynamic_k"],
                      "dynamic_blend": fold["dynamic_blend"],
                      "margin_sigma": float(mdl.margin_sigma)})
    return pd.concat(rows, ignore_index=True), folds


def fit_probit_slope(d: pd.DataFrame, seasons: list[int]) -> float:
    """Least-squares scale on Phi^-1(p), through the origin.

    `sigma` is the margin model's own residual spread, which is an assumption about
    the link rather than a measurement of it. Fitting the scale on earlier seasons
    only asks the data how many points a unit of normal score is worth.
    """
    z = d[d.season.isin(seasons)]
    x = ndtri(np.clip(z.p_blended.to_numpy(float), P_CLIP, 1 - P_CLIP))
    return float((x @ z.margin.to_numpy(float)) / (x @ x))


def margin_metrics(d: pd.DataFrame, col: str) -> dict:
    e = d[col].to_numpy(float) - d.margin.to_numpy(float)
    picked = np.sign(d[col].to_numpy(float))
    truth = np.sign(d.margin.to_numpy(float))
    played = truth != 0
    # A spread that names a different winner than the probability printed beside it is
    # the defect this whole script exists to measure, so it is reported as a number.
    disagree = (d.p_blended.to_numpy(float) - .5) * d[col].to_numpy(float) < 0
    return {"n": int(len(d)),
            "mae": float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e ** 2))),
            "bias": float(np.mean(e)),
            "winner_accuracy": float(np.mean(picked[played] == truth[played])),
            "disagrees_with_probability": float(np.mean(disagree))}


def market_frame(d: pd.DataFrame) -> pd.DataFrame:
    """Attach the archived posted line to each replayed game.

    Same source and same book preference as scripts.betting_backtest, so an ATS record
    here is comparable with the one the site publishes.
    """
    lookup = {(int(r.season), int(r.week), r.home_team, r.away_team): r
              for r in d.itertuples()}
    rows = []
    for year in sorted(d.season.unique()):
        raw = json.loads((ROOT / "data" / "raw" / f"lines_{year}.json").read_text())
        for g in raw:
            key = (int(year), int(g.get("week") or 0), g.get("homeTeam"), g.get("awayTeam"))
            p = lookup.get(key)
            _, line = select_line(g)
            hp, ap = g.get("homeScore"), g.get("awayScore")
            if p is None or line is None or hp is None or ap is None:
                continue
            if line.get("spread") is None:
                continue
            rows.append({"season": key[0], "week": key[1], "home": key[2], "away": key[3],
                         "actual_margin": hp - ap, "spread": float(line["spread"]),
                         "m_static": p.m_static,
                         "m_probit_static": p.m_probit_static,
                         "m_probit_blended": p.m_probit_blended,
                         "m_probit_fitted": p.m_probit_fitted,
                         "m_mix": p.m_mix})
    return pd.DataFrame(rows)


def ats(market: pd.DataFrame, col: str, gate: float) -> dict:
    z = market.copy()
    z["spread_gap"] = z.spread + z[col]
    z = settle_games(z, "spread", gate)
    decided = int((z.profit != 0).sum())
    return {"gate": gate, "bets": int(len(z)),
            "hit_rate": (float((z.profit > 0).sum() / decided) if decided else None),
            "roi": (float(z.profit.sum() / decided) if decided else None)}


def main() -> None:
    d, folds = replay()

    slope = fit_probit_slope(d, SELECTION_SEASONS)
    zscore = ndtri(np.clip(d.p_blended.to_numpy(float), P_CLIP, 1 - P_CLIP))
    d["m_probit_fitted"] = slope * zscore

    # Mix weight, selected on 2022-24 by MAE and then left alone.
    sel = d[d.season.isin(SELECTION_SEASONS)]
    mix_scores = {}
    for w in MIX_GRID:
        blended = (1 - w) * sel.m_static + w * sel.m_probit_blended
        mix_scores[str(w)] = float(np.mean(np.abs(blended - sel.margin)))
    best_w = min(MIX_GRID, key=lambda w: mix_scores[str(w)])
    d["m_mix"] = (1 - best_w) * d.m_static + best_w * d.m_probit_blended
    sel = d[d.season.isin(SELECTION_SEASONS)]      # now carries the derived columns

    variants = ["m_static", "m_probit_static", "m_probit_blended",
                "m_probit_dynamic", "m_probit_fitted", "m_mix"]
    result = {
        "contract": ("strict expanding window; feature set, knobs, k and blend read "
                     "back from artifacts/v4_backtest.json, never re-tuned here"),
        "selection_seasons": SELECTION_SEASONS, "holdout_season": HOLDOUT_SEASON,
        "probit_scale_fitted_on_selection": slope,
        "mix_weight_grid_mae_2022_24": mix_scores, "mix_weight_selected": best_w,
        "folds": folds,
        "margin_accuracy": {
            "pooled": {v: margin_metrics(d, v) for v in variants},
            "selection": {v: margin_metrics(sel, v) for v in variants},
            "holdout_2025": {v: margin_metrics(d[d.season == HOLDOUT_SEASON], v)
                             for v in variants},
            "by_season": {str(s): {v: margin_metrics(d[d.season == s], v)
                                   for v in variants}
                          for s in sorted(d.season.unique())},
        },
    }

    market = market_frame(d)
    graded = ["m_static", "m_probit_static", "m_probit_blended",
              "m_probit_fitted", "m_mix"]
    result["against_the_spread"] = {
        "games_with_a_line": int(len(market)),
        "pooled": {v: [ats(market, v, g) for g in SPREAD_GATES] for v in graded},
        "holdout_2025": {v: [ats(market[market.season == HOLDOUT_SEASON], v, g)
                             for g in SPREAD_GATES] for v in graded},
    }

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(result, indent=1, allow_nan=False))

    print(f"-> {OUT_JSON}\n-> {OUT_CSV}")
    print(f"\nprobit scale fitted on 2022-24: {slope:.2f} points per normal score "
          f"(model sigma runs {folds[0]['margin_sigma']:.1f}-{folds[-1]['margin_sigma']:.1f})")
    print(f"mix weight selected on 2022-24: {best_w}  (MAE {mix_scores})")
    for label in ("pooled", "selection", "holdout_2025"):
        print(f"\n{label}")
        print(f"  {'variant':<20} {'MAE':>6} {'RMSE':>7} {'bias':>7} "
              f"{'winner':>7} {'disagrees':>10}")
        for v in variants:
            s = result["margin_accuracy"][label][v]
            print(f"  {v:<20} {s['mae']:>6.2f} {s['rmse']:>7.2f} {s['bias']:>7.2f} "
                  f"{s['winner_accuracy']:>7.1%} {s['disagrees_with_probability']:>10.1%}")
    for label in ("pooled", "holdout_2025"):
        print(f"\nagainst the spread - {label}")
        for v in graded:
            cells = []
            for s in result["against_the_spread"][label][v]:
                hit = "—" if s["hit_rate"] is None else format(s["hit_rate"], ".1%")
                cells.append(f"gate {s['gate']:>4}: {s['bets']:>4} bets {hit:>6}")
            print(f"  {v:<20} " + " | ".join(cells))


if __name__ == "__main__":
    main()
