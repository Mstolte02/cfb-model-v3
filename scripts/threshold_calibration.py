"""What model gap, if any, is large enough to bet?

``scripts.betting_backtest`` already sweeps a threshold grid and picks the best
development ROI. That answers "which grid point looked best", which is not the same
question, for two reasons this script fixes:

1. **No uncertainty.** At -110 a single bet returns +0.909 or -1, so the standard
   deviation of one bet's profit is near 0.95 and the standard error of ROI over n
   bets is roughly 0.95/sqrt(n). At 200 bets that is +/-6.7pp. A threshold showing
   "+3% ROI" is not distinguishable from a coin.

2. **No selection penalty.** Taking the best of a nine-point grid is nine chances to
   find noise. The null here re-runs the entire selection on bets whose side has been
   randomised, so the reported edge is compared against the best-of-grid ROI that a
   model with no edge produces by construction.

The output is an ROI surface with blocked bootstrap intervals per threshold, plus a
recommendation that only names a threshold when its interval clears break-even and it
survives the selection null. "No threshold qualifies" is a legitimate and expected
answer, and it is reported as one rather than dressed up as the least-bad grid point.

Blocks are season-week, matching the paired bootstrap the Brier comparisons use: two
bets on the same slate share weather, injuries and a common market state, so treating
them as independent understates the interval.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import ARTIFACTS
from scripts import betting_backtest as BB


OUT_JSON = ARTIFACTS / "threshold_calibration.json"
FUTURES_CACHE = ARTIFACTS / "futures_win_totals.csv"
DRAWS = 4000
SEED = 20260825
# A recommendation needs enough bets that the interval means something at all.
MIN_BETS = 120
MIN_BETS_FUTURES = 60

GRIDS = {
    "spread": [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8],
    "total": [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8],
    "moneyline": [0, .01, .02, .03, .04, .05, .06, .08, .10, .12, .15, .20],
    "win_total": [0, .25, .5, .75, 1, 1.25, 1.5, 2, 2.5],
}


def _blocks(frame: pd.DataFrame) -> list[np.ndarray]:
    if "week" in frame.columns:
        keys = ["season", "week"]
    else:                                    # futures settle once per season
        keys = ["season"]
    return [g.profit.to_numpy(float) for _, g in frame.groupby(keys, sort=True)]


def _bootstrap_roi(frame: pd.DataFrame, rng) -> dict:
    blocks = _blocks(frame)
    if not blocks:
        return {"roi": None, "ci90": None, "p_positive": None}
    draws = np.empty(DRAWS)
    n = len(blocks)
    for i in range(DRAWS):
        pick = rng.integers(0, n, n)
        draws[i] = np.concatenate([blocks[j] for j in pick]).mean()
    return {
        "roi": float(frame.profit.mean()),
        "ci90": [float(np.quantile(draws, .05)), float(np.quantile(draws, .95))],
        "p_positive": float(np.mean(draws > 0)),
        "blocks": int(n),
    }


def _settle(frame: pd.DataFrame, market: str, threshold: float,
            futures: bool) -> pd.DataFrame:
    if futures:
        return BB.settle_futures(frame, threshold)
    return BB.settle_games(frame, market, threshold)


def _flip(frame: pd.DataFrame, market: str, futures: bool, rng) -> pd.DataFrame:
    """Randomise which side the gap points at, keeping every other property.

    This is the null of "the model orders games no better than chance" while leaving
    the schedule, the prices, the gap magnitudes and the block structure intact.
    """
    out = frame.copy()
    column = "futures_gap" if futures else f"{market}_gap"
    sign = rng.choice([-1.0, 1.0], len(out))
    out[column] = out[column].to_numpy(float) * sign
    return out


def sweep(frame: pd.DataFrame, market: str, grid, futures=False,
          rng=None) -> list[dict]:
    rng = rng or np.random.default_rng(SEED)
    rows = []
    for threshold in grid:
        settled = _settle(frame, market, threshold, futures)
        if not len(settled):
            rows.append({"threshold": threshold, "bets": 0})
            continue
        boot = _bootstrap_roi(settled, rng)
        rows.append({
            "threshold": float(threshold), "bets": int(len(settled)),
            "hit_rate": float(settled.won.mean()),
            "units": float(settled.profit.sum()), **boot,
        })
    return rows


def selection_null(frame: pd.DataFrame, market: str, grid, futures=False,
                   trials=400) -> dict:
    """Best-of-grid ROI when the model has no side-picking skill."""
    rng = np.random.default_rng(SEED + 1)
    minimum = MIN_BETS_FUTURES if futures else MIN_BETS
    best = []
    for _ in range(trials):
        shuffled = _flip(frame, market, futures, rng)
        candidates = []
        for threshold in grid:
            settled = _settle(shuffled, market, threshold, futures)
            if len(settled) >= minimum:
                candidates.append(float(settled.profit.mean()))
        if candidates:
            best.append(max(candidates))
    if not best:
        return {"trials": 0}
    best = np.asarray(best)
    return {
        "trials": int(len(best)),
        "mean_best_roi": float(best.mean()),
        "q90_best_roi": float(np.quantile(best, .90)),
        "q95_best_roi": float(np.quantile(best, .95)),
        "note": ("ROI a no-skill model reaches by picking the best of this grid; "
                 "an observed best below q95 is not evidence of an edge"),
    }


def recommend(curve: list[dict], null: dict, futures=False) -> dict:
    """Name a threshold only if its own interval clears zero and beats the null."""
    minimum = MIN_BETS_FUTURES if futures else MIN_BETS
    eligible = [row for row in curve
                if row.get("bets", 0) >= minimum and row.get("ci90")]
    qualified = [row for row in eligible
                 if row["ci90"][0] > 0 and
                 row["roi"] > null.get("q95_best_roi", float("inf"))]
    observed_best = max((row["roi"] for row in eligible), default=None)
    if qualified:
        # The smallest qualifying gap: the same edge on more bets.
        pick = min(qualified, key=lambda row: row["threshold"])
        return {"threshold": pick["threshold"], "basis": "interval clears zero and "
                "exceeds the selection null", "detail": pick}
    return {
        "threshold": None,
        "basis": "no threshold qualifies",
        "observed_best_roi": observed_best,
        "selection_null_q95": null.get("q95_best_roi"),
        "detail": ("Every grid point's 90% interval includes zero, or its ROI is "
                   "within the range a no-skill model reaches by searching the same "
                   "grid. The data cannot support a bet-to-place threshold for this "
                   "market."),
    }


def _futures_frame(pred: pd.DataFrame) -> pd.DataFrame:
    """Win-total futures, cached so a rerun does not depend on a live scrape."""
    if FUTURES_CACHE.exists():
        return pd.read_csv(FUTURES_CACHE)
    frame = BB.futures_rows(pred)
    frame.to_csv(FUTURES_CACHE, index=False)
    return frame


def main():
    pred = pd.read_csv(BB.PREDICTIONS)
    weekly = BB.game_market_rows(pred)
    futures = _futures_frame(pred)
    result = {
        "method": {
            "predictions": "strict expanding window from v4_backtest",
            "interval": "90% blocked bootstrap, blocks are season-week",
            "draws": DRAWS,
            "selection_null": ("sides randomised, best-of-grid ROI recomputed; "
                               "quantifies how much of a grid winner is search"),
            "recommendation_rule": ("smallest threshold whose 90% lower bound is "
                                    "above zero and whose ROI exceeds the null q95"),
            "price_assumption": "-110 where side prices are absent",
        },
        "coverage": {"weekly_games": int(len(weekly)),
                     "futures_team_seasons": int(len(futures)),
                     "seasons": sorted(int(s) for s in weekly.season.unique())},
        "markets": {},
    }
    for market in ("spread", "moneyline", "total"):
        curve = sweep(weekly, market, GRIDS[market])
        null = selection_null(weekly, market, GRIDS[market])
        result["markets"][market] = {
            "curve": curve, "selection_null": null,
            "recommendation": recommend(curve, null)}
    curve = sweep(futures, "win_total", GRIDS["win_total"], futures=True)
    null = selection_null(futures, "win_total", GRIDS["win_total"], futures=True)
    result["markets"]["win_total"] = {
        "curve": curve, "selection_null": null,
        "recommendation": recommend(curve, null, futures=True)}
    result["markets"]["make_cfp"] = {
        "recommendation": {
            "threshold": None, "basis": "no historical price archive",
            "detail": ("Playoff and conference-title futures have no stored "
                       "historical board, so no threshold can be calibrated. The "
                       "only futures market with an archive is season win totals."),
        }}

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for market, block in result["markets"].items():
        if "curve" not in block:
            continue
        print(f"\n=== {market} ===")
        print(f"{'gap':>6} {'bets':>6} {'hit':>7} {'ROI':>8} "
              f"{'90% interval':>20} {'P(>0)':>7}")
        for row in block["curve"]:
            if not row.get("bets"):
                continue
            ci = row.get("ci90")
            span = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
            print(f"{row['threshold']:>6} {row['bets']:>6} "
                  f"{row['hit_rate']:>7.3f} {row['roi']:>+8.4f} {span:>20} "
                  f"{row['p_positive']:>7.2f}")
        null = block["selection_null"]
        print(f"  selection null: no-skill best-of-grid ROI mean "
              f"{null.get('mean_best_roi', float('nan')):+.4f}, "
              f"q95 {null.get('q95_best_roi', float('nan')):+.4f}")
        print(f"  -> {block['recommendation']['basis']}: "
              f"{block['recommendation']['threshold']}")
    print(f"\n-> {OUT_JSON}")
    return result


if __name__ == "__main__":
    main()
