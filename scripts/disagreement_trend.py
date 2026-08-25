"""Does the model get better as it disagrees with the market more?

This is a different question from `threshold_calibration`, and a fairer one. That
script asked "which grid point has the best ROI", which is a maximum over fourteen
noisy estimates and is why the selection null eats the answer. This one asks whether
performance *trends* with the size of the disagreement.

A trend is much harder to fake than a maximum. Noise produces a best bucket every
time; it does not produce a monotone ordering across six buckets in a particular
direction. So the tests here are:

* **Spearman correlation between gap size and outcome**, over per-game rows, with a
  permutation null that shuffles the gap against the outcome. This is one test, not
  fourteen, and it has a direction declared in advance: if the model has real edge,
  bigger gaps should do better, so the correlation should be positive.
* **A monotonicity check** across gap deciles - how many adjacent steps go the right
  way, against the number that would by chance.
* **Calibration-in-the-large by gap bucket**: when the model says it is 15 points
  better than the price, does it win 15 points more often, or does the extra
  confidence buy nothing?

The last one is the interesting one even where ROI is hopeless. A model whose hit rate
climbs with its own disagreement has *something*, even if the vig eats it; a model
whose hit rate is flat in its own disagreement has nothing to size a bet with.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

from config import ARTIFACTS
from scripts import betting_backtest as BB


OUT_JSON = ARTIFACTS / "disagreement_trend.json"
PERMUTATIONS = 5000
SEED = 20260825
N_BUCKETS = 6


def _profit(frame: pd.DataFrame, market: str) -> pd.DataFrame:
    """Signed model side, whether it won, and the unit profit of backing it."""
    out = frame.copy()
    if market == "moneyline":
        out = out.dropna(subset=["moneyline_gap", "homeMoneyline", "awayMoneyline"])
        home = out.moneyline_gap >= 0
        won = np.where(home, out.actual_margin > 0, out.actual_margin < 0)
        odds = np.where(home, out.homeMoneyline, out.awayMoneyline)
        out["won"] = won
        out["profit"] = [BB.roi_result(bool(w), float(o)) for w, o in zip(won, odds)]
        out["gap"] = out.moneyline_gap.abs()
    elif market == "spread":
        out = out.dropna(subset=["spread", "spread_gap"])
        home = out.spread_gap >= 0
        result = np.where(home, out.actual_margin + out.spread,
                          -out.actual_margin - out.spread)
        out["won"] = result > 0
        out["profit"] = np.where(result > 0, 100/110, np.where(result < 0, -1, 0))
        out["gap"] = out.spread_gap.abs()
    else:
        out = out.dropna(subset=["overUnder", "total_gap"])
        over = out.total_gap >= 0
        result = np.where(over, out.actual_total - out.overUnder,
                          out.overUnder - out.actual_total)
        out["won"] = result > 0
        out["profit"] = np.where(result > 0, 100/110, np.where(result < 0, -1, 0))
        out["gap"] = out.total_gap.abs()
    return out[out.profit.notna()]


def _permutation_spearman(gap, value, rng) -> dict:
    """Spearman rho with a null built by shuffling the gap against the outcome.

    One test with a declared direction, so a one-sided p-value is the honest read.
    """
    rho = float(stats.spearmanr(gap, value).statistic)
    gap = np.asarray(gap, float)
    value = np.asarray(value, float)
    null = np.empty(PERMUTATIONS)
    for i in range(PERMUTATIONS):
        null[i] = stats.spearmanr(rng.permutation(gap), value).statistic
    return {"rho": rho,
            "p_one_sided_positive": float(np.mean(null >= rho)),
            "null_sd": float(null.std())}


def _buckets(frame: pd.DataFrame, n=N_BUCKETS) -> list[dict]:
    frame = frame.copy()
    frame["bucket"] = pd.qcut(frame.gap, n, labels=False, duplicates="drop")
    rows = []
    for b, g in frame.groupby("bucket"):
        rows.append({"bucket": int(b), "n": int(len(g)),
                     "gap_low": float(g.gap.min()), "gap_high": float(g.gap.max()),
                     "hit_rate": float(g.won.mean()),
                     "roi": float(g.profit.mean())})
    return rows


def _monotone(rows: list[dict], key: str) -> dict:
    """How many adjacent steps rise, against the coin-flip expectation."""
    values = [r[key] for r in rows]
    steps = [b - a for a, b in zip(values, values[1:])]
    up = sum(1 for s in steps if s > 0)
    return {"steps": len(steps), "steps_up": up,
            "expected_up_by_chance": len(steps) / 2,
            "first_to_last": values[-1] - values[0],
            "p_all_up_by_chance": float(0.5 ** len(steps)) if up == len(steps) else None}


def analyse(frame: pd.DataFrame, market: str, rng) -> dict:
    data = _profit(frame, market)
    buckets = _buckets(data)
    return {
        "n": int(len(data)),
        "spearman_gap_vs_win": _permutation_spearman(data.gap, data.won.astype(float),
                                                     rng),
        "spearman_gap_vs_profit": _permutation_spearman(data.gap, data.profit, rng),
        "buckets": buckets,
        "monotone_hit_rate": _monotone(buckets, "hit_rate"),
        "monotone_roi": _monotone(buckets, "roi"),
    }


def main():
    weekly = BB.game_market_rows(pd.read_csv(BB.PREDICTIONS))
    rng = np.random.default_rng(SEED)
    result = {
        "question": ("does performance trend with the size of the model-market "
                     "disagreement, rather than peak at one searched threshold"),
        "method": {"permutations": PERMUTATIONS, "buckets": N_BUCKETS,
                   "direction": "declared in advance: bigger gap should be better"},
        "markets": {m: analyse(weekly, m, rng)
                    for m in ("spread", "moneyline", "total")},
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for market, block in result["markets"].items():
        print(f"\n=== {market}  (n={block['n']}) ===")
        w = block["spearman_gap_vs_win"]
        p = block["spearman_gap_vs_profit"]
        print(f"  gap vs win     rho {w['rho']:+.4f}  p(one-sided) {w['p_one_sided_positive']:.3f}")
        print(f"  gap vs profit  rho {p['rho']:+.4f}  p(one-sided) {p['p_one_sided_positive']:.3f}")
        print(f"  {'bucket':>6} {'n':>6} {'gap range':>16} {'hit':>7} {'ROI':>8}")
        for r in block["buckets"]:
            span = f"{r['gap_low']:.2f}-{r['gap_high']:.2f}"
            print(f"  {r['bucket']:>6} {r['n']:>6} {span:>16} "
                  f"{r['hit_rate']:>7.3f} {r['roi']:>+8.4f}")
        mh, mr = block["monotone_hit_rate"], block["monotone_roi"]
        print(f"  hit rate rising in {mh['steps_up']}/{mh['steps']} steps "
              f"(chance {mh['expected_up_by_chance']:.1f}), "
              f"first to last {mh['first_to_last']:+.4f}")
        print(f"  ROI      rising in {mr['steps_up']}/{mr['steps']} steps, "
              f"first to last {mr['first_to_last']:+.4f}")
    print(f"\n-> {OUT_JSON}")
    return result


if __name__ == "__main__":
    main()
