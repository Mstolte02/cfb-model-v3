"""Research audit: consensus disagreement plus best available moneyline.

The signal is the model probability minus the median no-vig probability across books.
Execution uses the best archived payout for the chosen side.  The 15-point cutoff is
explicitly post-hoc discovery from the expanded audit, so positive 2025 results are
reported as a research candidate, never as an untouched validation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from config import ARTIFACTS, ROOT
from scripts.betting_backtest import american_profit, implied

PREDICTIONS = ARTIFACTS / "betting_backtest_predictions.csv"
OUT = ROOT / "audit" / "book_shopping_backtest.json"
SITE = ROOT / "viz" / "data" / "betting_validation.json"
THRESHOLD = .15


def rows() -> pd.DataFrame:
    p = pd.read_csv(PREDICTIONS)
    lookup = {(int(r.season), int(r.week), r.home, r.away): r for r in p.itertuples()}
    out = []
    for year in range(2022, 2026):
        games = json.loads((ROOT / "data" / "raw" / f"lines_{year}.json").read_text())
        for game in games:
            key = (year, int(game.get("week") or 0), game.get("homeTeam"), game.get("awayTeam"))
            pred = lookup.get(key)
            if pred is None:
                continue
            books = [line for line in (game.get("lines") or [])
                     if line.get("homeMoneyline") is not None
                     and line.get("awayMoneyline") is not None]
            if not books:
                continue
            probs = []
            for line in books:
                h, a = implied(float(line["homeMoneyline"])), implied(float(line["awayMoneyline"]))
                probs.append(h / (h + a))
            out.append({
                "season": year, "week": key[1], "home": key[2], "away": key[3],
                "actual_margin": float(pred.actual_margin),
                "model_home_p": float(pred.model_home_p),
                "consensus_home_p": float(np.median(probs)),
                "best_home_ml": max(float(x["homeMoneyline"]) for x in books),
                "best_away_ml": max(float(x["awayMoneyline"]) for x in books),
                "books": len(books),
            })
    d = pd.DataFrame(out)
    d["gap"] = d.model_home_p - d.consensus_home_p
    home = d.gap >= 0
    d["bet_side"] = np.where(home, "home", "away")
    d["price"] = np.where(home, d.best_home_ml, d.best_away_ml)
    d["won"] = np.where(home, d.actual_margin > 0, d.actual_margin < 0)
    d["profit"] = [american_profit(o) if won else -1.0 for won, o in zip(d.won, d.price)]
    return d


def summary(d: pd.DataFrame) -> dict:
    return {"bets": int(len(d)), "wins": int(d.won.sum()),
            "hit_rate": float(d.won.mean()) if len(d) else None,
            "units": float(d.profit.sum()),
            "roi": float(d.profit.mean()) if len(d) else None,
            "average_price": float(d.price.mean()) if len(d) else None,
            "underdog_share": float((d.price > 0).mean()) if len(d) else None}


def cluster_bootstrap(d: pd.DataFrame, n=10000, seed=20260812) -> dict:
    """Resample season-week clusters because games in one slate are not independent."""
    clusters = [g.profit.to_numpy(float) for _, g in d.groupby(["season", "week"])]
    rng = np.random.default_rng(seed)
    roi = np.empty(n)
    for i in range(n):
        draw = rng.integers(0, len(clusters), len(clusters))
        sample = np.concatenate([clusters[j] for j in draw])
        roi[i] = sample.mean()
    lo, hi = np.quantile(roi, [.025, .975])
    return {"cluster": "season-week", "draws": n, "roi_ci_95": [float(lo), float(hi)],
            "probability_roi_above_zero": float((roi > 0).mean())}


def calibration_buckets(d: pd.DataFrame) -> list[dict]:
    """Conservative realized win-rate prior for the live uncertainty gate.

    The forward board never uses its raw model probability as certainty.  It looks up
    the matching side/edge bucket and requires the 20th percentile of a Jeffreys
    posterior to clear the selected price by another percentage point.
    """
    home = d.gap >= 0
    z = d.copy()
    z["market_side_p"] = np.where(home, z.consensus_home_p, 1-z.consensus_home_p)
    z["model_side_p"] = np.where(home, z.model_home_p, 1-z.model_home_p)
    z["edge"] = z.model_side_p-z.market_side_p
    z["side_type"] = np.where(z.price > 0, "underdog", "favorite")
    rows = []
    for side in ("underdog", "favorite"):
        for low, high in ((.15, .20), (.20, .25), (.25, 1.01)):
            g = z[(z.side_type == side) & (z.edge >= low) & (z.edge < high)]
            wins, n = int(g.won.sum()), len(g)
            if not n:
                continue
            rows.append({"side_type": side, "edge_low": low, "edge_high": high,
                "bets": n, "wins": wins, "realized_win_rate": float(g.won.mean()),
                "mean_market_probability": float(g.market_side_p.mean()),
                "mean_model_probability": float(g.model_side_p.mean()),
                "posterior_lower_80": float(beta.ppf(.20, wins+.5, n-wins+.5)),
                "posterior_lower_90": float(beta.ppf(.10, wins+.5, n-wins+.5))})
    return rows


def main():
    d = rows()
    bets = d[d.gap.abs() >= THRESHOLD].copy()
    report = {
        "label": "Consensus disagreement + best-price moneyline",
        "status": "research_candidate_forward_validate_2026",
        "validated_edge": False,
        "threshold": THRESHOLD,
        "signal": "model probability minus median no-vig probability across available books",
        "execution": "best archived moneyline for the selected side",
        "selection_warning": "15% was chosen after expanded threshold inspection; 2025 is corroboration, not a pristine holdout",
        "coverage": {"games": int(len(d)), "average_books": float(d.books.mean()),
                     "max_books": int(d.books.max())},
        "development_2022_2024": summary(bets[bets.season <= 2024]),
        "season_2025": summary(bets[bets.season == 2025]),
        "all_2022_2025": summary(bets),
        "by_season": {str(y): summary(g) for y, g in bets.groupby("season")},
        "calibration_buckets": calibration_buckets(bets),
        "uncertainty": cluster_bootstrap(bets),
        "operating_rule": "Track in 2026; require timestamped prices and positive closing-line value before promotion to a bet label.",
    }
    OUT.write_text(json.dumps(report, indent=2))
    site = json.loads(SITE.read_text()) if SITE.exists() else {"markets": {}}
    site.setdefault("research_candidates", {})["book_shopped_moneyline"] = report
    SITE.write_text(json.dumps(site, separators=(",", ":"), allow_nan=False))
    print(json.dumps(report, indent=2))
    print(f"-> {OUT}\n-> {SITE}")


if __name__ == "__main__":
    main()
