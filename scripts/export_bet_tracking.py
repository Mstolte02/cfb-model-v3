"""viz/data/bet_tracking.json: how the flagged bets would have done, at a flat stake.

The Tracking tab has two halves and they answer different questions.

The LIVE half is graded in the browser, from the same `predict()` and `BET_RULES` the
Market board flags with, so a bet can never appear in the tracker that the board did
not show or vice versa. Nothing in this file feeds it.

This file is the BACKTEST half: the same three rules applied to the expanding-window
v4 backtest over 2022-25, which is the only place a settled record exists at all. It is
NOT a live record and the tab says so - 2026 has not played a game yet.

Two things it is careful about:

  - **The thresholds are read from the app, not restated here.** BET_RULES lives in
    viz/app.js and is parsed out of it, because a second copy in Python is a copy that
    goes stale the first time someone tunes a gate. If the parse fails the build fails.
  - **The threshold CURVE ships beside the record**, so the tab can show what the rest
    of the range would have done. A single number at one gate invites the reading that
    the gate was chosen because it was good; the curve shows the honest picture, which
    for totals is that the record wanders either side of break-even at every threshold.

Stake is flat - `UNIT` dollars a bet, win or lose - because a flat stake is the only
one that makes ROI mean "return per dollar risked". Spreads and totals are priced at
-110 throughout; moneylines settle at the archived price for the side taken.

Run:  venv/Scripts/python -m scripts.export_bet_tracking
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import ARTIFACTS, ROOT
from scripts.betting_backtest import settle_games

VIZ = ROOT / "viz" / "data"
PREDICTIONS = ARTIFACTS / "betting_backtest_predictions.csv"
APP_JS = ROOT / "viz" / "app.js"
OUT = VIZ / "bet_tracking.json"

UNIT = 50.0

# Where each market's curve is worth plotting. Points of spread and total; the
# moneyline gate is a probability, so it gets its own scale.
CURVES = {
    "spread": [round(x, 1) for x in np.arange(2, 16.5, 1.0)],
    "total": [round(x, 1) for x in np.arange(2, 16.5, 1.0)],
    "moneyline": [round(x, 2) for x in np.arange(0.05, 0.45, 0.05)],
}


def app_bet_rules() -> dict:
    """The weekly gates, read out of viz/app.js so there is only one copy of them."""
    src = APP_JS.read_text(encoding="utf-8")
    m = re.search(r"const BET_RULES = \{(.*?)\n  \};", src, re.S)
    if not m:
        raise SystemExit("could not find BET_RULES in viz/app.js")
    body = m.group(1)
    rules = {}
    for market in ("spread", "total", "moneyline"):
        r = re.search(rf"^\s*{market}:\s*\{{([^}}]*)\}}", body, re.M)
        if not r:
            raise SystemExit(f"BET_RULES has no {market} row")
        gap = re.search(r"minGap:\s*([0-9.]+)", r.group(1))
        if not gap:
            raise SystemExit(f"BET_RULES.{market} has no minGap")
        rules[market] = float(gap.group(1))
    return rules


def summarise(z: pd.DataFrame) -> dict:
    """Record and money for one settled set. `profit` is in units of one stake."""
    n = len(z)
    if not n:
        return {"bets": 0, "won": 0, "lost": 0, "push": 0, "hit_rate": None,
                "units": 0.0, "wagered": 0.0, "profit": 0.0, "roi": None}
    push = int((z.profit == 0).sum())
    won = int((z.profit > 0).sum())
    lost = n - won - push
    decided = won + lost
    # A push returns the stake, so it is not risked capital and is left out of the
    # denominator of both hit rate and ROI. Counting it as a bet would quietly
    # flatter every number here.
    return {"bets": n, "won": won, "lost": lost, "push": push,
            "hit_rate": (won / decided) if decided else None,
            "units": float(z.profit.sum()),
            "wagered": float(decided * UNIT),
            "profit": float(z.profit.sum() * UNIT),
            "roi": (float(z.profit.sum()) / decided) if decided else None}


def main() -> None:
    if not PREDICTIONS.exists():
        raise SystemExit(f"missing {PREDICTIONS}; run scripts.betting_backtest first")
    d = pd.read_csv(PREDICTIONS)
    rules = app_bet_rules()

    markets, bets = {}, []
    for market, gap in rules.items():
        z = settle_games(d, market, gap)
        markets[market] = {"min_gap": gap, **summarise(z),
                           "by_season": [
                               {"season": int(s), **summarise(g)}
                               for s, g in sorted(z.groupby("season"))]}
        line = np.where(market == "total", z.get("overUnder"), z.get("spread"))
        for r, ln in zip(z.itertuples(), line):
            bets.append({
                "market": market, "season": int(r.season), "week": int(r.week),
                "home": r.home, "away": r.away,
                "line": None if pd.isna(ln) else float(ln),
                "score": f"{int(r.home_score)}-{int(r.away_score)}",
                "profit": round(float(r.profit) * UNIT, 2),
            })

    allz = pd.concat([settle_games(d, m, g) for m, g in rules.items()])
    curves = {m: [{"gap": t, **summarise(settle_games(d, m, t))} for t in ts]
              for m, ts in CURVES.items()}

    out = {
        "unit": UNIT,
        "price_note": "spreads and totals settle at -110; moneylines at the archived price",
        "rules": rules,
        "backtest": {
            "seasons": sorted(int(s) for s in d.season.unique()),
            "source": "expanding-window v4 backtest; no season fits its own predictions",
            "overall": summarise(allz),
            "markets": markets,
            "curves": curves,
        },
        # newest first, and capped: the tab shows a sample, not a ledger of 2,000 rows
        "recent": sorted(bets, key=lambda b: (-b["season"], -b["week"]))[:60],
    }
    OUT.write_text(json.dumps(out, allow_nan=False))
    o = out["backtest"]["overall"]
    print(f"-> {OUT}")
    print(f"   gates {rules}")
    print(f"   {o['bets']} bets, {o['won']}-{o['lost']}-{o['push']}, "
          f"wagered ${o['wagered']:,.0f}, profit ${o['profit']:,.0f}, "
          f"ROI {0 if o['roi'] is None else o['roi']:.2%}")
    for m, s in markets.items():
        print(f"   {m:<10} gap>={s['min_gap']:<5} {s['bets']:>5} bets  "
              f"{'—' if s['hit_rate'] is None else format(s['hit_rate'], '.1%'):>6}  "
              f"${s['profit']:>9,.0f}")


if __name__ == "__main__":
    main()
