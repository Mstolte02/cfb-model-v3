"""What the moneyline board gains by betting outright disagreements.

The weekly moneyline gate asks for a .20 de-vigged probability edge. That gate has
no demonstrated edge behind it - ``audit/BET_THRESHOLD_CALIBRATION.md`` says so - and
it throws away the one case that is not a matter of degree: the games where the model
and the book name DIFFERENT WINNERS. A 12-point edge on a side the book has as an
underdog is not a small disagreement dressed up; it is the model saying the favourite
is wrong, at dog prices.

This script measures exactly that split on the 2022-25 expanding-window backtest, in
three pieces, so the audit can say what the change buys rather than what the whole
market does:

* **gate**    - what the .20 rule already takes.
* **added**   - outright disagreements the .20 rule turns away. These are the rows
                ``betToPlace`` starts flagging from ``ML_FLIP_FROM_WEEK``.
* **rule**    - the two together, which is the live rule from that week onward.

The gates are parsed out of ``viz/app.js`` for the same reason
``scripts.export_bet_tracking`` parses them: a second copy in Python is a copy that
goes stale the first time someone tunes a gate.

Intervals are a season-week block bootstrap, matching ``scripts.threshold_calibration``
- two bets on the same slate share weather, injuries and a common market state, so
treating them as independent understates the interval. A moneyline's profit swings by
more than one unit, so these intervals are WIDE and are meant to be read that way.

Run:  venv/Scripts/python -m scripts.moneyline_flip_backtest
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
from scripts.betting_backtest import roi_result

PREDICTIONS = ARTIFACTS / "betting_backtest_predictions.csv"
APP_JS = ROOT / "viz" / "app.js"
OUT_JSON = ARTIFACTS / "moneyline_flip_backtest.json"
DRAWS = 4000
SEED = 20260918


def app_moneyline_rule() -> dict:
    """`BET_RULES.moneyline`, read out of viz/app.js so there is only one copy."""
    src = APP_JS.read_text(encoding="utf-8")
    body = re.search(r"const BET_RULES = \{(.*?)\n  \};", src, re.S)
    if not body:
        raise SystemExit("could not find BET_RULES in viz/app.js")
    row = re.search(r"^\s*moneyline:\s*\{([^}]*)\}", body.group(1), re.M)
    if not row:
        raise SystemExit("BET_RULES has no moneyline row")
    rule = {}
    for key in ("minModelP", "minGap"):
        m = re.search(rf"{key}:\s*([0-9.]+)", row.group(1))
        if not m:
            raise SystemExit(f"BET_RULES.moneyline has no {key}")
        rule[key] = float(m.group(1))
    return rule


def priced(d: pd.DataFrame) -> pd.DataFrame:
    """One row per settled moneyline, from the side the model would take.

    `model_p` is the model on the side taken, not on the home team, which is the
    quantity `minModelP` gates. `flip` is the app's `picksOtherSide`: the model and
    the de-vigged price are on opposite sides of even money.
    """
    z = d.dropna(subset=["moneyline_gap", "homeMoneyline", "awayMoneyline",
                         "model_home_p", "market_home_p"]).copy()
    home = z.moneyline_gap >= 0
    z["price"] = np.where(home, z.homeMoneyline, z.awayMoneyline)
    z["model_p"] = np.where(home, z.model_home_p, 1 - z.model_home_p)
    won = np.where(home, z.actual_margin > 0, z.actual_margin < 0)
    z["won"] = won
    # A tie returns the stake. College football has none, but the settle rule in the
    # app has the branch, so the measurement keeps it rather than assuming.
    z["profit"] = [0.0 if m == 0 else roi_result(bool(w), float(o))
                   for w, o, m in zip(won, z.price, z.actual_margin)]
    z["flip"] = (z.model_home_p - .5) * (z.market_home_p - .5) < 0
    return z


def summarise(z: pd.DataFrame) -> dict:
    n = len(z)
    push = int((z.profit == 0).sum()) if n else 0
    won = int((z.profit > 0).sum()) if n else 0
    lost = n - won - push
    decided = won + lost
    return {"bets": n, "won": won, "lost": lost, "push": push,
            "hit_rate": (won / decided) if decided else None,
            "units": float(z.profit.sum()) if n else 0.0,
            "roi": (float(z.profit.sum()) / decided) if decided else None}


def bootstrap_roi(z: pd.DataFrame, rng) -> dict:
    """Season-week block bootstrap of ROI. Returns None when there is nothing to draw."""
    if not len(z):
        return {"lo": None, "hi": None}
    blocks = [g.profit.to_numpy() for _, g in z.groupby(["season", "week"])]
    draws = np.empty(DRAWS)
    for i in range(DRAWS):
        pick = rng.integers(0, len(blocks), len(blocks))
        sample = np.concatenate([blocks[j] for j in pick])
        draws[i] = sample.mean() if sample.size else np.nan
    return {"lo": float(np.nanpercentile(draws, 2.5)),
            "hi": float(np.nanpercentile(draws, 97.5))}


def main() -> None:
    if not PREDICTIONS.exists():
        raise SystemExit(f"missing {PREDICTIONS}; run scripts.betting_backtest first")
    rule = app_moneyline_rule()
    z = priced(pd.read_csv(PREDICTIONS))

    gate = z[(z.model_p > rule["minModelP"]) & (z.moneyline_gap.abs() >= rule["minGap"])]
    added = z[z.flip & ~z.index.isin(gate.index)]
    both = pd.concat([gate, added])

    rng = np.random.default_rng(SEED)
    sets = {"gate": gate, "added": added, "rule": both}
    result = {
        "source": "expanding-window v4 backtest; DraftKings archived prices",
        "rule_read_from_app": rule,
        "sets": {
            name: {**summarise(s), "roi_ci95": bootstrap_roi(s, rng),
                   "by_season": [{"season": int(y), **summarise(g)}
                                 for y, g in sorted(s.groupby("season"))]}
            for name, s in sets.items()
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"-> {OUT_JSON}")
    print(f"   gates {rule}")
    for name, s in result["sets"].items():
        ci = s["roi_ci95"]
        band = "" if ci["lo"] is None else \
            f"  95% [{ci['lo'] * 100:+.1f}%, {ci['hi'] * 100:+.1f}%]"
        print(f"   {name:<6} {s['bets']:>4} bets  {s['won']}-{s['lost']}-{s['push']}  "
              f"hit {0 if s['hit_rate'] is None else s['hit_rate'] * 100:.1f}%  "
              f"units {s['units']:+.2f}  "
              f"ROI {0 if s['roi'] is None else s['roi'] * 100:+.2f}%{band}")
        for y in s["by_season"]:
            print(f"       {y['season']}  {y['bets']:>3} bets  "
                  f"ROI {0 if y['roi'] is None else y['roi'] * 100:+.2f}%")


if __name__ == "__main__":
    main()
