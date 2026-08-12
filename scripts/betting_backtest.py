"""Leakage-safe validation of futures, spreads, moneylines and totals.

The model predictions are the expanding-window v4 backtest.  A season is never used
to fit its own probabilities, margins, or totals.  Betting thresholds are chosen on
2022-24 and reported on the untouched 2025 holdout.  CFBD's archived posted lines are
not represented as true closing lines; that timestamp is not present in the feed.

Run:  venv/Scripts/python -m scripts.betting_backtest
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import ARTIFACTS, GAME_YEARS, OPP_ADJ_ALPHA, ROOT
from scripts.train import load_bundle
from src import oppadj as OA
from src import spread as SP
from src import v4 as V4

PREDICTIONS = ARTIFACTS / "v4_backtest_predictions.csv"
OUT_CSV = ARTIFACTS / "betting_backtest_predictions.csv"
OUT_JSON = ARTIFACTS / "betting_backtest.json"
SITE_JSON = ROOT / "viz" / "data" / "betting_validation.json"
FUTURES_URL = ("https://www.sportsbettingdime.com/college-football/futures/"
               "win-totals-best-odds/past-seasons/")
BOOK_ORDER = ["DraftKings", "ESPN Bet", "Bovada", "Caesars Sportsbook",
              "Caesars", "consensus", "teamrankings"]
ALIASES = {"Appalachian State": "App State", "Hawaii": "Hawai'i",
           "Miami": "Miami (FL)", "Louisiana State": "LSU"}


def american_profit(odds: float) -> float:
    return odds / 100.0 if odds > 0 else 100.0 / -odds


def implied(odds: float) -> float:
    return 100.0 / (odds + 100.0) if odds > 0 else -odds / (-odds + 100.0)


def roi_result(won: bool, odds: float = -110.0) -> float:
    return american_profit(odds) if won else -1.0


def normalize_book(book: str) -> str:
    return "DraftKings" if book == "Draft Kings" else book


def select_line(game: dict) -> tuple[str | None, dict | None]:
    by_book = {}
    for line in game.get("lines") or []:
        by_book[normalize_book(line.get("provider") or "")] = line
    for book in BOOK_ORDER:
        if book in by_book:
            return book, by_book[book]
    return next(iter(by_book.items()), (None, None))


def point_totals() -> dict[tuple[int, int, str, str], float]:
    """Expanding-window total predictions from entering-season O/D only."""
    std, talent, ret, games, _ = load_bundle()
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    frames = {y: V4.build_frame(y, std, talent, ret, od) for y in GAME_YEARS}
    out = {}
    for test in range(2022, 2026):
        train = [y for y in GAME_YEARS if y < test and frames.get(y) is not None]
        X = np.vstack([SP.build_points_rows(frames[y], games[y])[0] for y in train])
        target = np.concatenate([SP.build_points_rows(frames[y], games[y])[1]
                                 for y in train])
        model, _ = SP.fit(X, target)
        fr = frames[test]
        for g in games[test].itertuples():
            if g.home_team not in fr.index or g.away_team not in fr.index:
                continue
            total = SP.game(model, fr, g.home_team, g.away_team,
                            bool(g.neutral_site))["total"]
            out[(test, int(g.week), g.home_team, g.away_team)] = total
    return out


def game_market_rows(pred: pd.DataFrame) -> pd.DataFrame:
    totals = point_totals()
    lookup = {(int(r.season), int(r.week), r.home_team, r.away_team): r
              for r in pred.itertuples()}
    rows = []
    for year in range(2022, 2026):
        raw = json.loads((ROOT / "data" / "raw" / f"lines_{year}.json").read_text())
        for g in raw:
            key = (year, int(g.get("week") or 0), g.get("homeTeam"), g.get("awayTeam"))
            p = lookup.get(key)
            book, line = select_line(g)
            if p is None or line is None:
                continue
            hp, ap = g.get("homeScore"), g.get("awayScore")
            if hp is None or ap is None:
                continue
            row = {"season": year, "week": key[1], "home": key[2], "away": key[3],
                   "book": book, "home_score": hp, "away_score": ap,
                   "actual_margin": hp - ap, "actual_total": hp + ap,
                   "model_home_p": float(p.p_dynamic),
                   "model_margin": float(p.pred_margin),
                   "model_total": totals.get(key), **line}
            rows.append(row)
    d = pd.DataFrame(rows)
    # Directional model-minus-market gaps. Positive always means bet the home/over.
    d["spread_gap"] = d["spread"] + d["model_margin"]
    both_ml = d.homeMoneyline.notna() & d.awayMoneyline.notna()
    ih = d.homeMoneyline.map(lambda x: implied(x) if pd.notna(x) else np.nan)
    ia = d.awayMoneyline.map(lambda x: implied(x) if pd.notna(x) else np.nan)
    d["market_home_p"] = np.where(both_ml, ih / (ih + ia), np.nan)
    d["moneyline_gap"] = d.model_home_p - d.market_home_p
    d["total_gap"] = d.model_total - d.overUnder
    return d


def clean_cell(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    return html.unescape(value).replace("\xa0", " ").strip()


def futures_rows(pred: pd.DataFrame) -> pd.DataFrame:
    """Parse the public DraftKings win-total archive and join preseason model wins."""
    page = subprocess.run(["curl.exe", "-L", "-s", FUTURES_URL], check=True,
                          capture_output=True).stdout.decode("utf-8", "replace")
    markets = []
    for year in range(2022, 2026):
        start = page.find(f"{year} College Football Win Totals")
        end = page.find(f"{year - 1} College Football Win Totals", start)
        block = page[start:end if end > start else None]
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", block, flags=re.S | re.I):
            cells = [clean_cell(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr,
                                                       flags=re.S | re.I)]
            if len(cells) < 3 or cells[0].lower() == "team":
                continue
            try:
                line = float(cells[1])
                nums = [int(x) for x in re.findall(r"[+-]\d+", " ".join(cells[2:]))]
            except ValueError:
                continue
            if len(nums) >= 2:
                markets.append({"season": year, "team": ALIASES.get(cells[0], cells[0]),
                                "win_total": line, "over_odds": nums[0],
                                "under_odds": nums[1]})
    m = pd.DataFrame(markets).drop_duplicates(["season", "team"])

    # Expected and actual regular-season wins. Conference championships and bowls do
    # not settle a preseason regular-season win total. FCS opponents MUST stay in
    # this accounting: dropping them subtracts an almost-certain win from both the
    # model and the result while leaving the sportsbook total untouched. The v4 team
    # model has no FCS ratings, so their probability is an expanding empirical FBS-v-
    # non-FBS base rate learned only from earlier seasons.
    pred_lookup = {(int(r.season), int(r.week), r.home_team, r.away_team): float(r.p_static)
                   for r in pred.itertuples()}
    team_rows = []
    for year in range(2022, 2026):
        raw = json.loads((ROOT / "data" / "raw" / f"games_{year}.json").read_text())
        prior_games = []
        for prior in range(2021, year):
            prior_games.extend(json.loads(
                (ROOT / "data" / "raw" / f"games_{prior}.json").read_text()))
        fbs_nonfbs = []
        for g in prior_games:
            hc, ac = g.get("homeClassification"), g.get("awayClassification")
            if {hc, ac} != {"fbs", "fcs"} or g.get("homePoints") is None:
                continue
            fbs_home = hc == "fbs"
            fbs_nonfbs.append((g["homePoints"] > g["awayPoints"]) == fbs_home)
        fbs_p = float(np.mean(fbs_nonfbs)) if fbs_nonfbs else .92
        accum: dict[str, dict[str, float]] = {}
        for g in raw:
            note = (g.get("notes") or "").lower()
            if (g.get("seasonType") == "postseason" or "championship" in note or
                    g.get("homePoints") is None):
                continue
            h, a = g.get("homeTeam"), g.get("awayTeam")
            key = (year, int(g.get("week") or 0), h, a)
            ph = pred_lookup.get(key)
            if ph is None:
                hc, ac = g.get("homeClassification"), g.get("awayClassification")
                if hc == "fbs" and ac != "fbs": ph = fbs_p
                elif ac == "fbs" and hc != "fbs": ph = 1 - fbs_p
                else: ph = .5
            for team, exp, won in [(h, ph, g["homePoints"] > g["awayPoints"]),
                                   (a, 1-ph, g["awayPoints"] > g["homePoints"])]:
                z = accum.setdefault(team, {"exp": 0.0, "wins": 0.0})
                z["exp"] += exp; z["wins"] += int(won)
        for team, z in accum.items():
            team_rows.append({"season": year, "team": team,
                              "model_wins": z["exp"], "actual_wins": z["wins"]})
    d = m.merge(pd.DataFrame(team_rows), on=["season", "team"], how="inner")
    d["futures_gap"] = d.model_wins - d.win_total
    return d


def settle_games(d: pd.DataFrame, market: str, threshold: float) -> pd.DataFrame:
    if market == "spread":
        z = d.dropna(subset=["spread", "spread_gap"]).copy()
        z = z[z.spread_gap.abs() >= threshold]
        home = z.spread_gap >= 0
        result = np.where(home, z.actual_margin + z.spread,
                          -z.actual_margin - z.spread)
        z["profit"] = np.where(result > 0, 100/110, np.where(result < 0, -1, 0))
        z["won"] = result > 0
    elif market == "moneyline":
        z = d.dropna(subset=["moneyline_gap", "homeMoneyline", "awayMoneyline"]).copy()
        z = z[z.moneyline_gap.abs() >= threshold]
        home = z.moneyline_gap >= 0
        won = np.where(home, z.actual_margin > 0, z.actual_margin < 0)
        odds = np.where(home, z.homeMoneyline, z.awayMoneyline)
        z["profit"] = [roi_result(bool(w), float(o)) for w, o in zip(won, odds)]
        z["won"] = won
    else:
        z = d.dropna(subset=["overUnder", "total_gap"]).copy()
        z = z[z.total_gap.abs() >= threshold]
        over = z.total_gap >= 0
        result = np.where(over, z.actual_total - z.overUnder,
                          z.overUnder - z.actual_total)
        z["profit"] = np.where(result > 0, 100/110, np.where(result < 0, -1, 0))
        z["won"] = result > 0
    return z


def settle_futures(d: pd.DataFrame, threshold: float) -> pd.DataFrame:
    z = d[d.futures_gap.abs() >= threshold].copy()
    over = z.futures_gap >= 0
    result = np.where(over, z.actual_wins - z.win_total, z.win_total - z.actual_wins)
    odds = np.where(over, z.over_odds, z.under_odds)
    z["profit"] = [0.0 if r == 0 else roi_result(r > 0, o)
                   for r, o in zip(result, odds)]
    z["won"] = result > 0
    return z


def summary(z: pd.DataFrame) -> dict:
    n = len(z)
    return {"bets": n, "wins": int(z.won.sum()) if n else 0,
            "hit_rate": float(z.won.mean()) if n else None,
            "units": float(z.profit.sum()) if n else 0.0,
            "roi": float(z.profit.mean()) if n else None}


def validate(d: pd.DataFrame, market: str, grid: list[float], futures=False) -> dict:
    settle = settle_futures if futures else lambda x, t: settle_games(x, market, t)
    dev = d[d.season <= 2024]
    curves = []
    for threshold in grid:
        s = summary(settle(dev, threshold))
        curves.append({"threshold": threshold, **s})
    eligible = [x for x in curves if x["bets"] >= (60 if futures else 150)]
    # Conservative selection: positive ROI first, then sample size; no 2025 peeking.
    positive = [x for x in eligible if x["roi"] is not None and x["roi"] > 0]
    selected = max(positive, key=lambda x: (x["bets"], x["roi"])) if positive else (
        max(eligible, key=lambda x: x["bets"]) if eligible else curves[0])
    threshold = selected["threshold"]
    holdout = summary(settle(d[d.season == 2025], threshold))
    return {"threshold": threshold,
            "validated_edge": bool(holdout["bets"] >= (25 if futures else 100)
                                   and holdout["roi"] is not None
                                   and holdout["roi"] > 0),
            "development_2022_2024": selected,
            "holdout_2025": holdout,
            "all_2022_2025": summary(settle(d, threshold)), "development_curve": curves}


def main():
    pred = pd.read_csv(PREDICTIONS)
    weekly = game_market_rows(pred)
    futures = futures_rows(pred)
    weekly.to_csv(OUT_CSV, index=False)
    result = {
        "method": {"prediction_contract": "strict expanding window",
                   "threshold_contract": "selected on 2022-24; 2025 untouched holdout",
                   "weekly_lines": "CFBD archived posted line; not asserted closing",
                   "spread_and_total_price": "-110 assumed where side prices absent",
                   "futures_source": FUTURES_URL},
        "coverage": {"weekly_games": len(weekly), "futures_team_seasons": len(futures)},
        "markets": {
            "spread": validate(weekly, "spread", [0, 1, 1.5, 2, 2.5, 3, 4, 5, 6]),
            "moneyline": validate(weekly, "moneyline", [0, .02, .03, .04, .05, .06, .08, .1]),
            "total": validate(weekly, "total", [0, 1, 1.5, 2, 2.5, 3, 4, 5, 6]),
            "win_total": validate(futures, "win_total", [0, .5, 1, 1.5, 2], futures=True),
        }}
    OUT_JSON.write_text(json.dumps(result, indent=2))
    SITE_JSON.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False))
    print(json.dumps(result, indent=2))
    print(f"-> {OUT_JSON}\n-> {OUT_CSV}\n-> {SITE_JSON}")


if __name__ == "__main__":
    main()
