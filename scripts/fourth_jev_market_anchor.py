"""Test whether Jev adds a market residual after CFB, with season-forward fits.

Historical Jev answers were elicited after these games were played, and CFBD's
historical prices have no quote timestamp. This is an exploratory falsification
test, not a deployable betting backtest or a source of live recommendations.
"""
from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import ARTIFACTS, DATA_RAW, ROOT
from fourth_jev.ledger import state_hash
from scripts.betting_backtest import american_profit, implied


LEDGER = ARTIFACTS / "fourth_jev" / "football_ledger.jsonl"
OUT = ARTIFACTS / "fourth_jev" / "market_anchor.json"
PREDICTIONS = ARTIFACTS / "fourth_jev" / "market_anchor_predictions.csv"
FORWARD = ARTIFACTS / "fourth_jev" / "market_anchor_2026_forward.csv"
RIDGE = 20.0
EV_GATE = 0.05
MAX_DOG_ODDS = 500.0
MIN_FAVORITE_ODDS = -300.0
WEEK_LOCK_COMMITS = {3: "cee8ade", 4: "a8b23f5"}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_week_prices(week: int) -> tuple[dict[str, dict], str]:
    """Recover the board committed at the week's pregame lock."""
    commit = WEEK_LOCK_COMMITS[week]
    raw = subprocess.run(
        ["git", "show", f"{commit}:viz/data/odds.json"],
        cwd=ROOT, check=True, capture_output=True).stdout
    committed_at = subprocess.run(
        ["git", "show", "-s", "--format=%cI", commit],
        cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    rows = json.loads(raw)["weekly"]
    return {str(row["id"]): row for row in rows if row.get("week") == week}, committed_at


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, .005, .995)
    return np.log(q / (1 - q))


def source_rows() -> pd.DataFrame:
    forecast = {}
    with LEDGER.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            state = row["state"]
            if state_hash(state) != row["state_hash"]:
                raise ValueError(f"state hash mismatch for game {row['game_id']}")
            p_jev = row["answers"].get("home_win", {}).get("noul")
            p_cfb = state["existing_model"].get("home_win_probability")
            outcome = row.get("metadata", {}).get("outcome", {})
            if p_jev is None or p_cfb is None or "home_win" not in outcome:
                continue
            key = (int(state["game"]["season"]), str(row["game_id"]))
            if key in forecast:
                raise ValueError(f"duplicate forecast for {key}")
            forecast[key] = (float(p_cfb), float(p_jev),
                             int(outcome["home_win"]), int(state["game"]["week"]))

    records = []
    for year in (2023, 2024, 2025):
        games = json.loads((DATA_RAW / f"lines_{year}.json").read_text(encoding="utf-8"))
        for game in games:
            key = (year, str(game["id"]))
            if key not in forecast:
                continue
            quotes = []
            for line in game.get("lines") or []:
                hm, am = line.get("homeMoneyline"), line.get("awayMoneyline")
                if hm is None or am is None:
                    continue
                h, a = implied(float(hm)), implied(float(am))
                quotes.append((h/(h+a), float(hm), float(am)))
            if len(quotes) < 2:
                continue
            p_cfb, p_jev, home_win, week = forecast[key]
            home_score, away_score = game.get("homeScore"), game.get("awayScore")
            if home_score is None or away_score is None:
                continue
            if home_win != int(home_score > away_score):
                raise ValueError(f"outcome mismatch for {key}")
            records.append({
                "season": year, "week": week, "game_id": key[1],
                "home": game["homeTeam"], "away": game["awayTeam"],
                "home_win": home_win, "cfb_p": p_cfb, "jev_p": p_jev,
                "market_p": float(np.median([q[0] for q in quotes])),
                "market_range": float(max(q[0] for q in quotes)-min(q[0] for q in quotes)),
                "books": len(quotes),
                "best_home_ml": max(q[1] for q in quotes),
                "best_away_ml": max(q[2] for q in quotes),
            })
    return pd.DataFrame(records)


def forward_rows() -> pd.DataFrame:
    """Read only frozen 2026 board prices and snapshots recorded before kickoff."""
    from datetime import datetime

    odds = json.loads((ROOT / "viz" / "data" / "odds.json").read_text(encoding="utf-8"))
    schedule = json.loads((ROOT / "viz" / "data" / "schedule.json").read_text(encoding="utf-8"))
    finals = {str(row["id"]): row for row in schedule if row.get("f") == 1}
    frozen = {week: frozen_week_prices(week) for week in WEEK_LOCK_COMMITS}
    rows = []
    for row in odds.get("weekly", []):
        game_id = str(row.get("id"))
        week = int(row.get("week") or 0)
        if week not in frozen:
            continue
        locked_board, commit_time = frozen[week]
        original = locked_board.get(game_id) or {}
        snap = row.get("modelSnapshot") or {}
        price = (row.get("books") or {}).get("DraftKings") or {}
        original_price = (original.get("books") or {}).get("DraftKings") or {}
        if (game_id not in finals or row.get("bettingExcluded")
                or not snap.get("recordedAt") or not snap.get("lockedAt")
                or price.get("homeMoneyline") is None
                or price.get("awayMoneyline") is None
                or price != original_price
                or snap != original.get("modelSnapshot")):
            continue
        start = datetime.fromisoformat(row["start"].replace("Z", "+00:00"))
        recorded = datetime.fromisoformat(snap["recordedAt"].replace("Z", "+00:00"))
        locked = datetime.fromisoformat(snap["lockedAt"].replace("Z", "+00:00"))
        committed = datetime.fromisoformat(commit_time)
        if recorded >= start or locked >= start or committed >= start:
            continue
        final = finals[game_id]
        h = implied(float(price["homeMoneyline"]))
        a = implied(float(price["awayMoneyline"]))
        rows.append({
            "season": 2026, "week": int(row["week"]), "game_id": game_id,
            "home": row["home"], "away": row["away"],
            "home_win": int(final["hp"] > final["ap"]),
            "cfb_p": float(snap["homeWinProbability"]),
            "market_p": h/(h+a),
            "best_home_ml": float(price["homeMoneyline"]),
            "best_away_ml": float(price["awayMoneyline"]),
            "recorded_at": snap["recordedAt"], "kickoff": row["start"],
            "price_commit": WEEK_LOCK_COMMITS[week],
        })
    return pd.DataFrame(rows)


def design(frame: pd.DataFrame, arm: str) -> tuple[np.ndarray, np.ndarray]:
    market = _logit(frame.market_p.to_numpy(float))
    cfb = _logit(frame.cfb_p.to_numpy(float)) - market
    if arm == "market":
        return market, np.empty((len(frame), 0))
    if arm == "cfb":
        return market, np.column_stack([np.ones(len(frame)), cfb])
    if arm == "cfb_jev":
        jev = _logit(frame.jev_p.to_numpy(float)) - _logit(frame.cfb_p.to_numpy(float))
        return market, np.column_stack([np.ones(len(frame)), cfb, jev])
    raise ValueError(arm)


def fit(frame: pd.DataFrame, arm: str) -> np.ndarray:
    offset, X = design(frame, arm)
    if not X.shape[1]:
        return np.empty(0)
    y = frame.home_win.to_numpy(float)

    def objective(beta):
        eta = offset + X @ beta
        penalty = np.dot(beta[1:], beta[1:]) + .1*beta[0]**2
        return np.logaddexp(0, eta).sum() - y @ eta + .5*RIDGE*penalty

    result = minimize(objective, np.zeros(X.shape[1]), method="BFGS")
    if not result.success and np.max(np.abs(result.jac)) > 1e-3:
        raise RuntimeError(result.message)
    return result.x


def predict(frame: pd.DataFrame, arm: str, beta: np.ndarray) -> np.ndarray:
    offset, X = design(frame, arm)
    eta = offset + X @ beta if X.shape[1] else offset
    return 1/(1+np.exp(-np.clip(eta, -30, 30)))


def metric(frame: pd.DataFrame, col: str) -> dict:
    p = np.clip(frame[col].to_numpy(float), 1e-8, 1-1e-8)
    y = frame.home_win.to_numpy(float)
    return {"n": len(frame), "brier": float(np.mean((p-y)**2)),
            "logloss": float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p))),
            "mean_p": float(np.mean(p)), "actual": float(np.mean(y))}


def settle(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    p = frame[col].to_numpy(float)
    hprofit = np.array([american_profit(x) for x in frame.best_home_ml])
    aprofit = np.array([american_profit(x) for x in frame.best_away_ml])
    h_ev = p*hprofit-(1-p)
    a_ev = (1-p)*aprofit-p
    home = h_ev >= a_ev
    best_ev = np.maximum(h_ev, a_ev)
    price = np.where(home, frame.best_home_ml, frame.best_away_ml)
    eligible = ((best_ev >= EV_GATE) & (price <= MAX_DOG_ODDS)
                & (price >= MIN_FAVORITE_ODDS))
    z = frame.loc[eligible, ["season", "week", "game_id"]].copy()
    z["side"] = np.where(home[eligible], "home", "away")
    z["price"] = price[eligible]
    z["estimated_ev"] = best_ev[eligible]
    z["won"] = np.where(home[eligible], frame.home_win.to_numpy()[eligible] == 1,
                        frame.home_win.to_numpy()[eligible] == 0)
    z["profit"] = np.where(z.won, np.array([american_profit(x) for x in z.price]), -1.0)
    return z


def bootstrap(frame: pd.DataFrame, value: str, reps: int = 4000) -> list[float]:
    if frame.empty:
        return [None, None]
    blocks = [g[value].to_numpy(float) for _, g in frame.groupby(["season", "week"])]
    rng = np.random.default_rng(20260928)
    draws = np.empty(reps)
    for i in range(reps):
        pick = rng.integers(0, len(blocks), len(blocks))
        draws[i] = np.concatenate([blocks[j] for j in pick]).mean()
    return [float(x) for x in np.quantile(draws, [.025, .975])]


def main() -> None:
    frame = source_rows()
    if frame.empty:
        raise SystemExit("no paired Jev/CFB/market rows")
    arms = ("market", "cfb", "cfb_jev")
    coefficients = {}
    predictions = []
    for year in (2024, 2025):
        train = frame[frame.season < year]
        test = frame[frame.season == year].copy()
        coefficients[str(year)] = {}
        for arm in arms:
            beta = fit(train, arm)
            coefficients[str(year)][arm] = beta.tolist()
            test[f"p_{arm}"] = predict(test, arm, beta)
        predictions.append(test)
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(PREDICTIONS, index=False)
    report = {
        "limitations": [
            "Jev historical answers were elicited after games were played; the model's pretraining cutoff is unknown.",
            "Historical CFBD moneylines have no quote timestamp, so executable pregame prices are unverified.",
            "2025 has already been examined in prior Fourth & Jev research; it is a diagnostic holdout, not a pristine confirmation.",
        ],
        "method": {"training": "fit on all earlier seasons only",
                   "baseline": "median no-vig probability across at least two books",
                   "features": "CFB-minus-market log odds; Jev-minus-CFB log odds",
                   "ridge_penalty": RIDGE,
                   "bet_rule": f"estimated EV >= {EV_GATE:.0%}, price from {MIN_FAVORITE_ODDS:+.0f} to +{MAX_DOG_ODDS:.0f}",
                   "price": "best archived side price; quote time unknown"},
        "coverage": {"source_games": len(frame), "by_season": {
            str(y): int(n) for y, n in frame.groupby("season").size().items()}},
        "input_sha256": {
            "jev_ledger": file_sha256(LEDGER),
            **{f"lines_{year}": file_sha256(DATA_RAW / f"lines_{year}.json")
               for year in (2023, 2024, 2025)},
            "2026_board": file_sha256(ROOT / "viz" / "data" / "odds.json"),
            "2026_schedule": file_sha256(ROOT / "viz" / "data" / "schedule.json"),
        },
        "coefficients": coefficients, "forecasts": {}, "bets": {},
    }
    for year in (2024, 2025):
        test = pred[pred.season == year]
        report["forecasts"][str(year)] = {
            arm: metric(test, f"p_{arm}") for arm in arms}
        report["bets"][str(year)] = {}
        for arm in arms:
            bets = settle(test, f"p_{arm}")
            report["bets"][str(year)][arm] = {
                "n": len(bets), "units": float(bets.profit.sum()),
                "roi": float(bets.profit.mean()) if len(bets) else None,
                "roi_ci95": bootstrap(bets, "profit"),
            }
        y = test.home_win.to_numpy(float)
        cfb_loss = (test.p_cfb.to_numpy(float)-y)**2
        jev_loss = (test.p_cfb_jev.to_numpy(float)-y)**2
        paired = test[["season", "week"]].copy()
        paired["delta"] = jev_loss-cfb_loss
        report["forecasts"][str(year)]["jev_incremental_brier"] = {
            "delta": float(paired.delta.mean()),
            "ci95": bootstrap(paired, "delta"),
        }
    forward = forward_rows()
    if not forward.empty:
        beta = fit(frame, "cfb")
        forward["p_market"] = forward.market_p
        forward["p_cfb"] = predict(forward, "cfb", beta)
        forward.to_csv(FORWARD, index=False)
        report["forward_2026"] = {
            "contract": "DraftKings quote and model snapshot match committed pregame week locks; CFB-only coefficients fitted on 2023-25",
            "n": len(forward),
            "weeks": sorted(int(x) for x in forward.week.unique()),
            "coefficients": beta.tolist(),
            "market": metric(forward, "p_market"),
            "cfb_anchor": metric(forward, "p_cfb"),
        }
        bets = settle(forward, "p_cfb")
        report["forward_2026"]["paper_bets"] = {
            "n": len(bets), "units": float(bets.profit.sum()),
            "roi": float(bets.profit.mean()) if len(bets) else None,
            "roi_ci95": bootstrap(bets, "profit"),
            "by_week": {str(week): {"n": len(group),
                                   "units": float(group.profit.sum()),
                                   "roi": float(group.profit.mean())}
                        for week, group in bets.groupby("week")},
        }
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"-> {OUT}\n-> {PREDICTIONS}\n-> {FORWARD}")


if __name__ == "__main__":
    main()
