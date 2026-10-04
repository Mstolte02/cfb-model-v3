"""Season-forward test of a market-relative cover probability, never a bet feed.

The v5.1 probabilities are strict outer-fold forecasts. Archived CFBD/DraftKings
spreads lack quote timestamps and side prices, so an assumed -110 return is only a
diagnostic; it cannot establish an executable edge. No threshold is selected here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, ndtri

from config import ARTIFACTS, DATA_RAW


SOURCE = ARTIFACTS / "v5_extension_predictions.csv"
OUT = ARTIFACTS / "spread_cover_research.json"
PREDICTIONS = ARTIFACTS / "spread_cover_research_predictions.csv"
ARM = "pff_outcome_composite"
MARGIN_SIGMA = 17.5  # fixed before this test; approximate historical game-margin SD
RIDGE = 100.0
PAPER_MIN_P = .55  # fixed safety margin above 52.38% break-even at assumed -110


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows() -> pd.DataFrame:
    forecast = pd.read_csv(SOURCE)
    forecast = forecast[forecast.variant == ARM].copy()
    key_cols = ["season", "week", "home_team", "away_team"]
    if forecast.duplicated(key_cols).any():
        raise ValueError("duplicate forecast key")
    forecast = forecast.set_index(key_cols)
    result = []
    for season in (2023, 2024, 2025):
        games = json.loads((DATA_RAW / f"lines_{season}.json").read_text())
        for game in games:
            key = (season, int(game.get("week") or 0), game.get("homeTeam"),
                   game.get("awayTeam"))
            if key not in forecast.index:
                continue
            hp, ap = game.get("homeScore"), game.get("awayScore")
            if hp is None or ap is None:
                continue
            lines = [line for line in game.get("lines") or []
                     if line.get("provider") in ("DraftKings", "Draft Kings")
                     and line.get("spread") is not None]
            if len(lines) != 1:
                continue
            p = float(forecast.loc[key, "p"])
            if int(hp > ap) != int(forecast.loc[key, "y"]):
                raise ValueError(f"forecast/outcome mismatch: {key}")
            spread = float(lines[0]["spread"])
            cover_margin = float(hp) - float(ap) + spread
            if cover_margin == 0:
                continue  # a push is not a binary cover outcome
            result.append({
                "season": season, "week": key[1], "game_id": int(game["id"]),
                "home": key[2], "away": key[3], "spread": spread,
                "model_p": p, "cover_home": int(cover_margin > 0),
                "margin_gap": MARGIN_SIGMA * float(ndtri(np.clip(p, .001, .999))) + spread,
            })
    return pd.DataFrame(result)


def fit(train: pd.DataFrame) -> np.ndarray:
    x = train.margin_gap.to_numpy(float)
    y = train.cover_home.to_numpy(float)

    def loss(b):
        eta = b[0] + b[1] * x
        return (np.logaddexp(0, eta).sum() - y @ eta
                + .5 * RIDGE * (b[0] ** 2 + b[1] ** 2))

    fitted = minimize(loss, np.zeros(2), method="BFGS")
    if not fitted.success and np.max(np.abs(fitted.jac)) > 1e-3:
        raise RuntimeError(fitted.message)
    return fitted.x


def report(test: pd.DataFrame, p_col: str) -> dict:
    p = test[p_col].to_numpy(float)
    y = test.cover_home.to_numpy(float)
    return {"n": len(test), "brier": float(np.mean((p - y) ** 2)),
            "logloss": float(np.mean(np.logaddexp(0, np.log(p / (1-p)))
                                     - y * np.log(p / (1-p))))}


def paper(test: pd.DataFrame) -> dict:
    p = test.p_cover.to_numpy(float)
    home = p >= .5
    side_p = np.where(home, p, 1-p)
    eligible = side_p >= PAPER_MIN_P
    won = np.where(home, test.cover_home.to_numpy(bool),
                   ~test.cover_home.to_numpy(bool))
    profit = np.where(won, 100/110, -1.0)[eligible]
    return {"n": int(len(profit)), "wins": int(won[eligible].sum()),
            "units_assuming_minus_110": float(profit.sum()),
            "roi_assuming_minus_110": float(profit.mean()) if len(profit) else None}


def main() -> None:
    source = rows()
    if source.empty:
        raise SystemExit("no paired forecasts and DraftKings spreads")
    forecasts = []
    coefficients = {}
    for season in (2024, 2025):
        train = source[source.season < season]
        test = source[source.season == season].copy()
        b = fit(train)
        test["p_cover"] = expit(b[0] + b[1] * test.margin_gap.to_numpy(float))
        test["p_market_baseline"] = .5
        coefficients[str(season)] = b.tolist()
        forecasts.append(test)
    pred = pd.concat(forecasts, ignore_index=True)
    result = {
        "status": "research_only_no_betting_signal",
        "limitations": [
            "CFBD archived DraftKings spreads have no quote timestamp.",
            "Spread side prices are absent; paper returns assume -110.",
            "2025 has been inspected by prior project studies; this is not a pristine holdout.",
            "The 2026 board before Week 5 used older model versions, so no v5.1 forward test exists yet.",
        ],
        "method": {"model": ARM, "training": "all earlier seasons only",
                   "feature": "17.5*normal_inverse(model_win_probability) + book_home_spread",
                   "ridge": RIDGE, "paper_gate": PAPER_MIN_P},
        "input_sha256": {"forecasts": sha256(SOURCE), **{
            f"lines_{season}": sha256(DATA_RAW / f"lines_{season}.json")
            for season in (2023, 2024, 2025)}},
        "coverage": {str(s): int(n) for s, n in source.groupby("season").size().items()},
        "coefficients": coefficients,
        "by_season": {},
    }
    for season, test in pred.groupby("season"):
        result["by_season"][str(season)] = {
            "market_baseline": report(test, "p_market_baseline"),
            "market_plus_model": report(test, "p_cover"),
            "paper": paper(test),
        }
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(PREDICTIONS, index=False)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
