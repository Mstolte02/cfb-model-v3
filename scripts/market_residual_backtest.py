"""Test whether the model adds information after the sportsbook market.

This is deliberately different from grading the raw model.  Market probability,
spread and total are the baseline; the model is allowed to predict only the residual
market error.  Parameters and betting thresholds expand through time.  The final
2025 season is not used to choose a threshold.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import ARTIFACTS
from scripts.betting_backtest import american_profit

SOURCE = ARTIFACTS / "betting_backtest_predictions.csv"
OUT_CSV = ARTIFACTS / "market_residual_predictions.csv"
OUT_JSON = ARTIFACTS / "market_residual_backtest.json"
ALPHAS = [.1, 1.0, 3.0, 10.0, 30.0, 100.0]


def clip_prob(x):
    return np.clip(np.asarray(x, float), .005, .995)


def ml_design(d: pd.DataFrame):
    market = clip_prob(d.market_home_p)
    model = clip_prob(d.model_home_p)
    offset = np.log(market / (1 - market))
    delta = np.log(model / (1 - model)) - offset
    return offset, np.column_stack([np.ones(len(d)), delta, delta * np.abs(delta)])


def fit_offset(d: pd.DataFrame, alpha: float):
    offset, X = ml_design(d)
    y = (d.actual_margin.to_numpy() > 0).astype(float)

    def objective(beta):
        eta = offset + X @ beta
        nll = np.logaddexp(0.0, eta).sum() - np.dot(y, eta)
        penalty = .1 * beta[0] ** 2 + np.dot(beta[1:], beta[1:])
        return nll + .5 * alpha * penalty

    return minimize(objective, np.zeros(X.shape[1]), method="BFGS").x


def predict_offset(d: pd.DataFrame, beta):
    offset, X = ml_design(d)
    eta = offset + X @ beta
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))


def residual_features(d: pd.DataFrame, market: str):
    if market == "spread":
        base = -d.spread.to_numpy(float)
        delta = d.model_margin.to_numpy(float) - base
    else:
        base = d.overUnder.to_numpy(float)
        delta = d.model_total.to_numpy(float) - base
    return base, np.column_stack([delta, delta * np.abs(delta), base])


def target_residual(d: pd.DataFrame, market: str):
    base, _ = residual_features(d, market)
    actual = d.actual_margin.to_numpy(float) if market == "spread" else d.actual_total.to_numpy(float)
    return actual - base


def choose_alpha(d: pd.DataFrame, test_year: int, market: str):
    prior = d[d.season < test_year]
    validation_years = [y for y in sorted(prior.season.unique()) if y > prior.season.min()]
    if not validation_years:
        return 10.0
    scores = {}
    for alpha in ALPHAS:
        fold_scores = []
        for year in validation_years:
            tr, va = prior[prior.season < year], prior[prior.season == year]
            if tr.empty or va.empty:
                continue
            if market == "moneyline":
                beta = fit_offset(tr, alpha)
                p = predict_offset(va, beta)
                y = (va.actual_margin > 0).astype(int)
                fold_scores.append(log_loss(y, p))
            else:
                _, X = residual_features(tr, market)
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(
                    X, target_residual(tr, market))
                _, XV = residual_features(va, market)
                fold_scores.append(mean_squared_error(target_residual(va, market), model.predict(XV)))
        scores[alpha] = float(np.mean(fold_scores)) if fold_scores else np.inf
    return min(scores, key=scores.get)


def expanding_predictions(d: pd.DataFrame, market: str) -> pd.DataFrame:
    rows = []
    for year in range(2023, 2026):
        needed = ["market_home_p", "model_home_p", "homeMoneyline", "awayMoneyline"] \
            if market == "moneyline" else (["spread", "model_margin"] if market == "spread"
                                             else ["overUnder", "model_total"])
        usable = d.dropna(subset=needed)
        tr, te = usable[usable.season < year], usable[usable.season == year].copy()
        if tr.empty or te.empty:
            continue
        alpha = choose_alpha(usable, year, market)
        if market == "moneyline":
            beta = fit_offset(tr, alpha)
            te["residual_prediction"] = predict_offset(te, beta)
        else:
            base, X = residual_features(tr, market)
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(
                X, target_residual(tr, market))
            base_te, Xte = residual_features(te, market)
            te["residual_prediction"] = base_te + model.predict(Xte)
        te["residual_alpha"] = alpha
        te["market"] = market
        rows.append(te)
    return pd.concat(rows, ignore_index=True)


def settle_ml(d: pd.DataFrame, probability_col: str, threshold: float):
    p = d[probability_col].to_numpy(float)
    hp = np.array([american_profit(x) for x in d.homeMoneyline], float)
    ap = np.array([american_profit(x) for x in d.awayMoneyline], float)
    ev_h, ev_a = p * hp - (1 - p), (1 - p) * ap - p
    home = ev_h >= ev_a
    ev = np.maximum(ev_h, ev_a)
    keep = ev >= threshold
    won = np.where(home, d.actual_margin.to_numpy() > 0, d.actual_margin.to_numpy() < 0)
    profit = np.where(won, np.where(home, hp, ap), -1.0)
    return profit[keep]


def settle_points(d: pd.DataFrame, col: str, threshold: float, market: str):
    if market == "spread":
        gap = d[col].to_numpy(float) + d.spread.to_numpy(float)
        result = np.where(gap >= 0, d.actual_margin + d.spread,
                          -d.actual_margin - d.spread)
    else:
        gap = d[col].to_numpy(float) - d.overUnder.to_numpy(float)
        result = np.where(gap >= 0, d.actual_total - d.overUnder,
                          d.overUnder - d.actual_total)
    keep = np.abs(gap) >= threshold
    return np.where(result[keep] > 0, 100/110,
                    np.where(result[keep] < 0, -1.0, 0.0))


def betting_summary(profit):
    p = np.asarray(profit, float)
    return {"bets": int(len(p)), "units": float(p.sum()),
            "roi": float(p.mean()) if len(p) else None,
            "hit_rate": float((p > 0).mean()) if len(p) else None}


def evaluate(pred: pd.DataFrame, market: str):
    if market == "moneyline":
        y = (pred.actual_margin > 0).astype(int)
        market_metric = {"brier": float(np.mean((pred.market_home_p - y) ** 2)),
                         "log_loss": float(log_loss(y, pred.market_home_p))}
        residual_metric = {"brier": float(np.mean((pred.residual_prediction - y) ** 2)),
                           "log_loss": float(log_loss(y, pred.residual_prediction))}
        grid = [0, .02, .05, .10, .15, .20]
        settle = lambda z, c, t: settle_ml(z, c, t)
        raw_col, residual_col, min_bets = "model_home_p", "residual_prediction", 100
    else:
        actual = pred.actual_margin if market == "spread" else pred.actual_total
        base = -pred.spread if market == "spread" else pred.overUnder
        market_metric = {"mae": float(mean_absolute_error(actual, base)),
                         "rmse": float(mean_squared_error(actual, base) ** .5)}
        residual_metric = {"mae": float(mean_absolute_error(actual, pred.residual_prediction)),
                           "rmse": float(mean_squared_error(actual, pred.residual_prediction) ** .5)}
        grid = [0, 1, 2, 3, 4, 5, 7]
        settle = lambda z, c, t: settle_points(z, c, t, market)
        raw_col = "model_margin" if market == "spread" else "model_total"
        residual_col, min_bets = "residual_prediction", 150

    dev, holdout = pred[pred.season <= 2024], pred[pred.season == 2025]
    curve = []
    for threshold in grid:
        curve.append({"threshold": threshold,
                      "raw_dev": betting_summary(settle(dev, raw_col, threshold)),
                      "residual_dev": betting_summary(settle(dev, residual_col, threshold)),
                      "raw_2025": betting_summary(settle(holdout, raw_col, threshold)),
                      "residual_2025": betting_summary(settle(holdout, residual_col, threshold))})
    eligible = [r for r in curve if r["residual_dev"]["bets"] >= min_bets]
    positive = [r for r in eligible if (r["residual_dev"]["roi"] or 0) > 0]
    selected = max(positive, key=lambda r: (r["residual_dev"]["roi"], r["residual_dev"]["bets"])) \
        if positive else (max(eligible, key=lambda r: r["residual_dev"]["bets"]) if eligible else None)
    return {"market_baseline": market_metric, "residual_model": residual_metric,
            "threshold_curve": curve, "selected_on_2023_24": selected,
            "validated_edge": bool(selected and (selected["residual_dev"]["roi"] or 0) > 0
                                   and (selected["residual_2025"]["roi"] or 0) > 0)}


def main():
    d = pd.read_csv(SOURCE)
    frames, report = [], {"method": "market baseline plus expanding-window regularized residual"}
    for market in ("moneyline", "spread", "total"):
        p = expanding_predictions(d, market)
        frames.append(p)
        report[market] = evaluate(p, market)
        m, r = report[market]["market_baseline"], report[market]["residual_model"]
        selected = report[market]["selected_on_2023_24"]
        print(f"{market}: market={m} residual={r}")
        print(f"  selected={selected}")
    pd.concat(frames, ignore_index=True).to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"-> {OUT_CSV}\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
