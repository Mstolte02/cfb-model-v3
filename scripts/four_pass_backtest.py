"""Strict test of quantitative Four-Pass-style overlays on locked v4 predictions.

The referenced framework publishes qualitative questions, not indices or weights.
This experiment turns the testable parts into reciprocal, pregame features:

* leader fragility: rolling giveback after halftime leads, lead-loss rate,
  second-half net decline, and final-margin volatility;
* trailer reversibility: rolling gain after halftime deficits, comeback-win rate,
  and second-half net gain;
* structural advantage: five offense-vs-defense edges already available in v4,
  summarized by magnitude, nonlinearity, breadth, and agreement with the base pick.

Every game uses only earlier game shapes.  For test season N, the overlay is fitted
only on locked v4 predictions from seasons before N.  The base logit/margin is an
offset, so the estimand is incremental value after the production model.

Run: python -m scripts.four_pass_backtest
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from config import ARTIFACTS, GAME_YEARS, ROOT
from scripts.train_v4 import build_frames, build_inputs
from scripts.v4_backtest import metric
from src import v4 as V4
from src.context import OffsetLogit


PRED_PATH = ARTIFACTS / "v4_backtest_predictions.csv"
OUT_JSON = ARTIFACTS / "four_pass_backtest.json"
OUT_CSV = ARTIFACTS / "four_pass_backtest_predictions.csv"
KEYS = ["season", "week", "home_team", "away_team"]
FAVORITE_THRESHOLD = .65
PENALTY = 20.0

FRAG_COMPONENTS = [
    "frag_giveback", "frag_lead_loss", "frag_late_decline", "frag_volatility",
]
REV_COMPONENTS = ["rev_comeback_gain", "rev_comeback_win", "rev_late_gain"]
STRUCTURAL_EDGES = ["struct_linear", "struct_nonlinear", "struct_breadth"]
STRUCTURAL_WEIGHT = ["base_x_struct_support", "base_x_struct_coherence"]


def _shape_summary(history: list[dict]) -> dict[str, float]:
    """Last-12-game shape with four pseudo-opportunities of zero effect.

    Shrinkage matters because a 1-for-1 comeback record is not a stable team trait.
    Fixed football scales in the composite indices avoid normalizing on future rows.
    """
    rows = history[-12:]
    empty = {"giveback": 0.0, "lead_loss": 0.0, "late_decline": 0.0,
             "volatility": 0.0, "comeback_gain": 0.0,
             "comeback_win": 0.0, "late_gain": 0.0}
    if not rows:
        return empty
    d = pd.DataFrame(rows)
    lead, trail = d[d.half_margin > 0], d[d.half_margin < 0]

    def shrunk_mean(values, n, pseudo=4.0):
        return float(values.mean()) * n / (n + pseudo) if n else 0.0

    return {
        "giveback": shrunk_mean(
            (lead.half_margin - lead.final_margin).clip(lower=0), len(lead)),
        "lead_loss": shrunk_mean((lead.final_margin < 0).astype(float), len(lead)),
        "late_decline": shrunk_mean(
            2 * lead.half_margin - lead.final_margin, len(lead)),
        "volatility": float(d.final_margin.std(ddof=0)) * len(d) / (len(d) + 4),
        "comeback_gain": shrunk_mean(
            (trail.final_margin - trail.half_margin).clip(lower=0), len(trail)),
        "comeback_win": shrunk_mean(
            (trail.final_margin > 0).astype(float), len(trail)),
        "late_gain": shrunk_mean(
            trail.final_margin - 2 * trail.half_margin, len(trail)),
    }


def build_shape_features() -> pd.DataFrame:
    """One pregame snapshot per completed game, before that result is recorded."""
    games = []
    for year in range(min(GAME_YEARS), max(GAME_YEARS) + 1):
        raw = json.loads((ROOT / "data" / "raw" / f"games_{year}.json").read_text())
        games.extend(g for g in raw if g.get("completed") and g.get("startDate")
                     and g.get("homeLineScores") and g.get("awayLineScores"))
    games.sort(key=lambda g: (g["startDate"], g.get("id", 0)))

    history: dict[str, list[dict]] = defaultdict(list)
    rows = []
    for g in games:
        home, away = g["homeTeam"], g["awayTeam"]
        hs, aws = _shape_summary(history[home]), _shape_summary(history[away])
        row = {"season": g["season"], "week": g["week"],
               "home_team": home, "away_team": away}
        row.update({f"home_{k}": v for k, v in hs.items()})
        row.update({f"away_{k}": v for k, v in aws.items()})
        rows.append(row)

        home_line = list(g.get("homeLineScores") or [])
        away_line = list(g.get("awayLineScores") or [])
        if len(home_line) < 4 or len(away_line) < 4:
            continue
        half = float(sum(home_line[:2]) - sum(away_line[:2]))
        final = float(g["homePoints"] - g["awayPoints"])
        history[home].append({"half_margin": half, "final_margin": final})
        history[away].append({"half_margin": -half, "final_margin": -final})
    return pd.DataFrame(rows).drop_duplicates(KEYS)


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def attach_role_features(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    logit = _logit(d.p_dynamic)
    side = np.where(logit >= 0, 1.0, -1.0)
    home_favorite = side > 0
    for source, name in (("giveback", "frag_giveback"),
                         ("lead_loss", "frag_lead_loss"),
                         ("late_decline", "frag_late_decline"),
                         ("volatility", "frag_volatility")):
        value = np.where(home_favorite, d[f"home_{source}"], d[f"away_{source}"])
        d[name] = side * value
    for source, name in (("comeback_gain", "rev_comeback_gain"),
                         ("comeback_win", "rev_comeback_win"),
                         ("late_gain", "rev_late_gain")):
        value = np.where(home_favorite, d[f"away_{source}"], d[f"home_{source}"])
        d[name] = side * value

    # Each unsigned index is on an interpretable, fixed scale. Multiplying by the
    # favorite orientation makes the feature negate when the matchup is reversed.
    frag = ((d.frag_giveback * side / 14.0) +
            (d.frag_lead_loss * side) +
            (d.frag_late_decline * side / 14.0) +
            (d.frag_volatility * side / 21.0)) / 4.0
    rev = ((d.rev_comeback_gain * side / 14.0) +
           (d.rev_comeback_win * side) +
           (d.rev_late_gain * side / 14.0)) / 3.0
    d["frag_index"], d["rev_index"] = side * frag, side * rev
    d["frag_x_rev"] = side * frag * rev
    strong = np.maximum(d.p_dynamic, 1 - d.p_dynamic) >= FAVORITE_THRESHOLD
    d["frag_index_strong"] = d.frag_index * strong
    d["rev_index_strong"] = d.rev_index * strong
    d["frag_x_rev_strong"] = d.frag_x_rev * strong
    return d


def attach_structural_features(data: pd.DataFrame) -> pd.DataFrame:
    std, talent, returning, games, od, pff, war, _ = build_inputs(
        include_projection=False)
    frames = build_frames(std, talent, returning, od, pff, war, GAME_YEARS)
    rows = []
    for r in data.itertuples():
        frame = frames[int(r.season)]
        home, away = frame.loc[r.home_team], frame.loc[r.away_team]
        edges = []
        for offense, defense in V4.MATCHUP_PAIRS.values():
            home_edge = float(home[offense]) - float(away[defense])
            away_edge = float(away[offense]) - float(home[defense])
            edges.append(home_edge - away_edge)
        edges = np.asarray(edges, float)
        base_logit = float(_logit([r.p_dynamic])[0])
        rows.append({
            "struct_linear": float(edges.mean()),
            "struct_nonlinear": float(np.mean(edges * np.abs(edges))),
            "struct_breadth": float(np.sign(edges).mean()),
            "base_x_struct_support": float(abs(base_logit) * edges.mean()),
            "base_x_struct_coherence": float(
                base_logit * abs(np.sign(edges).mean())),
        })
    return pd.concat([data.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def margin_overlay(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    X = train[features].to_numpy(float)
    Xt = test[features].to_numpy(float)
    scale = np.std(X, axis=0, ddof=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    model = Ridge(alpha=PENALTY, fit_intercept=False).fit(
        X / scale, train.margin - train.pred_margin)
    return test.pred_margin.to_numpy(float) + model.predict(Xt / scale)


def score_family(data: pd.DataFrame, name: str, features: list[str]):
    folds, pieces = [], []
    for test_year in sorted(data.season.unique())[1:]:
        train = data[data.season < test_year].dropna(subset=features)
        test = data[data.season == test_year].dropna(subset=features).copy()
        logit = OffsetLogit(features, penalty=PENALTY).fit(train)
        test["p_four_pass"] = logit.predict(test)
        test["margin_four_pass"] = margin_overlay(train, test, features)
        brier_change = (metric(test.y, test.p_four_pass)["brier"] -
                        metric(test.y, test.p_dynamic)["brier"])
        base_error = test.pred_margin - test.margin
        new_error = test.margin_four_pass - test.margin
        fold = {
            "season": int(test_year), "n": int(len(test)),
            "brier_change": float(brier_change),
            "margin_mae_change": float(new_error.abs().mean() - base_error.abs().mean()),
            "margin_rmse_change": float(
                np.sqrt(np.mean(new_error ** 2)) - np.sqrt(np.mean(base_error ** 2))),
            "logit_coef": logit.payload()["coef"],
        }
        folds.append(fold)
        test["family"] = name
        pieces.append(test)

    out = pd.concat(pieces, ignore_index=True)
    base_error = out.pred_margin - out.margin
    new_error = out.margin_four_pass - out.margin
    return {
        "name": name, "features": features, "n": int(len(out)),
        "pooled_brier_change": float(
            metric(out.y, out.p_four_pass)["brier"] -
            metric(out.y, out.p_dynamic)["brier"]),
        "pooled_margin_mae_change": float(
            new_error.abs().mean() - base_error.abs().mean()),
        "pooled_margin_rmse_change": float(
            np.sqrt(np.mean(new_error ** 2)) - np.sqrt(np.mean(base_error ** 2))),
        "folds": folds,
    }, out


def main():
    pred = pd.read_csv(PRED_PATH)
    data = pred.merge(build_shape_features(), on=KEYS, how="left",
                      validate="one_to_one")
    data = attach_role_features(data)
    data = attach_structural_features(data)
    families = {
        "fragility_components": FRAG_COMPONENTS,
        "reversibility_components": REV_COMPONENTS,
        "fragility_reversibility_indices": [
            "frag_index", "rev_index", "frag_x_rev"],
        "strong_favorite_indices": [
            "frag_index_strong", "rev_index_strong", "frag_x_rev_strong"],
        "structural_edges": STRUCTURAL_EDGES,
        "structural_context_weight": STRUCTURAL_WEIGHT,
        "all_four_pass": ["frag_index", "rev_index", "frag_x_rev",
                          *STRUCTURAL_EDGES, *STRUCTURAL_WEIGHT],
    }
    needed = sorted(set(sum(families.values(), [])))
    missing = int(data[needed].isna().any(axis=1).sum())
    if missing:
        raise RuntimeError(f"Four-Pass feature join missed {missing} modeled games")

    results, predictions = [], []
    for name, features in families.items():
        result, out = score_family(data, name, features)
        results.append(result)
        predictions.append(out[[*KEYS, "family", "y", "p_dynamic", "p_four_pass",
                                "margin", "pred_margin", "margin_four_pass"]])
        print(f"{name:<38} Brier {result['pooled_brier_change']:+.5f}  "
              f"MAE {result['pooled_margin_mae_change']:+.3f}  "
              f"RMSE {result['pooled_margin_rmse_change']:+.3f}")

    payload = {
        "contract": "season N overlay uses only locked v4 predictions and game shapes before N",
        "article_policy": "quantitative proxies devised here; article publishes no formula",
        "favorite_threshold": FAVORITE_THRESHOLD, "penalty": PENALTY,
        "families": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    pd.concat(predictions, ignore_index=True).to_csv(OUT_CSV, index=False)
    print(f"-> {OUT_JSON}\n-> {OUT_CSV}")


if __name__ == "__main__":
    main()
