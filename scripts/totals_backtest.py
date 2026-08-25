"""Can the totals model be repaired?

`MODEL_VS_MARKET_DIAGNOSIS.md` shows the published model total correlates .099 with
actual points over 2022-25 and .002 in 2025 - no measured information in the most
recent season - while the market's total correlates .379. This tests whether the two
structural omissions named in `src/totals.py` account for it.

Candidates add one thing at a time so the answer says *which* omission mattered:

* `current` reproduces the shipping three-feature points model exactly.
* `+level` adds the scorer's prior points-per-game and the opponent's prior
  points-allowed-per-game.
* `+pace` adds prior offensive plays run and defensive plays faced.
* `+both` adds all four.
* `+both_interact` adds level x pace, since points are closer to a product of
  efficiency and possessions than a sum.

Everything is fit inside expanding folds: season N trains on completed seasons < N
and the lagged inputs come from N-1. The market total is reported on the same games
as a reference line, not as a competitor to beat - matching it is not the bar, having
any real information is.

Adoption rule, fixed before running: a candidate replaces the current totals model
only if it improves RMSE with a 95% blocked bootstrap interval excluding zero AND
raises correlation with actual points materially above the current .099.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import ARTIFACTS, GAME_YEARS, OPP_ADJ_ALPHA
from scripts import betting_backtest as BB
from scripts.train import load_bundle
from src import oppadj as OA
from src import totals as TT
from src import v4 as V4
from src.data import plays as PLAYS
from sklearn.linear_model import Ridge


OUT_JSON = ARTIFACTS / "totals_backtest.json"
OUT_CSV = ARTIFACTS / "totals_backtest_predictions.csv"
DRAWS = 4000
SEED = 20260825

SPECS = {
    "current": ["O_scorer", "D_opponent", "home"],
    "level": ["O_scorer", "D_opponent", "home", "pf_scorer", "pa_opponent"],
    "pace": ["O_scorer", "D_opponent", "home", "offp_scorer", "defp_opponent"],
    "both": ["O_scorer", "D_opponent", "home", "pf_scorer", "pa_opponent",
             "offp_scorer", "defp_opponent"],
    "both_interact": ["O_scorer", "D_opponent", "home", "pf_scorer", "pa_opponent",
                      "offp_scorer", "defp_opponent", "pf_x_pace", "pa_x_pace"],
}


def _bootstrap_rmse(frame: pd.DataFrame, left: str, right: str) -> dict:
    """Blocked bootstrap on the squared-error difference; blocks are season-week."""
    data = frame.dropna(subset=[left, right]).copy()
    data["delta"] = ((data[left] - data.actual_total) ** 2 -
                     (data[right] - data.actual_total) ** 2)
    blocks = [g.delta.to_numpy(float) for _, g in
              data.groupby(["season", "week"], sort=True)]
    rng = np.random.default_rng(SEED)
    draws = np.asarray([
        np.concatenate([blocks[i] for i in
                        rng.integers(0, len(blocks), len(blocks))]).mean()
        for _ in range(DRAWS)])
    return {"mean_squared_error_difference": float(data.delta.mean()),
            "ci95": [float(np.quantile(draws, .025)),
                     float(np.quantile(draws, .975))],
            "probability_left_better": float(np.mean(draws < 0))}


def main():
    std, talent, ret, games, _ = load_bundle()
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    play_frame = PLAYS.load()
    profiles = TT.team_season_profiles(plays_frame=play_frame)
    lagged = TT.lagged_profiles(GAME_YEARS, profiles)

    frames = {}
    for year in GAME_YEARS:
        frame = V4.build_frame(year, std, talent, ret, od)
        if frame is None:
            continue
        frames[year] = TT.attach(frame, lagged[year])

    rows = []
    for test in (2022, 2023, 2024, 2025):
        pool = [y for y in GAME_YEARS if y < test and y in frames]
        if not pool or test not in frames:
            continue
        train_X = pd.concat([TT.build_points_rows(frames[y], games[y])[0] for y in pool],
                            ignore_index=True)
        train_y = np.concatenate([TT.build_points_rows(frames[y], games[y])[1] for y in pool])
        test_X, _, meta = TT.build_points_rows(frames[test], games[test])
        meta["season"] = test
        for name, columns in SPECS.items():
            model = Ridge(alpha=3.0).fit(train_X[columns].to_numpy(float), train_y)
            side = model.predict(test_X[columns].to_numpy(float))
            # Two rows per game: the total is their sum.
            meta[f"total_{name}"] = side[0::2] + side[1::2]
        rows.append(meta)

    predictions = pd.concat(rows, ignore_index=True)

    # Market reference on the same games.
    market = BB.game_market_rows(pd.read_csv(BB.PREDICTIONS))[
        ["season", "week", "home", "away", "overUnder", "model_total"]]
    predictions = predictions.merge(
        market.rename(columns={"home": "home_team", "away": "away_team"}),
        on=["season", "week", "home_team", "away_team"], how="left")

    def score(column, mask=None):
        data = predictions.dropna(subset=[column, "actual_total"])
        if mask is not None:
            data = data[mask.reindex(data.index).fillna(False)]
        if not len(data):
            return None
        error = data[column] - data.actual_total
        return {"n": int(len(data)),
                "rmse": float(np.sqrt((error ** 2).mean())),
                "mae": float(error.abs().mean()),
                "bias": float(error.mean()),
                "corr": float(data[column].corr(data.actual_total)),
                "prediction_sd": float(data[column].std())}

    scored = {name: score(f"total_{name}") for name in SPECS}
    scored["published_model_total"] = score("model_total")
    scored["market"] = score("overUnder")
    by_season = {name: {int(s): score(f"total_{name}", predictions.season.eq(s))
                        for s in sorted(predictions.season.unique())}
                 for name in SPECS}
    by_season["market"] = {int(s): score("overUnder", predictions.season.eq(s))
                           for s in sorted(predictions.season.unique())}

    comparisons = {name: _bootstrap_rmse(predictions, f"total_{name}",
                                          "total_current")
                   for name in SPECS if name != "current"}

    result = {
        "contract": ("season N trains on completed seasons < N; level and pace "
                     "inputs come from season N-1"),
        "adoption_rule": ("replace only on an RMSE improvement whose 95% interval "
                          "excludes zero, with correlation materially above .099"),
        "specs": SPECS, "pooled": scored, "by_season": by_season,
        "vs_current": comparisons,
        "actual_total_sd": float(predictions.actual_total.std()),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    predictions.to_csv(OUT_CSV, index=False)

    print(f"{'candidate':26s} {'n':>6} {'RMSE':>8} {'corr':>7} {'pred sd':>8} "
          f"{'bias':>7}")
    for name, block in scored.items():
        if block is None:
            continue
        print(f"{name:26s} {block['n']:>6} {block['rmse']:>8.3f} "
              f"{block['corr']:>7.3f} {block['prediction_sd']:>8.2f} "
              f"{block['bias']:>+7.2f}")
    print(f"\nactual total SD: {result['actual_total_sd']:.2f}")
    print("\n--- vs current (negative = better) ---")
    for name, block in comparisons.items():
        print(f"{name:26s} {block['mean_squared_error_difference']:>+10.3f} "
              f"[{block['ci95'][0]:+.3f}, {block['ci95'][1]:+.3f}]  "
              f"P(better)={block['probability_left_better']:.3f}")
    print("\n--- correlation by season ---")
    for name in [*SPECS, "market"]:
        cells = " ".join(f"{s}:{(by_season[name][s] or {}).get('corr', float('nan')):+.3f}"
                         for s in sorted(by_season[name]))
        print(f"{name:26s} {cells}")
    print(f"\n-> {OUT_JSON}\n-> {OUT_CSV}")
    return result


if __name__ == "__main__":
    main()
