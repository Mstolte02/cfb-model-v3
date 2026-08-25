"""Should a new feature improve the rating, or sit beside it?

The v4 model differences a team vector and feeds the whole thing to one logistic:

    logit = b1*(O_h - O_a) + b2*(D_h - D_a) + b3*(talent_h - talent_a)
          + b4*(ret_h - ret_a) + b5*(war_h - war_a)

So talent, returning production and roster WAR are **parallel terms next to** the O/D
power rating rather than inputs to it. Every experiment in this repo has added its
candidate the same way, which means none of them have ever asked whether the feature
belongs in the rating instead.

The alternative fits the rating first. For target season N, on training seasons only:

    O_N ~ prior O + talent + returning + WAR + positional recruiting + portal
    D_N ~ the same

and the fitted values become that team's entering rating. The win model then sees
**two columns**, the enriched offence and defence ratings, and nothing else. A feature
earns its place by making the rating better, which is also the only way it reaches the
published power rankings, the playoff simulator and the projected scores - all of
which read the rating, not the logistic's feature list.

Candidates:

* `outside_current`    - the shipping v4 feature list. The baseline.
* `outside_plus_new`   - shipping list plus positional recruiting and the portal, the
                         pattern this script exists to question.
* `inside_basic`       - rating fitted from talent, returning and WAR; two columns out.
* `inside_rich`        - rating fitted from those plus recruiting groups and portal.
* `inside_rich_cross`  - `inside_rich` plus an odd nonlinear offence-versus-defence
                         contrast, which is the one pairwise term the linear form
                         cannot express (see the MATCHUP_PAIRS note below).
* `inside_rich_outside`- rich rating AND the outside columns, to measure the
                         double-count directly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from config import ARTIFACTS, GAME_YEARS, OPP_ADJ_ALPHA
from scripts import v4_backtest as BT
from scripts.talent_sources_backtest import (GROUPS, PORTAL_RATED,
                                             group_features, portal_features)
from scripts.train import load_bundle
from src import oppadj as OA
from src import v4 as V4
from src.data import pff, war


OUT_JSON = ARTIFACTS / "rating_architecture_backtest.json"
BASIC = ["talent", "returning", "war_projected"]
RICH = [*BASIC, *GROUPS, *PORTAL_RATED]
RIDGE_ALPHA = 3.0

SPECS = {
    "outside_current": (None, ["O", "D", "talent", "returning", "war_projected"]),
    "outside_plus_new": (None, ["O", "D", "talent", "returning", "war_projected",
                                *GROUPS, *PORTAL_RATED]),
    "inside_basic": (BASIC, ["O", "D"]),
    "inside_rich": (RICH, ["O", "D"]),
    "inside_rich_cross": (RICH, ["O", "D", "od_cross"]),
    "inside_rich_outside": (RICH, ["O", "D", "talent", "returning",
                                   "war_projected", *GROUPS, *PORTAL_RATED]),
}


def fit_enriched_ratings(frames: dict, od: dict, predictors: list[str],
                         target_year: int) -> pd.DataFrame | None:
    """Rating for `target_year` fitted on completed seasons only.

    The target is the opponent-adjusted O/D the team actually posted in season N; the
    predictors are all preseason-available for N. Fitting on seasons strictly before
    the target keeps the same temporal contract as every other phase.
    """
    train_years = [y for y in frames if y < target_year and y in od]
    if not train_years or target_year not in frames:
        return None
    rows, targets_o, targets_d = [], [], []
    for year in train_years:
        frame = frames[year]
        realized = od.get(year)
        if realized is None:
            continue
        common = frame.index.intersection(realized.index)
        if not len(common):
            continue
        rows.append(frame.loc[common, predictors])
        targets_o.append(realized.loc[common, "O"])
        targets_d.append(realized.loc[common, "D"])
    if not rows:
        return None
    X = pd.concat(rows).to_numpy(float)
    model_o = Ridge(alpha=RIDGE_ALPHA).fit(X, pd.concat(targets_o).to_numpy(float))
    model_d = Ridge(alpha=RIDGE_ALPHA).fit(X, pd.concat(targets_d).to_numpy(float))
    current = frames[target_year]
    Z = current[predictors].to_numpy(float)
    out = pd.DataFrame({"O": model_o.predict(Z), "D": model_d.predict(Z)},
                       index=current.index)
    # Re-standardize so the enriched rating reaches the win model on the same scale
    # the plain O/D did, and the two architectures are not separated by units.
    for column in ("O", "D"):
        values = out[column]
        out[column] = (values - values.mean()) / (values.std(ddof=0) or 1.0)
    return out


# Register the headline offence-versus-defence contrast with the same machinery the
# granular matchup pairs use. `_row_matchup_vector` sees the name in MATCHUP_PAIRS and
# builds (O_h - D_a) against (O_a - D_h) with an odd nonlinear contrast, instead of
# differencing a column.
#
# Worth being precise about why this is the only cross term worth adding. The LINEAR
# version people expect - home offence against away defence - is already exactly
# representable: b*(O_h - D_a) + b*(D_h - O_a) expands to b*(O_h - O_a) + b*(D_h - D_a),
# which is what the model already fits. And it is antisymmetric only when both
# coefficients are equal, so the current separate-coefficient form is strictly more
# general, not less. What the linear form cannot express is a NONLINEAR mismatch: an
# elite offence against a poor defence being worth more than the sum of the parts.
V4.MATCHUP_PAIRS["od_cross"] = ("O", "D")


def build_frames():
    std, talent, ret, games, _ = load_bundle()
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    pff_lag = pff.build_lagged_team_talent()
    indices = {y: s.index for y, s in talent.items()}
    war_lag = war.lagged_team_talent(indices)
    war_projected = war.projected_team_talent(indices)
    portal = portal_features(GAME_YEARS)
    groups = group_features(GAME_YEARS)

    frames = {}
    for year in GAME_YEARS:
        frame = V4.build_frame(year, std, talent, ret, od, pff_lag, war_lag,
                               granular=True)
        if frame is None:
            continue
        wp = war_projected.get(year, pd.Series(dtype=float)).reindex(frame.index)
        frame["war_projected"] = wp.fillna(0.0)
        p = portal[year].reindex(frame.index)
        g = groups[year].reindex(frame.index)
        for column in PORTAL_RATED:
            values = p[column]
            frame[column] = ((values - values.mean()) /
                             (values.std(ddof=0) or 1.0)).fillna(0.0)
        for column in GROUPS:
            values = g[column]
            frame[column] = ((values - values.mean()) /
                             (values.std(ddof=0) or 1.0)).fillna(0.0)
        frames[year] = frame
    return frames, games, od


def _bootstrap(frame, left, right, draws=5000, seed=20260825):
    data = frame.dropna(subset=[left, right]).copy()
    data["delta"] = (data[left] - data.y) ** 2 - (data[right] - data.y) ** 2
    blocks = [g.delta.to_numpy(float) for _, g in
              data.groupby(["season", "week"], sort=True)]
    rng = np.random.default_rng(seed)
    samples = np.asarray([
        np.concatenate([blocks[i] for i in
                        rng.integers(0, len(blocks), len(blocks))]).mean()
        for _ in range(draws)])
    return {"difference": float(data.delta.mean()),
            "ci95": [float(v) for v in np.quantile(samples, [.025, .975])],
            "probability_left_better": float(np.mean(samples < 0))}


def main():
    base_frames, games, od = build_frames()
    years = sorted(base_frames)

    rows, rating_quality = [], {}
    for test in (2022, 2023, 2024, 2025):
        pool = [y for y in years if y < test]
        if not pool or test not in base_frames:
            continue
        # One enriched frame set per candidate: the rating has to be refit for every
        # fold, because it is fitted on completed seasons like everything else.
        variant_frames = {}
        for name, (predictors, _) in SPECS.items():
            if predictors is None:
                variant_frames[name] = base_frames
                continue
            built = {}
            for year in years:
                enriched = fit_enriched_ratings(base_frames, od, predictors, year)
                frame = base_frames[year].copy()
                if enriched is not None:
                    frame["O"] = enriched.O
                    frame["D"] = enriched.D
                built[year] = frame
            variant_frames[name] = built
            if test == 2025 and test in od:
                common = built[test].index.intersection(od[test].index)
                rating_quality[name] = {
                    "O_r": float(np.corrcoef(built[test].loc[common, "O"],
                                             od[test].loc[common, "O"])[0, 1]),
                    "D_r": float(np.corrcoef(built[test].loc[common, "D"],
                                             od[test].loc[common, "D"])[0, 1])}

        meta = None
        for name, (_, cols) in SPECS.items():
            frames = variant_frames[name]
            parts = V4.assemble(years, frames, games, cols)
            knobs, _ = BT.tune(parts, pool, cols)
            k, blend, _ = BT.tune_dynamic(parts, frames, pool, cols, knobs)
            model, static, margin = BT.fit_predict(parts, pool, test, cols, knobs)
            dynamic, _ = BT.dynamic_predictions(model, frames[test], parts[test],
                                                 k, blend)
            if meta is None:
                meta = parts[test][4].copy()
                meta["season"], meta["y"] = test, parts[test][1]
            meta[f"p_static_{name}"] = static
            meta[f"p_dynamic_{name}"] = dynamic
        rows.append(meta)
        print(f"{test} done", flush=True)

    predictions = pd.concat(rows, ignore_index=True)
    pooled = {n: {"static": BT.metric(predictions.y, predictions[f"p_static_{n}"]),
                  "dynamic": BT.metric(predictions.y, predictions[f"p_dynamic_{n}"])}
              for n in SPECS}
    vs_current = {n: {"static": _bootstrap(predictions, f"p_static_{n}",
                                            "p_static_outside_current"),
                      "dynamic": _bootstrap(predictions, f"p_dynamic_{n}",
                                             "p_dynamic_outside_current")}
                  for n in SPECS if n != "outside_current"}
    result = {
        "question": ("does a feature do more inside the O/D rating than beside it"),
        "specs": {k: {"rating_predictors": v[0], "model_columns": v[1]}
                  for k, v in SPECS.items()},
        "pooled": pooled, "vs_outside_current": vs_current,
        "rating_quality_2025": rating_quality,
        "selection_bar": BT.SELECTION_MIN_GAIN,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    bs = pooled["outside_current"]["static"]["brier"]
    bo = pooled["outside_current"]["dynamic"]["brier"]
    print(f"\n{'candidate':22s} {'cols':>5} {'static':>9} {'delta':>9} "
          f"{'online':>9} {'delta':>9}")
    for name, (_, cols) in SPECS.items():
        s = pooled[name]["static"]["brier"]
        o = pooled[name]["dynamic"]["brier"]
        print(f"{name:22s} {len(cols):>5} {s:>9.5f} {s-bs:>+9.5f} "
              f"{o:>9.5f} {o-bo:>+9.5f}")
    print("\nvs outside_current (negative = better):")
    for name, block in vs_current.items():
        st, dy = block["static"], block["dynamic"]
        print(f"  {name:22s} static {st['difference']:+.5f} "
              f"[{st['ci95'][0]:+.5f},{st['ci95'][1]:+.5f}]  online "
              f"{dy['difference']:+.5f} [{dy['ci95'][0]:+.5f},{dy['ci95'][1]:+.5f}]")
    if rating_quality:
        print("\n2025 rating vs realised opponent-adjusted O/D:")
        for name, q in rating_quality.items():
            print(f"  {name:22s} O r={q['O_r']:+.3f}  D r={q['D_r']:+.3f}")
    print(f"\n-> {OUT_JSON}")
    return result


if __name__ == "__main__":
    main()
