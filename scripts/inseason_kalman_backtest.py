"""Forward-only test of a variance-tracking (Kalman / Glicko) weekly update.

The shipping updater moves one scalar per team by a constant gain. The Bayesian
version of the same object carries a VARIANCE beside each rating, and the gain is
then not a parameter at all - it falls out of how much is still unknown:

    H     = d(predicted margin) / d(rating gap)     evaluated at this game
    S     = H^2 * (v_home + v_away) + sigma_e^2
    delta_home = (v_home * H / S) * (observed margin - expected margin)
    v_home    -= v_home^2 * H^2 / S                 seeing a game removes doubt
    v         += tau^2 once a week                  teams drift between games

This is the extended Kalman filter, which is what Glicko is: the local Gaussian
approximation on a nonlinear link. The link here is the model's own
margin <-> logit map, so H is computed from it rather than assumed.

`audit/INSEASON_UPDATE_EXPERIMENTS.md` already found the shadow of this. Its
moving-K arm let the learning rate decay over the season, every fold chose decay,
and the note calls that "directionally coherent with a Bayesian reading" - but the
decay had to be posited and given two tuned parameters, and it bought only .00048
Brier, under the project's .001 bar. A filter produces decay without being told to.

THREE ARMS, so the two things a filter does can be told apart:

  constant_k      the shipping rule. One gain for everybody, forever.
  kalman_shared   ONE pooled variance for all teams. The gain decays through the
                  season but is identical across teams on any given week. This is
                  the moving-K hypothesis, derived instead of parameterised.
  kalman          a variance per team. The gain now also differs between teams,
                  so a game moves the less-measured side further. This is the part
                  no constant or schedule-driven K can express.

If `kalman` beats `kalman_shared`, per-team uncertainty is doing real work. If they
tie, the whole gain is the decay and the simpler form should win on parsimony.

The score fed to the filter is the adopted robust margin residual, unchanged, so
this tests the GAIN and nothing else. Every parameter is selected on earlier
seasons only.

Run: python -m scripts.inseason_kalman_backtest
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.special import expit, ndtri
from scipy.stats import norm

from config import ARTIFACTS, GAME_YEARS
from scripts.inseason_evidence_backtest import (adjusted_margin_rows,
                                                public_pgwe_for_part,
                                                update_score)
from scripts.inseason_update_backtest import bootstrap_difference, build_frames
from scripts.train import load_bundle
from scripts.v4_backtest import (CANDIDATES, choose_candidate, fit_predict,
                                 metric, tune)
from src import v4 as V4
from src.data import load

OUT_JSON = ARTIFACTS / "inseason_kalman_backtest.json"
OUT_CSV = ARTIFACTS / "inseason_kalman_backtest_predictions.csv"

RULE = "raw_margin"          # the adopted score. Not under test here.
ARMS = ["constant_k", "kalman_shared", "kalman"]

K_GRID = [.15, .20, .25]
BLEND_GRID = [.75, 1.0]
# v0 is the preseason rating variance in LOGIT^2. Team ratings have SD ~1.04 logits,
# and v0 ~ .43 reproduces the shipping K = .20 on week 1, so the grid brackets that
# in both directions rather than centring on it.
V0_GRID = [.10, .20, .35, .55, .80, 1.20]
# tau is weekly drift in logits. At ~10.7 points per logit, tau = .10 is about a
# point of true week-to-week movement, which is the order the margin-space pilot
# selected. Zero is the nested no-drift filter.
TAU_GRID = [.0, .05, .10, .18, .28]
P_CLIP = .01


def link_slope(p, sigma):
    """d(margin)/d(logit gap) at this game's probability.

    The model reads a rating gap as a probability through a logistic and that
    probability as a margin through sigma * Phi^-1. The filter needs the derivative
    of the composition, because that is what converts a residual in POINTS into a
    correction in LOGITS. Taking it from the model's own link rather than assuming a
    constant is the whole reason this is an extended Kalman filter and not a
    relabelled Elo.
    """
    p = float(np.clip(p, P_CLIP, 1 - P_CLIP))
    return float(sigma * p * (1.0 - p) / norm.pdf(ndtri(p)))


def parameter_grid(arm):
    values = {"k": K_GRID if arm == "constant_k" else [0.0],
              "blend": BLEND_GRID,
              "v0": [0.0] if arm == "constant_k" else V0_GRID,
              "tau": [0.0] if arm == "constant_k" else TAU_GRID}
    keys = list(values)
    return [dict(zip(keys, item))
            for item in itertools.product(*(values[key] for key in keys))]


def default_parameters(arm):
    return {"k": .20 if arm == "constant_k" else 0.0, "blend": 1.0,
            "v0": 0.0 if arm == "constant_k" else .43,
            "tau": 0.0 if arm == "constant_k" else .10}


def replay(model, frame, part, arm, params, public_pgwe, return_trace=False):
    """Replay one season. A whole slate is predicted before any of its results are
    applied, so no game can inform its own prediction."""
    X, y, home_flag, margins, meta = part
    ratings = {team: model.team_logit_strength(frame, team) for team in frame.index}
    sigma = model.margin_sigma
    # kalman_shared keeps one number for the league; kalman keeps one per team. The
    # shared arm updates its single variance once per game as if every game informed
    # the same quantity, which is what makes it a pure decay schedule.
    variances = {team: params["v0"] for team in frame.index}
    shared = params["v0"]
    order = meta.assign(_row=np.arange(len(meta))).sort_values(["week", "_row"])
    pred = np.zeros(len(y))
    gains = np.zeros(len(y))

    for _, slate in order.groupby("week", sort=True, dropna=False):
        changes, var_cuts, shared_cut = {}, {}, 0.0
        for _, row in slate.iterrows():
            i, home, away = int(row._row), row.home_team, row.away_team
            gap = ratings[home] - ratings[away] + model.hfa_coef * home_flag[i]
            dynamic = float(expit(gap))
            static = model.win_prob(X[i], home_flag[i])
            pred[i] = (1.0 - params["blend"]) * static + params["blend"] * dynamic
            score = update_score(RULE, margins[i], margins[i], dynamic, sigma,
                                 {"sigma0": sigma, "beta": 0.0}, y[i],
                                 1.0, 0.0, public_pgwe[i])
            if arm == "constant_k":
                dh = da = params["k"] * score
            else:
                vh = shared if arm == "kalman_shared" else variances[home]
                va = shared if arm == "kalman_shared" else variances[away]
                H = link_slope(dynamic, sigma)
                s = H * H * (vh + va) + sigma * sigma
                # `score` is the residual in units of sigma and is already
                # winsorised at 2.5, so sigma * score is the residual in points
                # under the same robustness guard the adopted rule uses.
                dh = (vh * H / s) * sigma * score
                da = (va * H / s) * sigma * score
                cut_h, cut_a = vh * vh * H * H / s, va * va * H * H / s
                if arm == "kalman_shared":
                    shared_cut += cut_h
                else:
                    var_cuts[home] = var_cuts.get(home, 0.0) + cut_h
                    var_cuts[away] = var_cuts.get(away, 0.0) + cut_a
                gains[i] = vh * H / s * sigma
            changes[home] = changes.get(home, 0.0) + dh
            changes[away] = changes.get(away, 0.0) - da
        for team, change in changes.items():
            ratings[team] += change
        if arm == "kalman":
            for team, cut in var_cuts.items():
                variances[team] = max(variances[team] - cut, 1e-4)
            for team in variances:
                variances[team] += params["tau"] ** 2
        elif arm == "kalman_shared":
            # One variance for the league, so the slate's information is averaged
            # rather than summed - otherwise a big week would collapse it.
            n = max(1, len(slate))
            shared = max(shared - shared_cut / n, 1e-4) + params["tau"] ** 2
    if return_trace:
        return pred, gains
    return pred


def tune_arm(contexts, arm):
    if not contexts:
        return default_parameters(arm), {}
    scores = {}
    for params in parameter_grid(arm):
        losses = []
        for context in contexts:
            p = replay(context["model"], context["frame"], context["part"], arm,
                       params, context["public_pgwe"])
            losses.extend((p - context["part"][1]) ** 2)
        key = ",".join(f"{name}={value}" for name, value in params.items())
        scores[key] = float(np.mean(losses))
    winner = min(scores, key=scores.get)
    return ({item.split("=")[0]: float(item.split("=")[1])
             for item in winner.split(",")}, scores)


def make_context(model, frame, part, postgame_rows):
    public_pgwe, coverage = public_pgwe_for_part(postgame_rows, part[4], part[1])
    return {"model": model, "frame": frame, "part": part,
            "public_pgwe": public_pgwe, "coverage": coverage}


def main():
    std, talent, ret, games, _ = load_bundle()
    frames = build_frames(std, talent, ret, games)
    all_parts = {name: V4.assemble(GAME_YEARS, frames, games, columns)
                 for name, columns in CANDIDATES.items()}
    postgame = {year: adjusted_margin_rows(year, load.games(year))
                for year in [2020, *GAME_YEARS]}

    folds, outputs, traces = [], [], []
    for test in [2022, 2023, 2024, 2025]:
        pool = [year for year in GAME_YEARS if year < test]
        selected, _ = choose_candidate(all_parts, pool)
        names, parts = CANDIDATES[selected], all_parts[selected]
        knobs, _ = tune(parts, pool, names)
        validation = []
        for i in range(1, len(pool)):
            val, train = pool[i], pool[:i]
            val_model, _, _ = fit_predict(parts, train, val, names, knobs)
            validation.append(make_context(val_model, frames[val], parts[val],
                                           postgame[val]))
        model, _, _ = fit_predict(parts, pool, test, names, knobs)
        ctx = make_context(model, frames[test], parts[test], postgame[test])
        rows = parts[test][4].copy()
        rows["season"], rows["y"] = test, parts[test][1]
        fold = {"season": test, "selected_team_model": selected, "arms": {}}
        for arm in ARMS:
            params, trace = tune_arm(validation, arm)
            prediction, gain = replay(model, frames[test], parts[test], arm,
                                      params, ctx["public_pgwe"],
                                      return_trace=True)
            rows[arm] = prediction
            if arm == "kalman":
                rows["kalman_gain"] = gain
            fold["arms"][arm] = {"parameters": params,
                                 "metrics": metric(parts[test][1], prediction),
                                 "tuning": trace}
        folds.append(fold)
        outputs.append(rows)
        print(f"{test}: " + "  ".join(
            f"{a}={fold['arms'][a]['metrics']['brier']:.5f}" for a in ARMS),
            flush=True)

    predictions = pd.concat(outputs, ignore_index=True)
    primary = predictions[predictions.season >= 2023].copy()
    result = {
        "contract": "strict expanding replay; each held-out season uses parameters "
                    "selected on earlier seasons only",
        "score_rule": RULE,
        "primary_window": "2023-2025",
        "n_primary": int(len(primary)),
        "folds": folds,
        "pooled_2023_2025": {a: metric(primary.y, primary[a]) for a in ARMS},
        "bootstrap_2023_2025": {
            a: bootstrap_difference(primary, a, right="constant_k")
            for a in ARMS if a != "constant_k"},
    }
    # Does per-team variance add anything over a shared decaying one?
    result["bootstrap_2023_2025"]["kalman_vs_shared"] = bootstrap_difference(
        primary, "kalman", right="kalman_shared")

    OUT_JSON.write_text(json.dumps(result, indent=2))
    predictions.to_csv(OUT_CSV, index=False)
    print("\nPrimary 2023-25:")
    for name, item in result["pooled_2023_2025"].items():
        print(f"  {name:<16} Brier={item['brier']:.6f}  logloss={item['logloss']:.6f}")
    for name, item in result["bootstrap_2023_2025"].items():
        lo, hi = item["ci95"]
        print(f"  {name:<16} delta={item['brier_difference_vs_current']:+.6f}  "
              f"CI=[{lo:+.6f}, {hi:+.6f}]")
    live = predictions[(predictions.season == 2025) & (predictions.kalman_gain > 0)]
    if len(live):
        print("\n  implied gain by week, 2025 (equivalent K):")
        for week, g in live.groupby("week"):
            if int(week) <= 14:
                print(f"    week {int(week):>2}: mean {g.kalman_gain.mean():.3f}  "
                      f"range {g.kalman_gain.min():.3f}-{g.kalman_gain.max():.3f}")
    print(f"-> {OUT_JSON}\n-> {OUT_CSV}")


if __name__ == "__main__":
    main()
