"""Does the weekly team update do better if it carries a VARIANCE as well as a rating?

`src/dynamic.py` moves one number per team by a constant gain: every team, every
week, `K = .20`. That is Elo. The Bayesian version of the same object is Glickman &
Stern (1998): a state-space model where each team's rating is a distribution, the
filter's gain falls out of how much is still unknown, and the gain is therefore
different for each team and smaller in November than in September.

`audit/INSEASON_UPDATE_EXPERIMENTS.md` already found the shadow of this. Its moving-K
arm let the learning rate decay over the season, every fold chose decay, and the note
calls the result "directionally coherent with a Bayesian reading" - but it had to be
told to look for decay, and the two extra tuned parameters cost more than the .00048
Brier it bought. A filter produces the same decay without being asked.

THIS IS A PILOT, NOT THE PRODUCTION HARNESS. It reimplements both arms standalone in
margin space, from raw scores, with a weak carry-forward prior. It therefore does NOT
predict what the repo would gain - the shipping pipeline starts from the v4 preseason
model, a far better prior than this one, and a better prior leaves a filter less room.
What it can settle is narrower and still worth knowing: given the same prior and the
same strict week order, does a variance-tracked gain beat a constant one?

  Arm A, constant gain (the shipping shape):
      resid = y - (mu_h - mu_a + hfa)
      d     = Ka * clip(resid / sigma_e, -2.5, 2.5) * sigma_e
      mu_h += d ; mu_a -= d

  Arm B, Kalman / Glicko:
      s     = v_h + v_a + sigma_e^2
      resid = y - (mu_h - mu_a + hfa)
      mu_h += (v_h / s) * resid ; mu_a -= (v_a / s) * resid
      v_h  -= v_h^2 / s         ; v_a  -= v_a^2 / s
      v    += tau^2 once a week                       (process noise)

Margin space rather than the logit scale because the observation is then linear and
Gaussian, which makes the Kalman update exact rather than an approximation. The
shipping rule's Phi round trip is a reparameterisation, not the thing under test.

Both arms are tuned on 2022-23 and evaluated on 2024-25. BOTH ARE TUNED ON THE SAME
METRIC (MAE, which reads the point estimate only) - tuning them on different metrics
is how a pilot flatters its favourite, and doing that here moved the answer by a
factor of two. FCS is collapsed to one opponent, as Yurko & Benz (2025) do.

Run:  venv/Scripts/python -m scripts.kalman_rating_pilot
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ROOT

RAW = ROOT / "data" / "raw"
FCS = "__FCS__"
SEASONS = [2021, 2022, 2023, 2024, 2025]
TUNE = [2022, 2023]
TEST = [2024, 2025]
DRAWS = 4000
SEED = 20260918

# Shared nuisance parameters, held fixed so the comparison is only about the gain.
BASE = {"hfa": 2.4, "sigma_e": 16.5, "phi": .55, "fcs_mu": -26.0,
        "v0": 40.0, "v0_fcs": 5.0, "v_floor": 1.0, "tau": 1.2, "Ka": .18}
GRID_A = {"Ka": [.08, .10, .12, .15, .18, .21, .25, .30]}
GRID_B = {"v0": [15., 25., 40., 60., 90., 130., 180., 250., 400.],
          "tau": [.0, .8, 1.2, 1.8, 2.5]}


def load_season(year):
    fbs = {t["school"] for t in json.loads((RAW / f"teams_{year}.json").read_text())}
    out = []
    for g in json.loads((RAW / f"games_{year}.json").read_text()):
        if g.get("seasonType") != "regular" or not g.get("completed"):
            continue
        hp, ap = g.get("homePoints"), g.get("awayPoints")
        if hp is None or ap is None:
            continue
        h = g["homeTeam"] if g["homeTeam"] in fbs else FCS
        a = g["awayTeam"] if g["awayTeam"] in fbs else FCS
        if h == FCS and a == FCS:
            continue
        out.append({"week": int(g.get("week") or 0), "home": h, "away": a,
                    "neutral": bool(g.get("neutralSite")), "y": float(hp - ap),
                    # FCS results still move an FBS rating, but a game against the
                    # single collapsed FCS opponent is not scored: the model is not
                    # claiming to predict it.
                    "gradeable": h != FCS and a != FCS})
    out.sort(key=lambda r: r["week"])
    return fbs, out


DATA = {y: load_season(y) for y in SEASONS}


def run(arm, params, trace=False):
    """Filter every season in order, predicting a whole slate before applying any of
    its results. Between seasons the rating is pulled toward zero by phi and the
    variance is reset: a new roster is most of a new team."""
    hfa, sigma_e = params["hfa"], params["sigma_e"]
    rows, gains, carry = [], [], {}
    for year in SEASONS:
        fbs, games = DATA[year]
        teams = sorted(fbs) + [FCS]
        mu = {t: params["phi"] * carry.get(t, 0.0) for t in teams}
        v = {t: params["v0"] for t in teams}
        mu[FCS], v[FCS] = params["fcs_mu"], params["v0_fcs"]
        for week in sorted({g["week"] for g in games}):
            pend, wk_gain = [], []
            for g in (x for x in games if x["week"] == week):
                h, a = g["home"], g["away"]
                pred = mu[h] - mu[a] + (0.0 if g["neutral"] else hfa)
                if g["gradeable"]:
                    rows.append({"season": year, "week": week, "pred": pred,
                                 "pvar": v[h] + v[a] + sigma_e ** 2, "y": g["y"]})
                resid = g["y"] - pred
                if arm == "constant":
                    d = params["Ka"] * np.clip(resid / sigma_e, -2.5, 2.5) * sigma_e
                    pend.append((h, a, d, d, 0.0, 0.0))
                else:
                    s = v[h] + v[a] + sigma_e ** 2
                    if g["gradeable"]:
                        wk_gain.append(v[h] / s)
                    pend.append((h, a, (v[h] / s) * resid, (v[a] / s) * resid,
                                 v[h] ** 2 / s, v[a] ** 2 / s))
            for h, a, dh, da, vh, va in pend:
                mu[h] += dh
                mu[a] -= da
                v[h] = max(v[h] - vh, params["v_floor"])
                v[a] = max(v[a] - va, params["v_floor"])
            for t in teams:
                v[t] += params["tau"] ** 2
            if trace and wk_gain:
                gains.append({"season": year, "week": week,
                              "mean": float(np.mean(wk_gain)),
                              "lo": float(np.min(wk_gain)),
                              "hi": float(np.max(wk_gain)),
                              "rating_sd": float(np.sqrt(np.mean([v[t] for t in fbs])))})
        carry = {t: mu[t] for t in teams if t != FCS}
    return (rows, gains) if trace else rows


def score(rows, seasons, sigma):
    r = [x for x in rows if x["season"] in seasons]
    e = np.array([x["y"] - x["pred"] for x in r])
    p = norm.cdf(np.array([x["pred"] for x in r]) / sigma)
    win = np.array([1. if x["y"] > 0 else (0. if x["y"] < 0 else .5) for x in r])
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return {"n": len(r), "mae": float(np.abs(e).mean()),
            "rmse": float(np.sqrt((e ** 2).mean())),
            "brier": float(((p - win) ** 2).mean()),
            "logloss": float(-(win * np.log(pc) + (1 - win) * np.log(1 - pc)).mean())}


def bootstrap(ra, rb, seasons, sigma):
    """Paired season-week block bootstrap of the Brier difference, B minus A. Blocks
    are the slate, matching the rest of this repo: two games on one Saturday share a
    week's weather, injuries and market state."""
    a = [x for x in ra if x["season"] in seasons]
    b = [x for x in rb if x["season"] in seasons]
    win = np.array([1. if x["y"] > 0 else (0. if x["y"] < 0 else .5) for x in a])
    pa = norm.cdf(np.array([x["pred"] for x in a]) / sigma)
    pb = norm.cdf(np.array([x["pred"] for x in b]) / sigma)
    d = (pb - win) ** 2 - (pa - win) ** 2
    key = np.array([x["season"] * 100 + x["week"] for x in a])
    blocks = [d[key == k] for k in np.unique(key)]
    rng = np.random.default_rng(SEED)
    out = np.empty(DRAWS)
    for i in range(DRAWS):
        pick = rng.integers(0, len(blocks), len(blocks))
        out[i] = np.concatenate([blocks[j] for j in pick]).mean()
    return (float(d.mean()), float(np.percentile(out, 2.5)),
            float(np.percentile(out, 97.5)), float((out < 0).mean()))


def combos(grid):
    keys = list(grid)
    if len(keys) == 1:
        return [{keys[0]: x} for x in grid[keys[0]]]
    return [{keys[0]: x, keys[1]: y} for x in grid[keys[0]] for y in grid[keys[1]]]


def tune(arm, grid, sigma):
    rank = sorted((score(run(arm, {**BASE, **c}), TUNE, sigma)["mae"], i, c)
                  for i, c in enumerate(combos(grid)))
    return {**BASE, **rank[0][2]}, rank[0][2], rank[0][0]


def fmt(tag, s):
    return (f"{tag:<30} n={s['n']:<5} MAE {s['mae']:.3f}  RMSE {s['rmse']:.3f}  "
            f"Brier {s['brier']:.5f}  LL {s['logloss']:.5f}")


def main():
    sigma = BASE["sigma_e"]
    pa, ca, maea = tune("constant", GRID_A, sigma)
    pb, cb, maeb = tune("kalman", GRID_B, sigma)
    print(f"tuned on {TUNE} by MAE:  constant {ca} -> {maea:.3f}   "
          f"kalman {cb} -> {maeb:.3f}")

    ra = run("constant", pa)
    rb, gains = run("kalman", pb, trace=True)
    print(f"\nheld out {TEST}")
    print(" ", fmt("A constant gain", score(ra, TEST, sigma)))
    print(" ", fmt("B variance-tracked gain", score(rb, TEST, sigma)))
    d, lo, hi, p = bootstrap(ra, rb, TEST, sigma)
    print(f"   Brier B-A {d:+.5f}  95% [{lo:+.5f}, {hi:+.5f}]  P(B better) {p:.1%}")
    for y in TEST:
        print(" ", fmt(f"  A {y}", score(ra, [y], sigma)))
        print(" ", fmt(f"  B {y}", score(rb, [y], sigma)))

    # Does the filter's OWN variance make a better per-game sigma than a flat one?
    # c scales the epistemic part: c=0 is the flat sigma, c=1 is the raw filter.
    print("\nper-game sigma, epistemic variance scaled by c (fitted on 2022-23):")
    def ll(rows, seasons, c):
        r = [x for x in rows if x["season"] in seasons]
        sd = np.sqrt([c * (x["pvar"] - sigma ** 2) + sigma ** 2 for x in r])
        pr = np.clip(norm.cdf(np.array([x["pred"] for x in r]) / sd), 1e-6, 1 - 1e-6)
        w = np.array([1. if x["y"] > 0 else (0. if x["y"] < 0 else .5) for x in r])
        return float(-(w * np.log(pr) + (1 - w) * np.log(1 - pr)).mean()), \
            float(((pr - w) ** 2).mean())
    cs = [0., .1, .25, .4, .6, .8, 1., 1.3]
    best_c = min((ll(rb, TUNE, c)[0], c) for c in cs)[1]
    print(f"   fitted c = {best_c}")
    for c in sorted({0., best_c, 1.}):
        l, b = ll(rb, TEST, c)
        print(f"   c={c:<5} {TEST} Brier {b:.5f}  LL {l:.5f}")

    print("\nimplied Kalman gain by week (2025):")
    for g in [x for x in gains if x["season"] == 2025 and x["week"] <= 14]:
        print(f"   week {g['week']:>2}: gain {g['mean']:.3f}  "
              f"team spread {g['lo']:.3f}-{g['hi']:.3f}  "
              f"rating SD {g['rating_sd']:.2f} pts")


if __name__ == "__main__":
    main()
