"""Does the standardisation of each feature, and the collinearity between them, cost
accuracy? Strict expanding replay, same protocol as scripts/v4_backtest.

Every shipped feature is z-scored within season regardless of its shape. Two things
are worth separating before changing anything:

  * The model fits on DIFFERENCES (home - away). A difference of two draws from the
    same skewed distribution is close to symmetric, so marginal skew is largely
    irrelevant here - portal_blue_in has marginal skew 2.12 and difference skew 0.13.
    What survives differencing is EXCESS KURTOSIS, i.e. high-leverage rows.
  * portal_in_rated, portal_out_rated and portal_net_rated are built as in, out and
    in-out, then z-scored separately. The third is 96.4% explained by the first two
    (VIF 27.9). That is a redundant column, not a feature.

Each variant changes one thing and is scored on pooled out-of-sample Brier over the
expanding replay, so a change has to earn its place the way every other feature
decision in this repo has.

Run: ./venv/bin/python -m scripts.standardisation_backtest
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from config import GAME_YEARS, ARTIFACTS
from src import v4 as V4
from scripts.train_v4 import build_inputs, build_frames

DEFAULTS = dict(C=.1, alpha=10.0, ensemble_weight=.5, probability_scale=1.0)
SHIPPED = json.load(open(ARTIFACTS.parent / "viz" / "data" / "model_v4.json"))["features"]
REC = ["rec_qb", "rec_ol", "rec_skill", "rec_front7", "rec_secondary"]


def rank_normal(s: pd.Series) -> pd.Series:
    """Blom rank-based inverse normal. Maps any continuous shape onto a Gaussian and
    is invariant to monotone transforms, so it costs nothing to try and bounds
    leverage by construction."""
    r = s.rank(method="average")
    n = s.notna().sum()
    return pd.Series(stats.norm.ppf((r - 0.375) / (n + 0.25)), index=s.index)


def winsor(s: pd.Series, k: float = 3.0) -> pd.Series:
    return s.clip(lower=s.mean() - k * s.std(ddof=0), upper=s.mean() + k * s.std(ddof=0))


def signed_log(s: pd.Series) -> pd.Series:
    """log1p on the count scale, re-standardised. The counts were already z-scored
    upstream, so undo that first: the z of a count is affine in the count."""
    lo = s.min()
    x = np.log1p(s - lo)
    return (x - x.mean()) / (x.std(ddof=0) or 1.0)


def variants(frames):
    """Each variant returns (feature_names, frames). Frames are copied per variant."""
    out = {}
    out["ship (z-score, as built)"] = (SHIPPED, frames)

    def transformed(name, cols, fn):
        fs = {y: f.copy() for y, f in frames.items()}
        for y, f in fs.items():
            for c in cols:
                if c in f:
                    f[c] = fn(f[c].astype(float))
        out[name] = (SHIPPED, fs)

    heavy = ["talent", "returning", "portal_blue_in", "portal_blue_out"]
    transformed("winsorise heavy tails at 3sd", heavy, lambda s: winsor(s, 3.0))
    transformed("rank-normal heavy tails", heavy, rank_normal)
    transformed("rank-normal everything", SHIPPED, rank_normal)
    transformed("log1p the two blue-chip counts", ["portal_blue_in", "portal_blue_out"], signed_log)

    # collinearity: drop the redundant third portal column
    out["drop portal_net_rated"] = ([f for f in SHIPPED if f != "portal_net_rated"], frames)
    # collinearity: collapse the recruiting block to its first principal component
    fs = {y: f.copy() for y, f in frames.items()}
    for y, f in fs.items():
        M = f[REC].astype(float).values
        M = (M - M.mean(0)) / (M.std(0, ddof=0) + 1e-12)
        u, s, vt = np.linalg.svd(M, full_matrices=False)
        pc = u[:, 0] * s[0]
        if np.corrcoef(pc, M.mean(1))[0, 1] < 0:
            pc = -pc
        f["rec_pc1"] = (pc - pc.mean()) / (pc.std(ddof=0) or 1.0)
    out["recruiting -> 1 PC"] = ([f for f in SHIPPED if f not in REC] + ["rec_pc1"], fs)

    # both collinearity fixes together
    fs2 = {y: f.copy() for y, f in fs.items()}
    out["rec -> 1 PC + drop portal_net"] = (
        [f for f in SHIPPED if f not in REC and f != "portal_net_rated"] + ["rec_pc1"], fs2)

    # everything: collinearity fixes plus bounded leverage
    fs3 = {y: f.copy() for y, f in fs.items()}
    for y, f in fs3.items():
        for c in ["talent", "returning", "portal_blue_in", "portal_blue_out"]:
            if c in f:
                f[c] = winsor(f[c].astype(float), 3.0)
    out["both + winsorise"] = (
        [f for f in SHIPPED if f not in REC and f != "portal_net_rated"] + ["rec_pc1"], fs3)
    return out


def replay(frames, games, names, knobs=DEFAULTS):
    """Strict expanding replay: train on all prior seasons, predict the next."""
    parts = {}
    for y in GAME_YEARS:
        if y in frames and y in games:
            parts[y] = V4.build_year(frames[y], games[y], names)
    pool = [y for y in GAME_YEARS if y in parts]
    sq, ll, acc, n = [], [], [], 0
    per_season = {}
    for i in range(1, len(pool)):
        test, train = pool[i], pool[:i]
        X = np.vstack([parts[t][0] for t in train])
        yv = np.concatenate([parts[t][1] for t in train])
        h = np.concatenate([parts[t][2] for t in train])
        m = np.concatenate([parts[t][3] for t in train])
        mdl = V4.fit(X, yv, h, m, names, **knobs)
        p, _ = V4.predict(mdl, parts[test])
        yt = parts[test][1]
        p = np.clip(p, 1e-8, 1 - 1e-8)
        per_season[test] = float(np.mean((p - yt) ** 2))
        sq.extend((p - yt) ** 2)
        ll.extend(-(yt * np.log(p) + (1 - yt) * np.log(1 - p)))
        acc.extend((p >= .5) == yt)
        n += len(yt)
    return dict(n=n, brier=float(np.mean(sq)), logloss=float(np.mean(ll)),
                accuracy=float(np.mean(acc)), per_season=per_season,
                sq=np.asarray(sq))


def main():
    std, talent, returning, games, od, pff_lag, war_lag, meta = build_inputs()
    years = sorted(set(list(GAME_YEARS)))
    frames = build_frames(std, talent, returning, od, pff_lag, war_lag, years)
    frames = {y: f for y, f in frames.items() if f is not None}

    results, base_sq = {}, None
    print(f"{'variant':<34}{'k':>4}{'brier':>10}{'vs ship':>10}{'logloss':>10}{'acc':>8}{'paired p':>10}")
    for name, (names_, fs) in variants(frames).items():
        r = replay(fs, games, names_)
        if base_sq is None:
            base_sq = r["sq"]
            delta, pval = 0.0, float("nan")
        else:
            d = r["sq"] - base_sq
            delta = float(d.mean())
            # paired t on per-game squared error; negative delta = variant is better
            pval = float(stats.ttest_1samp(d, 0.0).pvalue)
        results[name] = {k: v for k, v in r.items() if k != "sq"}
        results[name].update(delta_brier=delta, paired_p=pval, k=len(names_))
        print(f"{name:<34}{len(names_):>4}{r['brier']:>10.5f}"
              f"{delta:>+10.5f}{r['logloss']:>10.5f}{r['accuracy']:>8.4f}"
              f"{pval:>10.3f}")

    out = ARTIFACTS / "standardisation_backtest.json"
    json.dump(results, open(out, "w"), indent=1, default=float)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
