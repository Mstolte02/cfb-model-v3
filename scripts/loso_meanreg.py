"""Mean-regression formula experiments (doc §C). How should we distribute the
shrinkage of prior O/D toward the talent baseline across teams?

Each formula defines a per-team uncertainty u in [0,1]; all formulas are scaled to
the SAME average shrinkage (mean u = 0.3, lam=1), so the comparison isolates the
*shape* of the regression, not its overall strength:

  none      - no shrinkage (pure prior performance)
  global    - everyone shrinks equally
  returning - shrink ~ (1 - returning production)        [current default idea]
  variance  - shrink ~ within-season inconsistency (Bayesian: noisy -> regress)
  rating    - shrink ~ |prior rating|  (extremes regress more; nonlinear)
  combo     - returning + variance

Run: ./venv/bin/python -m scripts.loso_meanreg
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import GAME_YEARS, STAT_YEARS, OPP_ADJ_ALPHA
from src.data import load, pff
from src import matchup as MU, oppadj as OA, model as M, ewma as E
from scripts.train import load_bundle, raw_returning, blended_talent

TARGET_MEAN = 0.3


def _scale(u):
    u = u.clip(lower=0).astype(float)
    m = u.mean()
    return (u * (TARGET_MEAN / m)).clip(0, 1) if m > 0 else u


def within_season_std():
    """{season: Series(team -> std of game-level net rating)} (inconsistency)."""
    out = {}
    for y in STAT_YEARS:
        try:
            g = E._game_composites(y)
        except Exception:
            continue
        net = (g["O_game"] - g["D_game"])
        out[y] = g.assign(net=net).groupby("team")["net"].std()
    return out


def main():
    load.require_key()
    print("Loading bundle + signals ...")
    std, cfbd_tal, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()
    talent = blended_talent(cfbd_tal, pff.build_roster_talent())
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    wstd = within_season_std()
    folds = [y for y in GAME_YEARS if (y - 1) in std]

    # Build per-entering-year u for each formula (uses N-1 info).
    def u_for(formula, N):
        prior = N - 1
        teams = od[prior].index if prior in od else std[prior].index
        if formula == "global":
            return pd.Series(TARGET_MEAN, index=teams)
        if formula == "returning" and N in ret_raw:
            return _scale(1.0 - ret_raw[N].reindex(teams))
        if formula == "variance" and prior in wstd:
            return _scale(wstd[prior].reindex(teams))
        if formula == "rating" and prior in od:
            ov = od[prior]["O"] + od[prior]["D"]
            return _scale((ov - ov.mean()).abs())
        if formula == "combo" and N in ret_raw and prior in wstd:
            a = _scale(1.0 - ret_raw[N].reindex(teams))
            b = _scale(wstd[prior].reindex(teams))
            return _scale((a.fillna(a.mean()) + b.fillna(b.mean())) / 2)
        return pd.Series(TARGET_MEAN, index=teams)

    def run(formula):
        lam = 0.0 if formula == "none" else 1.0
        u_by_year = None if formula == "none" else {N: u_for(formula, N) for N in GAME_YEARS}
        briers = []
        for ty in folds:
            tr = [g for g in GAME_YEARS if g != ty]
            b_o, b_d = MU.fit_talent_od_slopes(tr, std, talent, od_by_year=od)
            parts = MU.assemble(GAME_YEARS, std, pyth, talent, ret, games,
                                lam=lam, b_o=b_o, b_d=b_d, ret_raw_by_year=ret_raw,
                                od_by_year=od, u_by_year=u_by_year)
            Xtr = np.vstack([parts[g][0] for g in tr if g in parts])
            ytr = np.concatenate([parts[g][1] for g in tr if g in parts])
            hf = np.concatenate([parts[g][2] for g in tr if g in parts])
            mdl, _ = M.train(Xtr, ytr, hf)
            briers.append(M.evaluate(mdl, *parts[ty])["brier"])
        return float(np.mean(briers))

    print(f"\nMean-regression formulas (mean u={TARGET_MEAN}, LOSO folds {folds}):\n")
    print(f"{'formula':<12}{'Brier':>9}")
    print("-" * 21)
    for f in ["none", "global", "returning", "variance", "rating", "combo"]:
        print(f"{f:<12}{run(f):>9.4f}")


if __name__ == "__main__":
    main()
