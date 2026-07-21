"""LOSO: does weighting recent games (EWMA) beat the flat season average for
building prior-season O/D ratings? Same pipeline (talent + returning + Pythagorean
+ uncertainty + win logistic); only the O/D construction changes.

All variants use game-level stats (so the only difference is the recency weight):
half-life in games; inf = flat equal weight. Run: ./venv/bin/python -m scripts.loso_ewma
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import GAME_YEARS, STAT_YEARS, UNCERTAINTY_LAMBDA
from src.data import load
from src import matchup as MU
from src import model as M
from src import ewma as E
from scripts.train import load_bundle, raw_returning


def main():
    load.require_key()
    print("Pulling data + building game-level EWMA ratings (this pulls game stats) ...")
    std, talent, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()
    folds = [y for y in GAME_YEARS if (y - 1) in STAT_YEARS]

    halflives = [float("inf"), 8.0, 5.0, 3.0, 2.0]   # inf = flat average
    od_cache = {hl: E.build_od_by_year(STAT_YEARS, hl) for hl in halflives}

    agg = {hl: {"brier": [], "log_loss": [], "accuracy": []} for hl in halflives}
    for test_year in folds:
        train_years = [g for g in GAME_YEARS if g != test_year]
        for hl in halflives:
            od = od_cache[hl]
            b_o, b_d = MU.fit_talent_od_slopes(train_years, std, talent, od_by_year=od)
            parts = MU.assemble(GAME_YEARS, std, pyth, talent, ret, games,
                                lam=UNCERTAINTY_LAMBDA, b_o=b_o, b_d=b_d,
                                ret_raw_by_year=ret_raw, od_by_year=od)
            Xtr = np.vstack([parts[g][0] for g in train_years if g in parts])
            ytr = np.concatenate([parts[g][1] for g in train_years if g in parts])
            hftr = np.concatenate([parts[g][2] for g in train_years if g in parts])
            mdl, _ = M.train(Xtr, ytr, hftr)
            m = M.evaluate(mdl, *parts[test_year])
            for k in agg[hl]:
                agg[hl][k].append(m[k])

    print(f"\nLOSO folds: {folds}  (all variants use game-level stats)\n")
    print(f"{'half-life':>10} {'Brier':>8} {'LogLoss':>9} {'Acc':>7}")
    print("-" * 40)
    for hl in halflives:
        label = "flat" if not np.isfinite(hl) else f"{hl:g} games"
        b = agg[hl]["brier"]; l = agg[hl]["log_loss"]; a = agg[hl]["accuracy"]
        print(f"{label:>10} {np.mean(b):>8.4f} {np.mean(l):>9.4f} {np.mean(a):>7.3f}")

    flat = np.mean(agg[float('inf')]["brier"])
    best = min((h for h in halflives if np.isfinite(h)),
               key=lambda h: np.mean(agg[h]["brier"]))
    print(f"\nBest EWMA half-life by Brier: {best:g} games")
    print(f"  Brier {flat:.4f} (flat) -> {np.mean(agg[best]['brier']):.4f} "
          f"({(np.mean(agg[best]['brier'])-flat)/flat*100:+.2f}%)")


if __name__ == "__main__":
    main()
