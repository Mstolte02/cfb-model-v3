"""LOSO: does opponent (strength-of-schedule) adjustment of the O/D composites
help? Sweeps the adjustment strength alpha (0 = raw, the current model).
Run: ./venv/bin/python -m scripts.loso_oppadj
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import GAME_YEARS, STAT_YEARS, UNCERTAINTY_LAMBDA
from src.data import load
from src import matchup as MU
from src import oppadj as OA
from src import model as M
from scripts.train import load_bundle, raw_returning


def main():
    load.require_key()
    print("Pulling data ...")
    std, talent, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()
    folds = [y for y in GAME_YEARS if (y - 1) in std]

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    od_cache = {a: (None if a == 0 else OA.build_od_by_year(std, games, a)) for a in alphas}

    agg = {a: {"brier": [], "log_loss": [], "accuracy": []} for a in alphas}
    for test_year in folds:
        train_years = [g for g in GAME_YEARS if g != test_year]
        for a in alphas:
            od = od_cache[a]
            b_o, b_d = MU.fit_talent_od_slopes(train_years, std, talent, od_by_year=od)
            parts = MU.assemble(GAME_YEARS, std, pyth, talent, ret, games,
                                lam=UNCERTAINTY_LAMBDA, b_o=b_o, b_d=b_d,
                                ret_raw_by_year=ret_raw, od_by_year=od)
            Xtr = np.vstack([parts[g][0] for g in train_years if g in parts])
            ytr = np.concatenate([parts[g][1] for g in train_years if g in parts])
            hftr = np.concatenate([parts[g][2] for g in train_years if g in parts])
            mdl, _ = M.train(Xtr, ytr, hftr)
            m = M.evaluate(mdl, *parts[test_year])
            for k in agg[a]:
                agg[a][k].append(m[k])

    print(f"\nLOSO folds: {folds}\n")
    print(f"{'alpha':>6} {'Brier':>8} {'LogLoss':>9} {'Acc':>7}   (0 = raw / no SOS adj)")
    print("-" * 44)
    for a in alphas:
        b = agg[a]["brier"]; l = agg[a]["log_loss"]; ac = agg[a]["accuracy"]
        print(f"{a:>6.2f} {np.mean(b):>8.4f} {np.mean(l):>9.4f} {np.mean(ac):>7.3f}")
    raw = np.mean(agg[0.0]["brier"])
    best = min((a for a in alphas if a > 0), key=lambda a: np.mean(agg[a]["brier"]))
    print(f"\nBest alpha by Brier: {best}  "
          f"({(np.mean(agg[best]['brier'])-raw)/raw*100:+.2f}% vs raw)")


if __name__ == "__main__":
    main()
