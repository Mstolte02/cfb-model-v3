"""LOSO comparison: current per-stat CALIBRATED model vs the new true
MATCHUP-adjusted (offense-vs-defense) model. Both optimize game-outcome Brier
(target B); coefficients re-fit per fold. Run: ./venv/bin/python -m scripts.loso2
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import GAME_YEARS, TEST_GAME_YEAR
from src.data import load
from src import strength as S
from src import projection as P
from src import matchup as MU
from src import model as M
from scripts.train import load_bundle


def main():
    load.require_key()
    print("Pulling data from CFBD ...")
    std, talent, ret, games, pyth = load_bundle()  # all dicts already standardized
    folds = [y for y in GAME_YEARS if (y - 1) in std]
    agg = {"CALIBRATED": {"brier": [], "log_loss": [], "accuracy": []},
           "MATCHUP":    {"brier": [], "log_loss": [], "accuracy": []}}

    for test_year in folds:
        train_years = [g for g in GAME_YEARS if g != test_year]

        # --- per-stat calibrated (reference) ---
        pcoef = P.fit_projection(train_years, std, talent, ret, pyth)
        calib_fn = lambda N: (None if (N - 1) not in std else
                              P.project(N, std, talent, ret, pyth, pcoef))
        cparts = S.assemble(games, calib_fn, GAME_YEARS)
        m = _train_eval(cparts, train_years, test_year)
        for k in agg["CALIBRATED"]:
            agg["CALIBRATED"][k].append(m[k])

        # --- true matchup model ---
        mparts = MU.assemble(GAME_YEARS, std, pyth, talent, ret, games)
        m = _train_eval(mparts, train_years, test_year)
        for k in agg["MATCHUP"]:
            agg["MATCHUP"][k].append(m[k])

    print(f"\nLOSO folds: {folds}\n")
    print(f"{'Model':<14} {'Brier':>8} {'LogLoss':>9} {'Acc':>7}")
    print("-" * 42)
    for name in ["CALIBRATED", "MATCHUP"]:
        b = agg[name]["brier"]; l = agg[name]["log_loss"]; a = agg[name]["accuracy"]
        print(f"{name:<14} {np.mean(b):>8.4f} {np.mean(l):>9.4f} {np.mean(a):>7.3f}")
    cb, mb = np.mean(agg["CALIBRATED"]["brier"]), np.mean(agg["MATCHUP"]["brier"])
    print(f"\nMATCHUP vs CALIBRATED Brier: {(mb-cb)/cb*100:+.2f}%")
    print("\nPer-fold Brier:")
    print(f"  {'year':<6}{'CALIBRATED':>12}{'MATCHUP':>12}")
    for i, y in enumerate(folds):
        print(f"  {y:<6}{agg['CALIBRATED']['brier'][i]:>12.4f}"
              f"{agg['MATCHUP']['brier'][i]:>12.4f}")


def _train_eval(parts, train_years, test_year):
    Xtr = np.vstack([parts[g][0] for g in train_years if g in parts])
    ytr = np.concatenate([parts[g][1] for g in train_years if g in parts])
    hftr = np.concatenate([parts[g][2] for g in train_years if g in parts])
    model, _ = M.train(Xtr, ytr, hftr)
    return M.evaluate(model, *parts[test_year])


if __name__ == "__main__":
    main()
