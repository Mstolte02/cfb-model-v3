"""Leave-one-season-out (LOSO) cross-validation of the three strength methods.
A single held-out season is too noisy to rank methods that differ by ~0.2%
Brier; LOSO rotates the test season and averages, giving a stable read.

For each fold the talent slopes and projection coefficients are RE-FIT on that
fold's training seasons only (no leakage).

Run: ./venv/bin/python -m scripts.loso
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import (GAME_YEARS, STAT_YEARS, TALENT_YEARS, RETURNING_YEARS,
                    PYTHAG_YEARS, SHRINKAGE_LAMBDA)
from src.data import load
from src import strength as S
from src import projection as P
from src import model as M


def main():
    load.require_key()
    print("Pulling data from CFBD ...")
    std_by_year, talent_by_year = S.load_seasons(load, STAT_YEARS, TALENT_YEARS)
    returning_by_year = {y: P._z(load.returning_production(y).set_index("team")["rp"])
                         for y in RETURNING_YEARS}
    games_by_year = {y: load.games(y) for y in set(GAME_YEARS) | set(PYTHAG_YEARS)}
    pythag_by_year = P.build_pythag(games_by_year)

    # Folds: every game-year that has a prior stat year available.
    folds = [y for y in GAME_YEARS if (y - 1) in std_by_year]
    methods = ["RAW", "TALENT", "CALIBRATED"]
    agg = {m: {"brier": [], "log_loss": [], "accuracy": []} for m in methods}

    for test_year in folds:
        train_years = [g for g in GAME_YEARS if g != test_year]
        tcoef = S.fit_talent_coeffs(std_by_year, talent_by_year, train_years)
        pcoef = P.fit_projection(train_years, std_by_year, talent_by_year,
                                 returning_by_year, pythag_by_year)

        fns = {
            "RAW": lambda N: std_by_year.get(N - 1),
            "TALENT": lambda N: (None if (N - 1) not in std_by_year else
                                 S.entering_strength(N, std_by_year, talent_by_year,
                                                     tcoef, SHRINKAGE_LAMBDA, True)),
            "CALIBRATED": lambda N: (None if (N - 1) not in std_by_year else
                                     P.project(N, std_by_year, talent_by_year,
                                               returning_by_year, pythag_by_year, pcoef)),
        }
        for mth in methods:
            parts = S.assemble(games_by_year, fns[mth], GAME_YEARS)
            Xtr = np.vstack([parts[g][0] for g in train_years])
            ytr = np.concatenate([parts[g][1] for g in train_years])
            hftr = np.concatenate([parts[g][2] for g in train_years])
            model, _ = M.train(Xtr, ytr, hftr)
            m = M.evaluate(model, *parts[test_year])
            for k in agg[mth]:
                agg[mth][k].append(m[k])

    print(f"\nLOSO folds: {folds}\n")
    print(f"{'Method':<14} {'Brier':>8} {'LogLoss':>9} {'Acc':>7}   (mean +/- sd)")
    print("-" * 60)
    for mth in methods:
        b, l, a = agg[mth]["brier"], agg[mth]["log_loss"], agg[mth]["accuracy"]
        print(f"{mth:<14} {np.mean(b):>8.4f} {np.mean(l):>9.4f} {np.mean(a):>7.3f}"
              f"   (Brier sd {np.std(b):.4f})")
    base = np.mean(agg["RAW"]["brier"])
    print("\nMean Brier vs RAW:")
    for mth in ["TALENT", "CALIBRATED"]:
        d = (np.mean(agg[mth]["brier"]) - base) / base * 100
        print(f"  {mth:<12} {d:+.2f}%")
    print("\nPer-fold Brier:")
    print(f"  {'year':<6}" + "".join(f"{m:>12}" for m in methods))
    for i, y in enumerate(folds):
        print(f"  {y:<6}" + "".join(f"{agg[m]['brier'][i]:>12.4f}" for m in methods))


if __name__ == "__main__":
    main()
