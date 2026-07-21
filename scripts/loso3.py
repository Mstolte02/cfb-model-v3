"""LOSO comparison: matchup model WITHOUT vs WITH the per-team uncertainty index
(doc §C). Uncertainty u = 1 - returning_production: low-continuity teams (new QB /
roster churn) get their prior-year O/D pulled toward their talent baseline.

Sweeps the global uncertainty strength `lam`. Coefficients/slopes re-fit per fold.
Run: ./venv/bin/python -m scripts.loso3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import GAME_YEARS, RETURNING_YEARS
from src.data import load
from src import matchup as MU
from src import model as M
from scripts.train import load_bundle


def _train_eval(parts, train_years, test_year):
    Xtr = np.vstack([parts[g][0] for g in train_years if g in parts])
    ytr = np.concatenate([parts[g][1] for g in train_years if g in parts])
    hftr = np.concatenate([parts[g][2] for g in train_years if g in parts])
    model, _ = M.train(Xtr, ytr, hftr)
    return M.evaluate(model, *parts[test_year])


def main():
    load.require_key()
    print("Pulling data from CFBD ...")
    std, talent, ret, games, pyth = load_bundle()
    # RAW returning fractions (not z-scored) for the uncertainty index.
    ret_raw = {y: load.returning_production(y).set_index("team")["rp"]
               for y in RETURNING_YEARS}

    folds = [y for y in GAME_YEARS if (y - 1) in std]
    lams = [0.0, 0.25, 0.5, 0.75, 1.0]
    agg = {l: {"brier": [], "log_loss": [], "accuracy": []} for l in lams}

    for test_year in folds:
        train_years = [g for g in GAME_YEARS if g != test_year]
        b_o, b_d = MU.fit_talent_od_slopes(train_years, std, talent)
        for lam in lams:
            parts = MU.assemble(GAME_YEARS, std, pyth, talent, ret, games,
                                lam=lam, b_o=b_o, b_d=b_d, ret_raw_by_year=ret_raw)
            m = _train_eval(parts, train_years, test_year)
            for k in agg[lam]:
                agg[lam][k].append(m[k])

    print(f"\nLOSO folds: {folds}\n")
    print(f"{'lambda':>7} {'Brier':>8} {'LogLoss':>9} {'Acc':>7}   (0 = no uncertainty)")
    print("-" * 50)
    for lam in lams:
        b = agg[lam]["brier"]; l = agg[lam]["log_loss"]; a = agg[lam]["accuracy"]
        print(f"{lam:>7.2f} {np.mean(b):>8.4f} {np.mean(l):>9.4f} {np.mean(a):>7.3f}")

    base = np.mean(agg[0.0]["brier"])
    best = min((l for l in lams if l > 0), key=lambda l: np.mean(agg[l]["brier"]))
    print(f"\nBest lambda by Brier: {best}")
    print(f"  Brier {base:.4f} -> {np.mean(agg[best]['brier']):.4f} "
          f"({(np.mean(agg[best]['brier'])-base)/base*100:+.2f}%)")
    print("\nPer-fold Brier (lam=0 vs best):")
    print(f"  {'year':<6}{'lam=0':>10}{f'lam={best}':>10}")
    for i, y in enumerate(folds):
        print(f"  {y:<6}{agg[0.0]['brier'][i]:>10.4f}{agg[best]['brier'][i]:>10.4f}")


if __name__ == "__main__":
    main()
