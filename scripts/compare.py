"""Three-way held-out comparison of how we build each team's entering-season
strength vector. Everything downstream (features, L2 CV, isotonic calibration,
held-out season) is identical, so differences isolate the strength method.

  1. RAW          - prior-season stats only
  2. TALENT       - prior stats shrunk toward raw talent (lambda)
  3. CALIBRATED   - fitted projection from prior stat + talent + returning +
                    Pythagorean win expectation ("a more accurate mean")

Run: ./venv/bin/python -m scripts.compare
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import (GAME_YEARS, STAT_YEARS, TALENT_YEARS, RETURNING_YEARS,
                    PYTHAG_YEARS, TEST_GAME_YEAR, SHRINKAGE_LAMBDA)
from src.data import load
from src import strength as S
from src import projection as P
from src import model as M


def eval_method(name, strength_fn, games_by_year, train_years):
    parts = S.assemble(games_by_year, strength_fn, GAME_YEARS)
    Xtr = np.vstack([parts[g][0] for g in train_years])
    ytr = np.concatenate([parts[g][1] for g in train_years])
    hftr = np.concatenate([parts[g][2] for g in train_years])
    Xte, yte, hfte = parts[TEST_GAME_YEAR][:3]
    model, _ = M.train(Xtr, ytr, hftr)
    m = M.evaluate(model, Xte, yte, hfte)
    print(f"{name:<22} {model.C:>5} {m['accuracy']:>7.3f} "
          f"{m['brier']:>8.4f} {m['log_loss']:>9.4f}")
    return m


def main():
    load.require_key()
    print("Pulling stats, talent, returning production, games from CFBD ...")
    std_by_year, talent_by_year = S.load_seasons(load, STAT_YEARS, TALENT_YEARS)
    returning_by_year = {y: P._z(load.returning_production(y).set_index("team")["rp"])
                         for y in RETURNING_YEARS}
    games_by_year = {y: load.games(y) for y in set(GAME_YEARS) | set(PYTHAG_YEARS)}
    pythag_by_year = P.build_pythag(games_by_year)

    train_years = [g for g in GAME_YEARS if g != TEST_GAME_YEAR]
    talent_coeffs = S.fit_talent_coeffs(std_by_year, talent_by_year, train_years)
    proj_coeffs = P.fit_projection(train_years, std_by_year, talent_by_year,
                                   returning_by_year, pythag_by_year)

    def raw_fn(N):
        return std_by_year.get(N - 1)

    def talent_fn(N):
        if (N - 1) not in std_by_year:
            return None
        return S.entering_strength(N, std_by_year, talent_by_year,
                                   talent_coeffs, SHRINKAGE_LAMBDA, use_talent=True)

    def calib_fn(N):
        if (N - 1) not in std_by_year:
            return None
        return P.project(N, std_by_year, talent_by_year, returning_by_year,
                         pythag_by_year, proj_coeffs)

    print(f"\nHeld-out test season: {TEST_GAME_YEAR}\n")
    print(f"{'Method':<22} {'C':>5} {'Acc':>7} {'Brier':>8} {'LogLoss':>9}")
    print("-" * 54)
    raw = eval_method("RAW (prior stats)", raw_fn, games_by_year, train_years)
    tal = eval_method(f"TALENT (lam={SHRINKAGE_LAMBDA})", talent_fn,
                      games_by_year, train_years)
    cal = eval_method("CALIBRATED", calib_fn, games_by_year, train_years)

    print("\nvs RAW baseline:")
    for nm, m in [("TALENT", tal), ("CALIBRATED", cal)]:
        print(f"  {nm:<11} Brier {(m['brier']-raw['brier'])/raw['brier']*100:+.1f}%"
              f"   LogLoss {(m['log_loss']-raw['log_loss'])/raw['log_loss']*100:+.1f}%")

    print("\nProjection coefficients (target = stat_z(N)):")
    print(f"  {'feature':20} {'intcpt':>7} {'prior':>7} {'talent':>7} "
          f"{'return':>7} {'pythag':>7}")
    from src.features import FEATURE_COLS
    for f in FEATURE_COLS:
        b = proj_coeffs[f]
        print(f"  {f:20} {b[0]:>7.3f} {b[1]:>7.3f} {b[2]:>7.3f} {b[3]:>7.3f} {b[4]:>7.3f}")


if __name__ == "__main__":
    main()
