"""Which feature set should the model actually run on?

feature_audit.py established the shape of the problem: pythag correlates 0.61/0.63
with the O and D composites and carries the highest VIF, while fp_margin costs
accuracy. This scores the candidate sets head to head under ONE harness that refits
the talent->O/D slopes and the model per fold, so the numbers are comparable to the
0.2040 that ships.

Also answers the separate question of whether WAR can stand on its own: a WAR-only
talent vector would let the lens toggle become "roster-weighted" vs "WAR" instead of
two variations of the same blend.

Run: ./venv/bin/python -m scripts.feature_sets
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import GAME_YEARS, UNCERTAINTY_LAMBDA, OPP_ADJ_ALPHA
from src.data import load, pff, war
from src import matchup as MU
from src import oppadj as OA
from src import model as M
from scripts.train import load_bundle, raw_returning, blended_talent

FEATURES = ["O", "D", "fp_margin", "pythag", "talent", "returning"]

CANDIDATES = {
    "all six (ships today)":  FEATURES,
    "drop fp_margin":         ["O", "D", "pythag", "talent", "returning"],
    "drop pythag":            ["O", "D", "fp_margin", "talent", "returning"],
    "drop both":              ["O", "D", "talent", "returning"],
    "O/D + talent":           ["O", "D", "talent"],
    "O/D only":               ["O", "D"],
    "talent + returning":     ["talent", "returning"],
    "talent only":            ["talent"],
}


def run(cols, std, ret, games, pyth, ret_raw, od, talent):
    """LOSO with a feature subset, refitting slopes and model per fold."""
    idx = [FEATURES.index(c) for c in cols]
    out = {"brier": [], "log_loss": [], "accuracy": []}
    for ty in GAME_YEARS:
        tr = [g for g in GAME_YEARS if g != ty]
        b_o, b_d = MU.fit_talent_od_slopes(tr, std, talent, od_by_year=od)
        parts = MU.assemble(GAME_YEARS, std, pyth, talent, ret, games,
                            lam=UNCERTAINTY_LAMBDA, b_o=b_o, b_d=b_d,
                            ret_raw_by_year=ret_raw, od_by_year=od)
        if ty not in parts:
            continue
        Xtr = np.vstack([parts[g][0][:, idx] for g in tr if g in parts])
        ytr = np.concatenate([parts[g][1] for g in tr if g in parts])
        hf = np.concatenate([parts[g][2] for g in tr if g in parts])
        mdl, _ = M.train(Xtr, ytr, hf)
        m = M.evaluate(mdl, parts[ty][0][:, idx], parts[ty][1], parts[ty][2])
        for k in out:
            out[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in out.items()}


def main():
    load.require_key()
    std, cfbd_tal, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    roster = pff.build_roster_talent()
    talent = blended_talent(cfbd_tal, roster)

    # a WAR-only talent vector, for the "can WAR stand alone" question
    war_tal = war.talent_by_year(
        {y: s.index for y, s in cfbd_tal.items() if s is not None})
    war_only = {}
    for N, base in cfbd_tal.items():
        v = war_tal.get(N)
        war_only[N] = base if v is None else v.reindex(base.index).fillna(base)

    args = (std, ret, games, pyth, ret_raw, od)

    print("=" * 70)
    print("FEATURE SETS  (talent = shipping PFF+CFBD+WAR blend), LOSO 2021-25")
    print("=" * 70)
    print(f"  {'set':<26}{'n':>3}{'Brier':>10}{'LogLoss':>10}{'Acc':>8}{'dBrier':>9}")
    base = None
    results = {}
    for nm, cols in CANDIDATES.items():
        m = run(cols, *args, talent)
        results[nm] = m
        if base is None:
            base = m["brier"]
        print(f"  {nm:<26}{len(cols):>3}{m['brier']:>10.4f}{m['log_loss']:>10.4f}"
              f"{m['accuracy']:>8.3f}{m['brier'] - base:>+9.4f}")
    best = min(results, key=lambda k: results[k]["brier"])
    print(f"\n  best: {best} ({results[best]['brier']:.4f})")

    print("\n" + "=" * 70)
    print("CAN WAR CARRY THE TALENT SLOT ALONE?  (talent = WAR only)")
    print("=" * 70)
    print(f"  {'set':<26}{'n':>3}{'Brier':>10}{'LogLoss':>10}{'Acc':>8}")
    for nm in ("all six (ships today)", "drop both", "O/D + talent", "talent only"):
        m = run(CANDIDATES[nm], *args, war_only)
        print(f"  {nm:<26}{len(CANDIDATES[nm]):>3}{m['brier']:>10.4f}"
              f"{m['log_loss']:>10.4f}{m['accuracy']:>8.3f}")


if __name__ == "__main__":
    main()
