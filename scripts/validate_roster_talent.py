"""The real test: roster-aware, transfer-aware PFF talent (N roster + N-1 grades)
vs CFBD-247 talent, as the model's talent signal. This captures roster turnover —
the thing the historical team-level PFSN scores could NOT (last test was flat).

LOSO over seasons where prior-year grades exist (2022-2025). Where a team lacks
PFF roster talent, fall back to CFBD. Run: ./venv/bin/python -m scripts.validate_roster_talent
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import GAME_YEARS, UNCERTAINTY_LAMBDA, OPP_ADJ_ALPHA
from src.data import load, pff
from src import matchup as MU
from src import oppadj as OA
from src import model as M
from scripts.train import load_bundle, raw_returning


def run_loso(talent_by_year, std, ret, games, pyth, ret_raw, folds):
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    out = {"brier": [], "log_loss": [], "accuracy": []}
    for ty in folds:
        tr = [g for g in GAME_YEARS if g != ty]
        b_o, b_d = MU.fit_talent_od_slopes(tr, std, talent_by_year, od_by_year=od)
        parts = MU.assemble(GAME_YEARS, std, pyth, talent_by_year, ret, games,
                            lam=UNCERTAINTY_LAMBDA, b_o=b_o, b_d=b_d,
                            ret_raw_by_year=ret_raw, od_by_year=od)
        Xtr = np.vstack([parts[g][0] for g in tr if g in parts])
        ytr = np.concatenate([parts[g][1] for g in tr if g in parts])
        hf = np.concatenate([parts[g][2] for g in tr if g in parts])
        mdl, _ = M.train(Xtr, ytr, hf)
        m = M.evaluate(mdl, *parts[ty])
        for k in out:
            out[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in out.items()}


def main():
    load.require_key()
    print("Building roster-aware PFF talent + bundle ...")
    std, cfbd_tal, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()
    roster = pff.build_roster_talent()
    print(f"  roster-aware talent seasons: {sorted(roster)} "
          f"(teams/season: {[len(roster[y]) for y in sorted(roster)]})")

    folds = [y for y in GAME_YEARS if y in roster]
    print(f"  LOSO folds (roster talent available): {folds}\n")

    cfbd_var, roster_var, blend_var = {}, {}, {}
    for N in std:
        base = cfbd_tal.get(N)
        if base is None:
            continue
        cfbd_var[N] = base
        r = roster.get(N)
        if r is None:
            roster_var[N] = base; blend_var[N] = base
        else:
            r = r.reindex(base.index)
            roster_var[N] = r.fillna(base)
            blend_var[N] = 0.5 * r.fillna(base) + 0.5 * base

    print(f"{'talent':<18}{'Brier':>9}{'LogLoss':>10}{'Acc':>8}   (folds {folds})")
    print("-" * 50)
    for name, var in [("CFBD (base)", cfbd_var), ("PFF roster-aware", roster_var),
                      ("blend 50/50", blend_var)]:
        m = run_loso(var, std, ret, games, pyth, ret_raw, folds)
        print(f"{name:<18}{m['brier']:>9.4f}{m['log_loss']:>10.4f}{m['accuracy']:>8.3f}")


if __name__ == "__main__":
    main()
