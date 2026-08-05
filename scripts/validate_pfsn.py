"""Validate PFSN talent vs CFBD-247 talent as the model's talent signal.

LEAKAGE-SAFE: PFSN talent for season N is built from season-N performance grades,
so it is NOT known preseason. We therefore use PRIOR-season PFSN (N-1) as the
talent input for predicting season N (exactly how the 2026 file maps 2025 grades
onto the 2026 roster). CFBD recruiting talent is forward-known, so it stays at N.

Variants compared via LOSO (everything else identical: opp-adj O/D, TruMedia,
uncertainty): CFBD[N]  vs  PFSN[N-1]  vs  blend. Where a team lacks PFSN, fall
back to CFBD (the realistic deployment). Run: ./venv/bin/python -m scripts.validate_pfsn
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import GAME_YEARS, UNCERTAINTY_LAMBDA, OPP_ADJ_ALPHA
from src.data import load
from src import matchup as MU
from src import oppadj as OA
from src import model as M
from scripts.train import load_bundle, raw_returning

from config import PFSN_MASTER, require

MASTER = str(require(PFSN_MASTER, "the PFSN master workbook", "CFB_PFSN_MASTER"))
COMBO = ("stable_7pos_available_2019_2025", "depth_curve_top5", "free_simplex")
ALIAS = {"Miami (FL)": "Miami", "Connecticut": "UConn", "Mississippi": "Ole Miss",
         "Appalachian State": "App State", "Hawaii": "Hawai'i", "USF": "South Florida",
         "San Jose State": "San José State", "Louisiana-Monroe": "UL Monroe",
         "Miami (Ohio)": "Miami (OH)", "North Carolina State": "NC State",
         "Sam Houston State": "Sam Houston"}


def _z(s):
    s = s.astype(float)
    return (s - s.mean()) / (s.std(ddof=0) or 1.0)


def load_pfsn(cfbd_names):
    df = pd.read_excel(MASTER, sheet_name="Optimizer Team Scores")
    sc, me, op = COMBO
    df = df[(df.scope == sc) & (df.method == me) & (df.optimizer == op)]
    cset = set(cfbd_names)

    def m(t):
        t = str(t)
        if t in cset:
            return t
        return ALIAS.get(t) if ALIAS.get(t) in cset else None
    df = df.copy()
    df["cfbd"] = df["team"].map(m)
    unm = sorted(df.loc[df.cfbd.isna(), "team"].unique())
    if unm:
        print(f"  [warn] unmatched PFSN teams ({len(unm)}): {unm[:20]}")
    df = df.dropna(subset=["cfbd"]).drop_duplicates(["season", "cfbd"])
    pfsn_z = {y: _z(g.set_index("cfbd")["team_talent_score"])
              for y, g in df.groupby("season")}
    winpct = {y: g.set_index("cfbd")["win_pct"] for y, g in df.groupby("season")}
    return pfsn_z, winpct


def run_loso(talent_by_year, std, ret, games, pyth, ret_raw):
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    folds = [y for y in GAME_YEARS if (y - 1) in std]
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
    print("Loading bundle + PFSN historical talent ...")
    std, cfbd_tal, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()
    cfbd_names = set().union(*[set(std[y].index) for y in std])
    pfsn_z, winpct = load_pfsn(cfbd_names)

    # Build talent variants keyed by GAME-YEAR N.
    cfbd_var, pfsn_var, blend_var = {}, {}, {}
    for N in std:                       # cover all years team_frame may touch
        cfbd_var[N] = cfbd_tal.get(N)
        prior = pfsn_z.get(N - 1)
        base = cfbd_tal.get(N)
        if base is None:
            continue
        if prior is None:
            pfsn_var[N] = base
            blend_var[N] = base
        else:
            p = prior.reindex(base.index)
            pfsn_var[N] = p.fillna(base)                       # PFSN[N-1], CFBD fill
            blend_var[N] = (0.5 * p.fillna(base) + 0.5 * base)  # 50/50 blend

    # ---- Test 1: forward correlation with next-year win pct ----
    print("\n=== Forward signal: talent[N-1] -> win_pct[N] (leakage-free) ===")
    for name, src in [("CFBD[N-1]", {y: cfbd_tal[y] for y in cfbd_tal}),
                      ("PFSN[N-1]", pfsn_z)]:
        xs, ys = [], []
        for N in winpct:
            if (N - 1) in src and N in winpct:
                a, b = src[N - 1], winpct[N]
                c = a.index.intersection(b.index)
                xs += list(a.loc[c]); ys += list(b.loc[c])
        print(f"  {name:<10} r with next-year win% = {np.corrcoef(xs, ys)[0,1]:.3f}  (n={len(xs)})")

    # ---- Test 2: in-model LOSO ----
    print("\n=== In-model LOSO (talent signal swapped; all else identical) ===")
    print(f"{'talent':<16}{'Brier':>9}{'LogLoss':>10}{'Acc':>8}")
    print("-" * 44)
    for name, var in [("CFBD[N] (base)", cfbd_var), ("PFSN[N-1]", pfsn_var),
                      ("blend 50/50", blend_var)]:
        m = run_loso(var, std, ret, games, pyth, ret_raw)
        print(f"{name:<16}{m['brier']:>9.4f}{m['log_loss']:>10.4f}{m['accuracy']:>8.3f}")


if __name__ == "__main__":
    main()
