"""Are the six model inputs actually six things?

The suspicion is reasonable: talent, returning and WAR are all roster-quality proxies,
and pythag is built from points scored and allowed, which is what the O and D
composites already measure. This checks it three ways rather than by eye:

  1. correlation matrix and variance inflation factors on the matchup difference
     vectors the model actually fits (not the raw team frame - collinearity in the
     fitted design is what matters)
  2. leave-one-feature-out LOSO, to see what each input is worth once the others
     are present
  3. forward selection, to see how many features the data actually supports

Run: ./venv/bin/python -m scripts.feature_audit
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
from scripts.train import load_bundle, raw_returning, blended_talent

FEATURES = ["O", "D", "fp_margin", "pythag", "talent", "returning"]


def design():
    """The stacked (X, y, home) the model trains on, across all seasons."""
    std, cfbd_tal, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()
    talent = blended_talent(cfbd_tal, pff.build_roster_talent())
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    b_o, b_d = MU.fit_talent_od_slopes(list(GAME_YEARS), std, talent, od_by_year=od)
    parts = MU.assemble(GAME_YEARS, std, pyth, talent, ret, games,
                        lam=UNCERTAINTY_LAMBDA, b_o=b_o, b_d=b_d,
                        ret_raw_by_year=ret_raw, od_by_year=od)
    return parts


def vif(X):
    """Variance inflation factor per column, via R^2 of each on the rest."""
    out = []
    for j in range(X.shape[1]):
        y = X[:, j]
        Z = np.delete(X, j, axis=1)
        Z = np.column_stack([np.ones(len(Z)), Z])
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        resid = y - Z @ beta
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else 0.0
        out.append(1.0 / max(1e-9, 1 - r2))
    return out


def loso(parts, cols):
    """Season-blocked CV on a subset of feature columns."""
    idx = [FEATURES.index(c) for c in cols]
    res = {"brier": [], "log_loss": [], "accuracy": []}
    for ty in GAME_YEARS:
        tr = [g for g in GAME_YEARS if g != ty and g in parts]
        if ty not in parts or not tr:
            continue
        Xtr = np.vstack([parts[g][0][:, idx] for g in tr])
        ytr = np.concatenate([parts[g][1] for g in tr])
        hf = np.concatenate([parts[g][2] for g in tr])
        mdl, _ = M.train(Xtr, ytr, hf)
        m = M.evaluate(mdl, parts[ty][0][:, idx], parts[ty][1], parts[ty][2])
        for k in res:
            res[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in res.items()}


def main():
    load.require_key()
    print("Building the design the model actually fits ...\n")
    parts = design()
    X = np.vstack([parts[g][0] for g in GAME_YEARS if g in parts])
    print(f"rows: {len(X)}   features: {len(FEATURES)}\n")

    # ---- 1. correlation + VIF ------------------------------------------------
    print("=" * 66)
    print("1. CORRELATION between feature differences (the fitted design)")
    print("=" * 66)
    C = pd.DataFrame(np.corrcoef(X, rowvar=False), index=FEATURES, columns=FEATURES)
    print(C.round(2).to_string())
    hi = [(a, b, C.loc[a, b]) for i, a in enumerate(FEATURES)
          for b in FEATURES[i + 1:] if abs(C.loc[a, b]) >= 0.6]
    print("\n  pairs at |r| >= 0.60:")
    for a, b, r in sorted(hi, key=lambda t: -abs(t[2])) or []:
        print(f"    {a:<10} {b:<10} {r:+.2f}")
    if not hi:
        print("    none")

    print("\n  variance inflation factors (>5 is usually called a problem):")
    for f, v in zip(FEATURES, vif(X)):
        flag = "  <-- high" if v > 5 else ""
        print(f"    {f:<12} {v:6.2f}{flag}")

    # ---- 2. drop-one ---------------------------------------------------------
    print("\n" + "=" * 66)
    print("2. LEAVE-ONE-FEATURE-OUT (LOSO 2021-25)")
    print("=" * 66)
    full = loso(parts, FEATURES)
    print(f"  {'feature set':<28}{'Brier':>9}{'LogLoss':>10}{'Acc':>8}{'dBrier':>9}")
    print(f"  {'all six':<28}{full['brier']:>9.4f}{full['log_loss']:>10.4f}"
          f"{full['accuracy']:>8.3f}{'':>9}")
    drops = []
    for f in FEATURES:
        sub = [c for c in FEATURES if c != f]
        m = loso(parts, sub)
        d = m["brier"] - full["brier"]
        drops.append((f, d, m))
        print(f"  {'  without ' + f:<28}{m['brier']:>9.4f}{m['log_loss']:>10.4f}"
              f"{m['accuracy']:>8.3f}{d:>+9.4f}")
    print("\n  (a positive dBrier means dropping the feature HURT - it was earning "
          "its place)")

    # ---- 3. forward selection ------------------------------------------------
    print("\n" + "=" * 66)
    print("3. FORWARD SELECTION - how many features does the data support?")
    print("=" * 66)
    chosen, remaining = [], FEATURES[:]
    best_so_far = 1.0
    while remaining:
        scored = []
        for f in remaining:
            m = loso(parts, chosen + [f])
            scored.append((m["brier"], f, m))
        scored.sort()
        b, f, m = scored[0]
        gain = best_so_far - b
        chosen.append(f)
        remaining.remove(f)
        mark = "" if gain > 0.0002 else "   <-- no material gain from here"
        print(f"  +{f:<12} ({len(chosen)}) Brier {b:.4f}  acc {m['accuracy']:.3f}{mark}")
        best_so_far = b
    print(f"\nfull-set Brier for reference: {full['brier']:.4f}")


if __name__ == "__main__":
    main()
