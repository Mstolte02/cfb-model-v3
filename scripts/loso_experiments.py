"""LOSO experiment harness for accuracy improvements (cfb-model-v2).

Runs leave-one-season-out CV (folds 2022-2025, where PFF roster talent exists)
for the current default config (baseline) and a set of candidate improvements:

  - two-year O/D prior blend (N-1 stats blended with N-2)
  - calibration variants (isotonic default vs Platt vs none)
  - talent blend weight grid (PFF roster vs CFBD)
  - uncertainty lambda grid
  - Pythagorean exponent grid

Keep whatever beats baseline on Brier + log-loss. Run:
    ./venv/bin/python -m scripts.loso_experiments
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict

from config import GAME_YEARS, UNCERTAINTY_LAMBDA, OPP_ADJ_ALPHA, C_GRID, TALENT_BLEND
from src.data import load, pff
from src import matchup as MU
from src import oppadj as OA
from src import projection as P
from scripts.train import load_bundle, raw_returning, blended_talent

FOLDS = [2022, 2023, 2024, 2025]


def train_variant(X, y, hf, calib="iso"):
    """L2 logistic (C by CV) + calibration variant. Returns predict_fn(X, hf)."""
    design = np.column_stack([X, hf.astype(float)])
    scores = {c: cross_val_score(LogisticRegression(C=c, max_iter=2000),
                                 design, y, cv=5, scoring="neg_brier_score").mean()
              for c in C_GRID}
    C = max(scores, key=scores.get)
    clf = LogisticRegression(C=C, max_iter=2000).fit(design, y)

    calibrate = lambda p: p
    if calib != "none":
        oof = cross_val_predict(LogisticRegression(C=C, max_iter=2000), design, y,
                                cv=5, method="predict_proba")[:, 1]
        if calib == "iso":
            iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
            grid = np.linspace(0, 1, 101)
            gy = iso.predict(grid)
            calibrate = lambda p: np.interp(p, grid, gy)
        elif calib == "platt":
            z = np.log(np.clip(oof, 1e-6, 1 - 1e-6) / np.clip(1 - oof, 1e-6, 1))
            pl = LogisticRegression(C=1e6, max_iter=2000).fit(z.reshape(-1, 1), y)
            calibrate = lambda p: pl.predict_proba(
                np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
                .reshape(-1, 1))[:, 1]

    def predict(Xt, hft):
        d = np.column_stack([Xt, hft.astype(float)])
        return np.asarray(calibrate(clf.predict_proba(d)[:, 1]))
    return predict


def metrics(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "accuracy": float(np.mean((p > 0.5) == y))}


def blend_od(od, w):
    """Blend each year's opponent-adjusted O/D with the previous year's (2-yr prior)."""
    out = {}
    for y, cur in od.items():
        prev = od.get(y - 1)
        if prev is None or w >= 1.0:
            out[y] = cur
            continue
        b = pd.DataFrame(index=cur.index)
        for c in ("O", "D"):
            pv = prev[c].reindex(cur.index).fillna(cur[c])
            v = w * cur[c] + (1 - w) * pv
            b[c] = (v - v.mean()) / (v.std(ddof=0) or 1.0)
        out[y] = b
    return out


def run_loso(std, talent, ret, games, pyth, ret_raw,
             od_w=1.0, calib="iso", unc_lam=UNCERTAINTY_LAMBDA):
    od = blend_od(OA.build_od_by_year(std, games, OPP_ADJ_ALPHA), od_w)
    agg = {"brier": [], "log_loss": [], "accuracy": []}
    for ty in FOLDS:
        tr = [g for g in GAME_YEARS if g != ty]
        b_o, b_d = MU.fit_talent_od_slopes(tr, std, talent, od_by_year=od)
        parts = MU.assemble(GAME_YEARS, std, pyth, talent, ret, games,
                            lam=unc_lam, b_o=b_o, b_d=b_d,
                            ret_raw_by_year=ret_raw, od_by_year=od)
        Xtr = np.vstack([parts[g][0] for g in tr if g in parts])
        ytr = np.concatenate([parts[g][1] for g in tr if g in parts])
        hf = np.concatenate([parts[g][2] for g in tr if g in parts])
        predict = train_variant(Xtr, ytr, hf, calib=calib)
        Xt, yt, hft = parts[ty]
        m = metrics(predict(Xt, hft), yt)
        for k in agg:
            agg[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in agg.items()}


def main():
    load.require_key()
    print("Loading bundle + PFF roster talent ...")
    std, cfbd_tal, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()
    roster = pff.build_roster_talent()

    def talent_w(w):
        return blended_talent(cfbd_tal, roster, w=w)

    tal_default = talent_w(TALENT_BLEND)

    experiments = [
        ("baseline (current default)", dict(talent=tal_default)),
        ("2yr O/D prior w=0.85", dict(talent=tal_default, od_w=0.85)),
        ("2yr O/D prior w=0.70", dict(talent=tal_default, od_w=0.70)),
        ("calib=platt", dict(talent=tal_default, calib="platt")),
        ("calib=none", dict(talent=tal_default, calib="none")),
        ("talent blend w=0.35", dict(talent=talent_w(0.35))),
        ("talent blend w=0.65", dict(talent=talent_w(0.65))),
        ("uncertainty lam=0.0", dict(talent=tal_default, unc_lam=0.0)),
        ("uncertainty lam=0.5", dict(talent=tal_default, unc_lam=0.5)),
    ]

    print(f"\n{'experiment':<28}{'Brier':>9}{'LogLoss':>10}{'Acc':>8}")
    print("-" * 55)
    results = {}
    for name, kw in experiments:
        tal = kw.pop("talent")
        m = run_loso(std, tal, ret, games, pyth, ret_raw, **kw)
        results[name] = m
        print(f"{name:<28}{m['brier']:>9.4f}{m['log_loss']:>10.4f}{m['accuracy']:>8.3f}",
              flush=True)

    # Pythag exponent grid (rebuilds pythag)
    for exp in (2.0, 2.8):
        pyth_v = P.build_pythag(games, exp)
        m = run_loso(std, tal_default, ret, games, pyth_v, ret_raw)
        name = f"pythag exp={exp}"
        results[name] = m
        print(f"{name:<28}{m['brier']:>9.4f}{m['log_loss']:>10.4f}{m['accuracy']:>8.3f}",
              flush=True)


if __name__ == "__main__":
    main()
