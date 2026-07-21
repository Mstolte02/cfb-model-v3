"""LOSO round 2: combine round-1 winners (no isotonic, pythag exponent) and test
margin-of-victory information two ways:

  A. sample-weighted logistic (blowouts weigh more than one-score games)
  B. ensemble with a ridge margin model on the SAME matchup features,
     converted to prob via Phi(pred_margin / sigma)

Run: ./venv/bin/python -m scripts.loso_experiments2
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, cross_val_predict

from config import GAME_YEARS, UNCERTAINTY_LAMBDA, OPP_ADJ_ALPHA, C_GRID, TALENT_BLEND
from src.data import load, pff
from src import matchup as MU
from src import oppadj as OA
from src import projection as P
from scripts.train import load_bundle, raw_returning, blended_talent
from scripts.loso_experiments import metrics

FOLDS = [2022, 2023, 2024, 2025]


def build_year_m(frame, games_df):
    """Like MU.build_year but also returns the home margin per game."""
    teams = set(frame.index)
    X, y, hf, mg = [], [], [], []
    for _, g in games_df.iterrows():
        h, a = g["home_team"], g["away_team"]
        if h not in teams or a not in teams or g["home_points"] == g["away_points"]:
            continue
        fh, fa = frame.loc[h], frame.loc[a]
        X.append([fh.O - fa.D, fh.D - fa.O, fh.fp_margin - fa.fp_margin,
                  fh.pythag - fa.pythag, fh.talent - fa.talent,
                  fh.returning - fa.returning])
        y.append(1 if g["home_points"] > g["away_points"] else 0)
        hf.append(0 if g.get("neutral_site", False) else 1)
        mg.append(float(g["home_points"] - g["away_points"]))
    return (np.array(X, float), np.array(y), np.array(hf, float), np.array(mg))


def assemble_m(std, pyth, talent, ret, games, ret_raw, od, tr_years):
    b_o, b_d = MU.fit_talent_od_slopes(tr_years, std, talent, od_by_year=od)
    parts = {}
    for N in GAME_YEARS:
        u = (1.0 - ret_raw[N]).clip(lower=0, upper=1) if N in ret_raw else None
        unc = (UNCERTAINTY_LAMBDA, b_o, b_d, u) if u is not None else None
        frame = MU.team_frame(N, std, pyth, talent, ret, uncertainty=unc,
                              od_by_year=od)
        if frame is None:
            continue
        parts[N] = build_year_m(frame, games[N])
    return parts


def fit_logistic(X, y, hf, sample_weight=None):
    design = np.column_stack([X, hf])
    scores = {c: cross_val_score(LogisticRegression(C=c, max_iter=2000), design, y,
                                 cv=5, scoring="neg_brier_score").mean()
              for c in C_GRID}
    C = max(scores, key=scores.get)
    clf = LogisticRegression(C=C, max_iter=2000).fit(design, y,
                                                     sample_weight=sample_weight)
    return clf


def platt_from(clf, design, y):
    oof = clf.predict_proba(design)[:, 1]
    z = np.log(np.clip(oof, 1e-6, 1 - 1e-6) / np.clip(1 - oof, 1e-6, 1))
    return LogisticRegression(C=1e6, max_iter=2000).fit(z.reshape(-1, 1), y)


def apply_platt(pl, p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    return pl.predict_proba(z.reshape(-1, 1))[:, 1]


def run(std, talent, ret, games, pyth, ret_raw, mode, ens_w=0.7):
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    agg = {"brier": [], "log_loss": [], "accuracy": []}
    for ty in FOLDS:
        tr = [g for g in GAME_YEARS if g != ty]
        parts = assemble_m(std, pyth, talent, ret, games, ret_raw, od, tr)
        Xtr = np.vstack([parts[g][0] for g in tr if g in parts])
        ytr = np.concatenate([parts[g][1] for g in tr if g in parts])
        hft = np.concatenate([parts[g][2] for g in tr if g in parts])
        mgt = np.concatenate([parts[g][3] for g in tr if g in parts])
        Xte, yte, hfe, _ = parts[ty]
        dtr = np.column_stack([Xtr, hft])
        dte = np.column_stack([Xte, hfe])

        if mode == "plain":                       # logistic, no calibration
            clf = fit_logistic(Xtr, ytr, hft)
            p = clf.predict_proba(dte)[:, 1]
        elif mode == "mov_weight":                # sample-weighted + platt fix
            w = 0.5 + np.minimum(np.abs(mgt), 28.0) / 28.0
            clf = fit_logistic(Xtr, ytr, hft, sample_weight=w)
            pl = platt_from(clf, dtr, ytr)
            p = apply_platt(pl, clf.predict_proba(dte)[:, 1])
        elif mode == "ensemble":                  # logistic + margin-model prob
            clf = fit_logistic(Xtr, ytr, hft)
            grid = [0.5, 1.0, 3.0, 10.0, 30.0]
            alpha = max(grid, key=lambda a: cross_val_score(
                Ridge(alpha=a), dtr, mgt, cv=5,
                scoring="neg_mean_absolute_error").mean())
            rg = Ridge(alpha=alpha).fit(dtr, mgt)
            sigma = float(np.std(mgt - rg.predict(dtr)))
            p_lg = clf.predict_proba(dte)[:, 1]
            p_mg = norm.cdf(rg.predict(dte) / sigma)
            p = ens_w * p_lg + (1 - ens_w) * p_mg
        else:
            raise ValueError(mode)

        m = metrics(np.asarray(p), yte)
        for k in agg:
            agg[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in agg.items()}


def main():
    load.require_key()
    print("Loading bundle + PFF roster talent ...")
    std, cfbd_tal, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()
    talent = blended_talent(cfbd_tal, pff.build_roster_talent(), w=TALENT_BLEND)
    pyth28 = P.build_pythag(games, 2.8)

    exps = [
        ("plain (no calib)  py2.37", pyth,   "plain",      {}),
        ("plain (no calib)  py2.8",  pyth28, "plain",      {}),
        ("mov-weighted      py2.37", pyth,   "mov_weight", {}),
        ("ensemble w=0.7    py2.37", pyth,   "ensemble",   {"ens_w": 0.7}),
        ("ensemble w=0.5    py2.37", pyth,   "ensemble",   {"ens_w": 0.5}),
        ("ensemble w=0.5    py2.8",  pyth28, "ensemble",   {"ens_w": 0.5}),
        ("ensemble w=0.3    py2.37", pyth,   "ensemble",   {"ens_w": 0.3}),
    ]
    print(f"\n{'experiment':<28}{'Brier':>9}{'LogLoss':>10}{'Acc':>8}")
    print("-" * 55)
    for name, py, mode, kw in exps:
        m = run(std, talent, ret, games, py, ret_raw, mode, **kw)
        print(f"{name:<28}{m['brier']:>9.4f}{m['log_loss']:>10.4f}{m['accuracy']:>8.3f}",
              flush=True)


if __name__ == "__main__":
    main()
