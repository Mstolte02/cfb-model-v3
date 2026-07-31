"""Model efficacy / validity diagnostics for the win model (and the spread OLS):

  1. Collinearity     - VIF of the matchup features
  2. Significance     - statsmodels Logit coefficients + p-values
  3. Overfit          - in-sample vs out-of-fold (LOSO) Brier/acc, per-season
  4. Calibration      - reliability curve + calibration slope/intercept (OOF)
  5. Stability        - bootstrap coefficient variation
  6. Autocorrelation  - Durbin-Watson over time + residual clustering BY TEAM
  7. Heteroscedasticity - Breusch-Pagan on the spread (points) OLS residuals

Run: ./venv/bin/python -m scripts.diagnostics
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

from config import GAME_YEARS, UNCERTAINTY_LAMBDA, OPP_ADJ_ALPHA
from src.data import load, pff
from src import matchup as MU, oppadj as OA, model as M, spread as SP
from scripts.train import load_bundle, raw_returning, blended_talent

COLS = MU.MATCHUP_COLS + ["home"]


def rows_with_meta(frame, gdf):
    teams = set(frame.index)
    X, y, hf, meta = [], [], [], []
    for _, g in gdf.iterrows():
        h, a = g["home_team"], g["away_team"]
        if h not in teams or a not in teams or g["home_points"] == g["away_points"]:
            continue
        X.append(MU.matchup_vector(frame, h, a))
        y.append(1 if g["home_points"] > g["away_points"] else 0)
        hf.append(0 if g.get("neutral_site", False) else 1)
        meta.append((g["season"], g.get("week", 0), h, a))
    return (np.array(X), np.array(y), np.array(hf),
            pd.DataFrame(meta, columns=["season", "week", "home", "away"]))


def build(std, talent, ret_raw, pyth, ret, games, od, slope_years):
    """frames + rows for all GAME_YEARS, uncertainty slopes fit on slope_years."""
    b_o, b_d = MU.fit_talent_od_slopes(slope_years, std, talent, od_by_year=od)
    out = {}
    for N in GAME_YEARS:
        if (N - 1) not in std and (N - 1) not in od:
            continue
        u = MU.uncertainty_u(ret_raw[N]) if N in ret_raw else None
        unc = (UNCERTAINTY_LAMBDA, b_o, b_d, u) if u is not None else None
        fr = MU.team_frame(N, std, pyth, talent, ret, uncertainty=unc, od_by_year=od)
        if fr is not None:
            out[N] = rows_with_meta(fr, games[N])
    return out


def main():
    load.require_key()
    print("Assembling model data ...")
    std, cfbd_tal, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()
    talent = blended_talent(cfbd_tal, pff.build_roster_talent())
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)

    parts = build(std, talent, ret_raw, pyth, ret, games, od, GAME_YEARS)
    X = np.vstack([np.column_stack([parts[g][0], parts[g][2]]) for g in parts])
    y = np.concatenate([parts[g][1] for g in parts])
    design = pd.DataFrame(X, columns=COLS)

    # 1. VIF
    print("\n=== 1. Collinearity (VIF; >5 notable, >10 serious) ===")
    Xc = sm.add_constant(design)
    for i, c in enumerate(Xc.columns):
        if c == "const":
            continue
        print(f"  {c:<12} VIF = {variance_inflation_factor(Xc.values, i):.2f}")

    # 2. Coefficient significance (unregularized Logit for inference)
    print("\n=== 2. Coefficients + significance (statsmodels Logit) ===")
    logit = sm.Logit(y, sm.add_constant(design)).fit(disp=0)
    for c in Xc.columns:
        print(f"  {c:<12} coef={logit.params[c]:+.3f}  p={logit.pvalues[c]:.4f}")
    print(f"  pseudo-R^2 = {logit.prsquared:.4f}")

    # 3 + 4. Out-of-fold (LOSO) predictions -> overfit + calibration + residuals
    folds = sorted(parts)
    oof_p, oof_y, oof_meta, in_brier = [], [], [], []
    for ty in folds:
        tr = [g for g in folds if g != ty]
        p2 = build(std, talent, ret_raw, pyth, ret, games, od, tr)  # refit slopes
        Xtr = np.vstack([p2[g][0] for g in tr]); ytr = np.concatenate([p2[g][1] for g in tr])
        hf = np.concatenate([p2[g][2] for g in tr])
        mdl, _ = M.train(Xtr, ytr, hf)
        in_brier.append(M.evaluate(mdl, Xtr, ytr, hf)["brier"])
        Xte, yte, hfte, meta = p2[ty]
        p = np.array([mdl.win_prob(Xte[i], hfte[i]) for i in range(len(yte))])
        oof_p.append(p); oof_y.append(yte); meta = meta.assign(p=p, y=yte); oof_meta.append(meta)
    p = np.clip(np.concatenate(oof_p), 1e-6, 1 - 1e-6)
    yv = np.concatenate(oof_y)
    md = pd.concat(oof_meta, ignore_index=True)

    print("\n=== 3. Overfit (in-sample vs out-of-fold) ===")
    print(f"  in-sample Brier  : {np.mean(in_brier):.4f}")
    print(f"  out-of-fold Brier: {np.mean((p-yv)**2):.4f}  (gap "
          f"{np.mean((p-yv)**2)-np.mean(in_brier):+.4f})")
    print(f"  OOF accuracy: {np.mean((p>0.5)==yv):.3f}")
    print("  per-season OOF Brier:")
    for s in sorted(md.season.unique()):
        m = md.season == s
        print(f"    {int(s)}: {np.mean((md.p[m]-md.y[m])**2):.4f} (n={m.sum()})")

    print("\n=== 4. Calibration (out-of-fold) ===")
    logit_p = np.log(p / (1 - p))
    cal = sm.Logit(yv, sm.add_constant(logit_p)).fit(disp=0)
    print(f"  calibration slope = {cal.params[1]:.3f} (1.0=perfect), "
          f"intercept = {cal.params[0]:+.3f} (0=perfect)")
    bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    print("  reliability:")
    for b in range(10):
        mb = idx == b
        if mb.sum() > 5:
            print(f"    {bins[b]:.1f}-{bins[b+1]:.1f}: pred {p[mb].mean():.3f} "
                  f"actual {yv[mb].mean():.3f} (n={mb.sum()})")

    # 5. Bootstrap coefficient stability
    print("\n=== 5. Coefficient stability (100 bootstraps) ===")
    boots = []
    Xa = design.values
    for i in range(100):
        Xb, yb = resample(Xa, y, random_state=i)
        boots.append(LogisticRegression(C=0.1, max_iter=2000).fit(Xb, yb).coef_[0])
    boots = np.array(boots)
    for j, c in enumerate(COLS):
        mean, sd = boots[:, j].mean(), boots[:, j].std()
        print(f"  {c:<12} {mean:+.3f} +/- {sd:.3f}  (CV {abs(sd/mean) if mean else 0:.2f})")

    # 6. Autocorrelation
    print("\n=== 6. Autocorrelation / independence ===")
    md2 = md.sort_values(["season", "week"]).reset_index(drop=True)
    dw = durbin_watson(md2.y - md2.p)
    print(f"  Durbin-Watson (residuals over time) = {dw:.3f}  (2.0 = no autocorrelation)")
    # residual clustering by team (signed to each team's perspective)
    tr_res = {}
    for _, r in md.iterrows():
        tr_res.setdefault(r.home, []).append(r.y - r.p)
        tr_res.setdefault(r.away, []).append(r.p - r.y)
    team_means = pd.Series({t: np.mean(v) for t, v in tr_res.items() if len(v) >= 8})
    null_se = np.sqrt(np.mean(p * (1 - p))) / np.sqrt(np.mean([len(v) for v in tr_res.values()]))
    print(f"  per-team mean-residual std = {team_means.std():.4f} vs ~{null_se:.4f} "
          f"expected if independent  ({'OK' if team_means.std() < 2*null_se else 'clustered'})")
    print(f"  most over-rated teams: "
          f"{', '.join(team_means.nsmallest(3).index)}; "
          f"under-rated: {', '.join(team_means.nlargest(3).index)}")

    # 7. Heteroscedasticity on the spread (points) OLS
    print("\n=== 7. Heteroscedasticity (spread/points OLS, Breusch-Pagan) ===")
    fr_all = {}
    b_o, b_d = MU.fit_talent_od_slopes(GAME_YEARS, std, talent, od_by_year=od)
    for N in GAME_YEARS:
        u = MU.uncertainty_u(ret_raw[N]) if N in ret_raw else None
        unc = (UNCERTAINTY_LAMBDA, b_o, b_d, u) if u is not None else None
        f = MU.team_frame(N, std, pyth, talent, ret, uncertainty=unc, od_by_year=od)
        if f is not None:
            fr_all[N] = f
    Xpts = np.vstack([SP.build_points_rows(fr_all[g], games[g])[0] for g in fr_all])
    ypts = np.concatenate([SP.build_points_rows(fr_all[g], games[g])[1] for g in fr_all])
    ols = sm.OLS(ypts, sm.add_constant(Xpts)).fit()
    lm, lm_p, _, _ = het_breuschpagan(ols.resid, ols.model.exog)
    print(f"  points-model R^2 = {ols.rsquared:.3f}, residual SD = {np.std(ols.resid):.2f} pts")
    print(f"  Breusch-Pagan p = {lm_p:.4f}  ({'heteroscedastic' if lm_p < 0.05 else 'homoscedastic'})")


if __name__ == "__main__":
    main()
