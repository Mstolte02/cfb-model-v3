"""Data exploration over the COMBINED pool (CFBD + TruMedia + tempo), BEFORE
pruning. Addresses three questions:

  1. Do the new TruMedia stats (RZ TD%, PFF pressure, field position, ...) carry
     forward predictive signal?
  2. Tempo as an interaction: does efficiency x opportunities (plays/possessions
     per game) beat efficiency alone? (Per-play stats are aggregated from a play
     level; volume could compound.)
  3. Residualize collinear stats instead of dropping: regress the stat on PPA and
     test whether the orthogonal residual still predicts.

All targets are FORWARD: feature in year N-1 -> team strength in year N.
Run: ./venv/bin/python -m scripts.explore
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import STAT_YEARS
from src.data import load, cfbd_client, trumedia
from scripts.feature_analysis import _flatten, _z, FEATS


def build_pool():
    """Merged per-(team,season) frame: CFBD broad stats + tempo + TruMedia."""
    tm = trumedia.load()
    frames = []
    for y in STAT_YEARS:
        raw = cfbd_client.advanced_season_stats(y)
        f = _flatten(raw, y)
        # tempo / opportunities from CFBD season totals
        opp = pd.DataFrame([{"team": t["team"],
                             "off_plays": (t.get("offense") or {}).get("plays"),
                             "off_drives": (t.get("offense") or {}).get("drives")}
                            for t in raw])
        f = f.merge(opp, on="team", how="left")
        g = tm[tm["season"] == y][["team", "games"] + trumedia.STAT_COLS]
        f = f.merge(g, on="team", how="left")
        f["plays_pg"] = f["off_plays"] / f["games"]
        f["drives_pg"] = f["off_drives"] / f["games"]
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df["strength_raw"] = df["off_ppa"] - df["def_ppa"]
    return df


def forward(df, feats):
    """Pooled forward correlation of feature(N-1) with strength(N)."""
    by = {y: g.set_index("team") for y, g in df.groupby("season")}
    res = {}
    for f in feats:
        x, t = [], []
        for N in STAT_YEARS:
            if N - 1 in by and N in by:
                a, b = by[N - 1], by[N]
                c = a.index.intersection(b.index)
                aa = a.loc[c, f]; bb = _z(b.loc[c, "strength_raw"])
                ok = aa.notna() & bb.notna()
                x += list(_z(aa[ok])); t += list(bb[ok])
        res[f] = np.corrcoef(x, t)[0, 1] if len(x) > 30 else np.nan
    return pd.Series(res)


def main():
    load.require_key()
    print("Building combined pool (CFBD + TruMedia + tempo) ...")
    df = build_pool()
    tm_feats = trumedia.STAT_COLS
    tempo_feats = ["plays_pg", "drives_pg"]

    # ---- 1. TruMedia + tempo forward signal ----
    sig = forward(df, tm_feats + tempo_feats).reindex(
        (forward(df, tm_feats + tempo_feats).abs().sort_values(ascending=False)).index)
    print("\n=== Forward predictive r (feature N-1 -> strength N) ===")
    for f, r in sig.items():
        print(f"  {f:<16}{r:>7.3f}")

    # ---- 2. Tempo as an interaction with efficiency ----
    print("\n=== Tempo interaction: does efficiency x pace beat efficiency alone? ===")
    by = {y: g.set_index("team") for y, g in df.groupby("season")}
    rows = []
    for N in STAT_YEARS:
        if N - 1 not in by or N not in by:
            continue
        a, b = by[N - 1], by[N]
        c = a.index.intersection(b.index)
        sub = pd.DataFrame({
            "eff": _z(a.loc[c, "off_ppa"]),
            "pace": _z(a.loc[c, "plays_pg"]),
            "y": _z(b.loc[c, "strength_raw"]),
        }).dropna()
        rows.append(sub)
    d = pd.concat(rows)
    d["eff_x_pace"] = d["eff"] * d["pace"]
    m1 = sm.OLS(d["y"], sm.add_constant(d[["eff"]])).fit()
    m2 = sm.OLS(d["y"], sm.add_constant(d[["eff", "pace", "eff_x_pace"]])).fit()
    print(f"  efficiency only         R^2 = {m1.rsquared:.4f}")
    print(f"  + pace + eff x pace     R^2 = {m2.rsquared:.4f}")
    print(f"  interaction coef = {m2.params['eff_x_pace']:+.4f}  "
          f"p = {m2.pvalues['eff_x_pace']:.3f}")
    print(f"  pace main coef   = {m2.params['pace']:+.4f}  p = {m2.pvalues['pace']:.3f}")

    # ---- 3. Residualize a collinear stat instead of dropping ----
    print("\n=== Residualization: orthogonalize a collinear stat vs off_ppa ===")
    for stat in ["off_expl", "off_std_ppa", "off_pts_opp"]:
        if stat not in df.columns:
            continue
        rows = []
        for N in STAT_YEARS:
            if N - 1 not in by or N not in by:
                continue
            a, b = by[N - 1], by[N]
            c = a.index.intersection(b.index)
            sub = pd.DataFrame({"ppa": a.loc[c, "off_ppa"], "x": a.loc[c, stat],
                                "y": b.loc[c, "strength_raw"]}).dropna()
            rows.append(sub)
        d = pd.concat(rows)
        for col in ["ppa", "x", "y"]:
            d[col] = _z(d[col])
        # residual of stat after regressing on ppa
        resid = d["x"] - sm.OLS(d["x"], sm.add_constant(d[["ppa"]])).fit().predict(
            sm.add_constant(d[["ppa"]]))
        r_raw = np.corrcoef(d["x"], d["y"])[0, 1]
        r_res = np.corrcoef(resid, d["y"])[0, 1]
        print(f"  {stat:<18} raw r={r_raw:+.3f}  |  residual-vs-ppa r={r_res:+.3f}  "
              f"(corr w/ ppa = {np.corrcoef(d['x'], d['ppa'])[0,1]:+.2f})")

    # ---- 4. Nonlinearity: does a squared term add anything? ----
    print("\n=== Nonlinearity: add x^2 to the forward fit (p<0.05 = nonlinear) ===")
    for feat in ["off_success", "def_ppa", "off_havoc", "fp_margin", "third_conv"]:
        if feat not in df.columns:
            continue
        rows = []
        for N in STAT_YEARS:
            if N - 1 not in by or N not in by:
                continue
            a, b = by[N - 1], by[N]
            c = a.index.intersection(b.index)
            rows.append(pd.DataFrame({"x": a.loc[c, feat], "y": b.loc[c, "strength_raw"]}).dropna())
        d = pd.concat(rows)
        d["x"] = _z(d["x"]); d["y"] = _z(d["y"]); d["x2"] = d["x"] ** 2
        lin = sm.OLS(d["y"], sm.add_constant(d[["x"]])).fit()
        quad = sm.OLS(d["y"], sm.add_constant(d[["x", "x2"]])).fit()
        print(f"  {feat:<14} linear R^2={lin.rsquared:.4f} -> +x^2 R^2={quad.rsquared:.4f}  "
              f"| x^2 coef={quad.params['x2']:+.3f} p={quad.pvalues['x2']:.3f}")


if __name__ == "__main__":
    main()
