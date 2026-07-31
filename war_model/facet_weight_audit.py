"""Why does coverage outweigh pass rush by 8x, and is that weighting real?

The WAR build weights facets by random-forest impurity importance against team wins.
That produces cov_db at 23% of all weight - more than quarterback passing - while
prsh_dl gets 2.9%. Every public study of positional value in football puts edge
rushers well above corners, so the burden is on the model here, not the literature.

Three things could produce this and they have different fixes:

  1. impurity importance is biased. It favours features with more split points and
     inflates whichever of a correlated pair gets used first. Permutation importance
     and ridge coefficients do not share that bias, so disagreement between them is
     diagnostic.
  2. the facets are collinear. Pressure and coverage produce each other - a corner
     covers better when the quarterback is hurried - so a forest can hand nearly all
     the credit to one of them arbitrarily.
  3. the weighting is real, and college football genuinely differs from the NFL.

Run: ./rbenv/bin/python facet_weight_audit.py
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict

from build_hybrid import unified_facets
from facets import YEARS

SIDE = {  # which unit each facet belongs to, for the summary
    "pass_qb": "OFF", "run_qb": "OFF", "run_rb": "OFF", "recv_wr": "OFF",
    "recv_te": "OFF", "recv_rb": "OFF", "pblk_ol": "OFF", "pblk_skill": "OFF",
    "rblk_ol": "OFF", "rblk_skill": "OFF", "fumble": "OFF", "pen_off": "OFF",
}
PASS_RUSH = ["prsh_dl", "prsh_sec", "cfbd_havoc_dl", "cfbd_havoc_lb", "cfbd_havoc_db"]
COVERAGE = ["cov_db", "cov_lb", "cfbd_cov_db"]


def team_matrix():
    fv = unified_facets()
    names = sorted(fv.facet.unique())
    tot = (fv.groupby(["season", "team", "facet"], as_index=False)["value"].sum()
             .pivot(index=["season", "team"], columns="facet", values="value")
             .reindex(columns=names).fillna(0.0))
    Z = tot.groupby(level="season").transform(
        lambda c: (c - c.mean()) / c.std(ddof=0)).fillna(0.0)
    recs = pd.read_csv("records.csv").set_index(["season", "team"])
    df = Z.join(recs, how="inner")
    return df, names


def main():
    df, names = team_matrix()
    X = df[names].to_numpy(float)
    y = df.adj_win_pct.to_numpy(float)
    groups = df.index.get_level_values("season").to_numpy()
    print(f"team-seasons: {len(df)}   facets: {len(names)}\n")

    # ---- 1. do three importance measures agree? ----------------------------
    rf = RandomForestRegressor(n_estimators=600, min_samples_leaf=3,
                               random_state=0, n_jobs=-1).fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=names)
    imp /= imp.sum()

    perm = permutation_importance(rf, X, y, n_repeats=15, random_state=0, n_jobs=-1)
    pim = pd.Series(np.clip(perm.importances_mean, 0, None), index=names)
    pim = pim / pim.sum() if pim.sum() else pim

    ridge = RidgeCV(alphas=np.logspace(-2, 3, 40)).fit(X, y)
    rw = pd.Series(ridge.coef_, index=names)
    rwp = rw.clip(lower=0)
    rwp = rwp / rwp.sum() if rwp.sum() else rwp

    comp = pd.DataFrame({"rf_impurity": imp, "rf_permutation": pim,
                         "ridge_pos": rwp, "ridge_raw": rw})
    print("=" * 74)
    print("1. THREE WAYS OF ASKING WHICH FACETS MATTER")
    print("=" * 74)
    print(comp.sort_values("rf_impurity", ascending=False).head(14).round(4).to_string())

    print("\n  pass rush vs coverage, by measure:")
    for col in ("rf_impurity", "rf_permutation", "ridge_pos"):
        pr = comp.loc[[f for f in PASS_RUSH if f in comp.index], col].sum()
        cv = comp.loc[[f for f in COVERAGE if f in comp.index], col].sum()
        print(f"    {col:<16} pass rush {pr*100:5.1f}%   coverage {cv*100:5.1f}%   "
              f"ratio {cv/max(pr,1e-9):.1f}x")

    # ---- 2. are they collinear? --------------------------------------------
    print("\n" + "=" * 74)
    print("2. ARE COVERAGE AND PASS RUSH MEASURING THE SAME THING?")
    print("=" * 74)
    keyf = [f for f in ("cov_db", "prsh_dl", "rdef_sec", "rdef_dl", "cov_lb",
                        "prsh_sec", "cfbd_havoc_dl", "tackle") if f in df.columns]
    C = df[keyf].corr()
    print(C.round(2).to_string())
    print("\n  each defensive facet's own correlation with adjusted win pct:")
    for f in keyf:
        print(f"    {f:<16} r = {np.corrcoef(df[f], y)[0,1]:+.3f}")

    # ---- 3. what happens if you drop one? ----------------------------------
    print("\n" + "=" * 74)
    print("3. DROP-ONE CV (season-blocked). If a facet is doing unique work,")
    print("   removing it should hurt.")
    print("=" * 74)

    def cv(cols):
        m = RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                  random_state=0, n_jobs=-1)
        p = cross_val_predict(m, df[cols].to_numpy(float), y,
                              cv=GroupKFold(n_splits=5), groups=groups)
        return float(np.corrcoef(p, y)[0, 1])

    base = cv(names)
    print(f"  all facets: r = {base:.4f}")
    for f in ("cov_db", "prsh_dl", "pass_qb", "rdef_sec", "cfbd_pass_qb"):
        if f not in names:
            continue
        r = cv([c for c in names if c != f])
        print(f"    without {f:<16} r = {r:.4f}   delta {r-base:+.4f}")

    # ---- 4. stability of each facet's own player-level signal ---------------
    print("\n" + "=" * 74)
    print("4. YEAR-OVER-YEAR STABILITY of each facet at the player level.")
    print("   A facet carrying heavy weight but low stability is being asked to")
    print("   predict wins with a number that barely repeats.")
    print("=" * 74)
    fv = unified_facets()
    rows = []
    for f in ("cov_db", "prsh_dl", "pass_qb", "rdef_dl", "rdef_sec", "cov_lb"):
        d = fv[fv.facet == f]
        if d.empty:
            continue
        d = d[d.snaps >= d.snaps.quantile(0.5)]
        nxt = d[["season", "uid", "z"]].copy()
        nxt["season"] -= 1
        j = d.merge(nxt.rename(columns={"z": "z1"}), on=["season", "uid"])
        if len(j) > 40:
            rows.append({"facet": f, "yoy_r": round(float(np.corrcoef(j.z, j.z1)[0, 1]), 3),
                         "n": len(j), "weight": round(float(imp.get(f, 0)), 4)})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
