"""Which of the 86 candidates are measuring the same football ability?

Three questions, and they are not the same question:

  correlation   do two features move together?
  VIF           is a feature predictable from ALL the others together? A feature can
                correlate below 0.5 with everything and still be a linear combination
                of six of them, which pairwise correlation cannot see.
  factors       is there a small number of latent abilities behind the observed
                metrics - a "passing skill" that accuracy, big-time-throw rate and
                grade are all noisy views of?

The last one is the one that matters for design. If four coverage metrics load on one
factor, they are one skill measured four ways and should probably be combined; if they
load on two, corners are doing two separable jobs and collapsing them destroys real
information.

Outputs the correlation matrix, VIF table, correlated clusters, PCA loadings and
explained variance, factor-analysis loadings, and a keep-separate/combine call per
cluster.

Run: ./rbenv/bin/python collinearity.py [vif_threshold]
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
VIF_THRESHOLD = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
# Distance at which two features are treated as one skill. 0.30 on (1 - |r|) means
# |r| >= 0.70, the conventional line for "these are the same variable".
CLUSTER_CUT = 0.30


def load():
    Z = pd.read_csv(f"{HERE}/candidate_team_z.csv", index_col=[0, 1])
    recs = pd.read_csv(f"{HERE}/records.csv").set_index(["season", "team"])
    df = Z.join(recs, how="inner")
    names = list(Z.columns)
    return df, names


def vif_table(X, names):
    """VIF per feature: 1 / (1 - R^2) of that feature on all the others."""
    out = []
    for j, f in enumerate(names):
        y = X[:, j]
        Zo = np.column_stack([np.ones(len(X)), np.delete(X, j, axis=1)])
        beta, *_ = np.linalg.lstsq(Zo, y, rcond=None)
        ss = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ((y - Zo @ beta) ** 2).sum() / ss if ss > 0 else 0.0
        out.append({"feature": f, "vif": 1 / max(1e-9, 1 - r2), "r2_on_rest": r2})
    return pd.DataFrame(out).sort_values("vif", ascending=False)


def clusters(C, names, cut=CLUSTER_CUT):
    """Average-linkage clusters on 1 - |correlation|."""
    D = 1 - C.abs().to_numpy()
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    lab = hierarchy.fcluster(hierarchy.linkage(squareform(D, checks=False),
                                               method="average"), cut, "distance")
    g = {}
    for f, c in zip(names, lab):
        g.setdefault(int(c), []).append(f)
    return {k: v for k, v in sorted(g.items(), key=lambda kv: -len(kv[1]))}


def factor_report(X, names, members, y):
    """For one correlated cluster: how much of it is a single latent ability?

    A cluster whose first component carries nearly all the variance is one skill; a
    cluster that needs two is two jobs wearing one name. The recommendation also
    checks whether collapsing costs predictive power against the target, because a
    tidy factor structure is not worth losing signal over.
    """
    idx = [names.index(m) for m in members]
    S = StandardScaler().fit_transform(X[:, idx])
    k = min(len(members), 3)
    p = PCA(n_components=k).fit(S)
    ev = p.explained_variance_ratio_

    # does one component predict the target as well as the raw members do?
    c1 = S @ p.components_[0]
    r_pc1 = abs(np.corrcoef(c1, y)[0, 1])
    B, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(S)), S]), y, rcond=None)
    pred = np.column_stack([np.ones(len(S)), S]) @ B
    r_full = abs(np.corrcoef(pred, y)[0, 1])

    combine = ev[0] >= 0.65 and r_pc1 >= r_full - 0.02
    return {
        "members": members,
        "n": len(members),
        "pc1_explained": float(ev[0]),
        "pc2_explained": float(ev[1]) if len(ev) > 1 else 0.0,
        "r_pc1_vs_target": float(r_pc1),
        "r_allmembers_vs_target": float(r_full),
        "loadings_pc1": {m: float(l) for m, l in zip(members, p.components_[0])},
        "recommendation": "combine" if combine else "keep separate",
        "why": ("one component carries most of the variance and predicts as well as "
                "the members together" if combine else
                ("more than one distinct ability here" if ev[0] < 0.65 else
                 "collapsing would cost predictive power")),
    }


def main():
    df, names = load()
    X = df[names].to_numpy(float)
    y = df.adj_win_pct.to_numpy(float)
    print(f"features: {len(names)}   team-seasons: {len(df)}   "
          f"VIF threshold: {VIF_THRESHOLD}\n")

    # ---- 1. correlation ------------------------------------------------------
    C = df[names].corr()
    off = C.where(~np.eye(len(C), dtype=bool))
    pairs = (off.abs().stack().sort_values(ascending=False)
                .iloc[::2].head(15))
    print("=" * 74)
    print("1. MOST CORRELATED PAIRS")
    print("=" * 74)
    for (a, b), v in pairs.items():
        print(f"  {v:+.3f}  {a:<30} {b}")
    print(f"\n  pairs above |r| 0.70: "
          f"{int((off.abs() >= 0.70).sum().sum() // 2)}")

    # ---- 2. VIF --------------------------------------------------------------
    V = vif_table(X, names)
    flagged = V[V.vif > VIF_THRESHOLD]
    print("\n" + "=" * 74)
    print(f"2. VARIANCE INFLATION — {len(flagged)} of {len(names)} features "
          f"above {VIF_THRESHOLD}")
    print("=" * 74)
    print(V.head(18).round(2).to_string(index=False))

    # ---- 3. clusters ---------------------------------------------------------
    cl = clusters(C, names)
    multi = {k: v for k, v in cl.items() if len(v) > 1}
    print("\n" + "=" * 74)
    print(f"3. CORRELATED GROUPS — {len(multi)} groups of 2+, "
          f"{len(cl) - len(multi)} features standing alone")
    print("=" * 74)

    # ---- 4. factor structure per group + recommendation ----------------------
    recs = {}
    for k, members in multi.items():
        r = factor_report(X, names, members, y)
        recs[k] = r
        print(f"\n  group of {r['n']}: {', '.join(members)}")
        print(f"    first component explains {r['pc1_explained']*100:.0f}% "
              f"(second {r['pc2_explained']*100:.0f}%)")
        print(f"    predicts target: component alone {r['r_pc1_vs_target']:.3f}  "
              f"vs all members {r['r_allmembers_vs_target']:.3f}")
        print(f"    -> {r['recommendation'].upper()}: {r['why']}")

    # ---- 5. whole-set PCA ----------------------------------------------------
    S = StandardScaler().fit_transform(X)
    p = PCA().fit(S)
    cum = np.cumsum(p.explained_variance_ratio_)
    n80 = int(np.searchsorted(cum, 0.80) + 1)
    n95 = int(np.searchsorted(cum, 0.95) + 1)
    print("\n" + "=" * 74)
    print("5. PCA ON THE WHOLE SET")
    print("=" * 74)
    print(f"  {len(names)} features -> {n80} components carry 80% of the variance, "
          f"{n95} carry 95%")
    print("  first six components, heaviest loadings:")
    fa = FactorAnalysis(n_components=6, random_state=0).fit(S)
    for i in range(6):
        L = pd.Series(p.components_[i], index=names)
        top = L.abs().sort_values(ascending=False).head(5).index
        print(f"    PC{i+1} ({p.explained_variance_ratio_[i]*100:4.1f}%): "
              + ", ".join(f"{t}{'+' if L[t] > 0 else '-'}" for t in top))

    out = {
        "vif_threshold": VIF_THRESHOLD,
        "n_features": len(names),
        "n_team_seasons": int(len(df)),
        "correlation": {a: {b: round(float(C.loc[a, b]), 3) for b in names}
                        for a in names},
        "vif": json.loads(V.round(3).to_json(orient="records")),
        "clusters": {str(k): v for k, v in cl.items()},
        "group_recommendations": {str(k): v for k, v in recs.items()},
        "pca": {
            "explained_variance_ratio": [round(float(v), 5)
                                         for v in p.explained_variance_ratio_],
            "n_components_80pct": n80, "n_components_95pct": n95,
            "loadings": {f"PC{i+1}": {n: round(float(v), 4)
                                      for n, v in zip(names, p.components_[i])}
                         for i in range(min(10, len(names)))},
        },
        "factor_analysis_loadings": {
            f"F{i+1}": {n: round(float(v), 4)
                        for n, v in zip(names, fa.components_[i])}
            for i in range(6)},
    }
    json.dump(out, open(f"{HERE}/collinearity.json", "w"), indent=1)
    print(f"\n-> collinearity.json")


if __name__ == "__main__":
    main()
