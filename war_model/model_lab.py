"""Six ways to turn football skills into a team rating, scored on identical splits.

The benchmark is the model that ships: twenty hand-picked facets, ridge-weighted
against the following season. Everything else has to beat that to justify itself.

  A  current      the hand-built facets that ship (20 PFF + 12 CFBD),
                  ridge on next season
  B  selected     86 candidates -> variance filter -> correlation clustering ->
                  elastic net -> ridge refit on survivors
  C  pca          86 candidates -> components carrying 90% of variance -> ridge
  D  interactions selected features plus CV-gated interaction terms -> ridge
  E  elasticnet   86 candidates straight into elastic net, no separate refit
  F  boosting     86 candidates into gradient boosting

THE TARGET IS NEXT SEASON. Fitting to the season being described is what made
coverage outrank pass rush, so every model here is scored on its ability to predict
the following year's adjusted win percentage, out of sample, season-blocked.

SELECTION HAPPENS INSIDE EACH FOLD. Choosing features on all the data and then
cross-validating the survivors leaks the test seasons into the feature set and
flatters every model that does selection. The filter, the clustering and the elastic
net all refit per fold; only the held-out season's rows are ever unseen.

Run: ./rbenv/bin/python model_lab.py
"""
import json, os, time
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
NEAR_ZERO_VAR = 0.01      # sd on the standardized scale; below this a column is inert
CLUSTER_CUT = 0.30        # 1 - |r|, so |r| >= 0.70 counts as one skill
PCA_VARIANCE = 0.90

# Interaction candidates, named as football rather than as column algebra. Each is a
# pair of feature-name patterns; every matching pair of survivors gets multiplied.
INTERACTIONS = [
    ("QB x WR",              r"^QB_",  r"^WR_"),
    ("QB x OL",              r"^QB_",  r"^OL_"),
    ("pass rush x coverage", r"^(ED|DI)_", r"^(CB|S|CB-S)_cover"),
    ("run block x RB",       r"^OL_run", r"^RB_"),
    ("coverage x tackling",  r"^(CB|S|CB-S)_cover", r"_tackle"),
    ("protection x receiving", r"^OL_pass", r"^WR_(pass_route|yprr)"),
]


# --------------------------------------------------------------------- data
def load_frames():
    """Candidate matrix, benchmark matrix, and the next-season target."""
    Zc = pd.read_csv(f"{HERE}/candidate_team_z.csv", index_col=[0, 1])
    Zb = pd.read_csv(f"{HERE}/hybrid_team_facet_totals.csv", index_col=[0, 1])
    # the benchmark ships standardized within season; do the same here
    Zb = Zb.groupby(level="season").transform(
        lambda c: (c - c.mean()) / c.std(ddof=0)).fillna(0.0)
    recs = pd.read_csv(f"{HERE}/records.csv")
    nxt = recs[["season", "team", "adj_win_pct"]].copy()
    nxt["season"] -= 1
    nxt = nxt.rename(columns={"adj_win_pct": "y"}).set_index(["season", "team"])

    df = Zc.join(Zb, how="inner", rsuffix="__bench").join(nxt, how="inner").dropna(subset=["y"])
    cand = [c for c in Zc.columns if c in df.columns]
    bench = [c for c in df.columns if c.endswith("__bench")] or \
            [c for c in Zb.columns if c in df.columns and c not in cand]
    return df, cand, bench


# ------------------------------------------------------- selection primitives
def drop_inert(X, names):
    keep = [i for i, _ in enumerate(names) if X[:, i].std() > NEAR_ZERO_VAR]
    return keep


def cluster_pick(X, y, names, keep):
    """One representative per correlated cluster: the member most correlated with y.

    Clustering before the elastic net matters. Given six views of the same throw the
    net will keep whichever one the fold happens to favour, and the choice flips
    between folds - which is exactly the instability the diagnostics flagged.
    """
    if len(keep) < 3:
        return keep
    C = np.corrcoef(X[:, keep], rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    D = 1 - np.abs(C)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    lab = hierarchy.fcluster(
        hierarchy.linkage(squareform(D, checks=False), method="average"),
        CLUSTER_CUT, "distance")
    out = []
    for c in np.unique(lab):
        members = [keep[i] for i in np.where(lab == c)[0]]
        r = [abs(np.corrcoef(X[:, m], y)[0, 1]) if X[:, m].std() > 0 else 0
             for m in members]
        out.append(members[int(np.argmax(r))])
    return sorted(out)


def enet_select(X, y, keep, seed=0):
    """Elastic net on the cluster representatives; survivors are the non-zero ones."""
    if not keep:
        return keep
    S = StandardScaler().fit_transform(X[:, keep])
    m = ElasticNetCV(l1_ratio=[0.3, 0.5, 0.7, 0.9, 1.0], cv=4, alphas=40,
                     max_iter=20000, random_state=seed).fit(S, y)
    live = [k for k, c in zip(keep, m.coef_) if abs(c) > 1e-8]
    return live or keep


def build_interactions(X, names, keep):
    """CV-gated interaction columns, generated after standardization."""
    import re
    cols, labels = [], []
    for label, pa, pb in INTERACTIONS:
        A = [i for i in keep if re.search(pa, names[i])]
        B = [i for i in keep if re.search(pb, names[i])]
        for a in A:
            for b in B:
                if a == b:
                    continue
                cols.append(X[:, a] * X[:, b])
                labels.append(f"{label}: {names[a]} x {names[b]}")
    if not cols:
        return np.empty((len(X), 0)), []
    return np.column_stack(cols), labels


# ------------------------------------------------------------------- models
def fit_ridge(Xtr, ytr):
    return RidgeCV(alphas=np.logspace(-3, 4, 60)).fit(Xtr, ytr)


def model_A(Xtr, ytr, Xte, bench_tr, bench_te, names):
    m = fit_ridge(bench_tr, ytr)
    return m.predict(bench_te), {"n_features": bench_tr.shape[1]}


def model_B(Xtr, ytr, Xte, *_):
    keep = drop_inert(Xtr, [""] * Xtr.shape[1])
    keep = cluster_pick(Xtr, ytr, [f"f{i}" for i in range(Xtr.shape[1])], keep)
    keep = enet_select(Xtr, ytr, keep)
    m = fit_ridge(Xtr[:, keep], ytr)
    return m.predict(Xte[:, keep]), {"n_features": len(keep), "kept": keep}


def model_C(Xtr, ytr, Xte, *_):
    sc = StandardScaler().fit(Xtr)
    p = PCA(n_components=PCA_VARIANCE, svd_solver="full").fit(sc.transform(Xtr))
    m = fit_ridge(p.transform(sc.transform(Xtr)), ytr)
    return m.predict(p.transform(sc.transform(Xte))), {"n_features": p.n_components_}


def model_D(Xtr, ytr, Xte, bench_tr, bench_te, names):
    keep = drop_inert(Xtr, names)
    keep = cluster_pick(Xtr, ytr, names, keep)
    keep = enet_select(Xtr, ytr, keep)
    Itr, labels = build_interactions(Xtr, names, keep)
    Ite, _ = build_interactions(Xte, names, keep)
    if Itr.shape[1] == 0:
        return model_B(Xtr, ytr, Xte)
    # gate the interactions with their own elastic net so only the ones that pay
    # for themselves survive into the ridge
    A = np.hstack([Xtr[:, keep], Itr])
    live = enet_select(A, ytr, list(range(A.shape[1])))
    B = np.hstack([Xte[:, keep], Ite])
    m = fit_ridge(A[:, live], ytr)
    n_int = sum(1 for i in live if i >= len(keep))
    return m.predict(B[:, live]), {"n_features": len(live), "n_interactions": n_int}


def model_E(Xtr, ytr, Xte, *_):
    sc = StandardScaler().fit(Xtr)
    m = ElasticNetCV(l1_ratio=[0.3, 0.5, 0.7, 0.9, 1.0], cv=4, alphas=40,
                     max_iter=20000, random_state=0).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte)), {
        "n_features": int((np.abs(m.coef_) > 1e-8).sum())}


def model_F(Xtr, ytr, Xte, *_):
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                      max_leaf_nodes=15, min_samples_leaf=25,
                                      l2_regularization=1.0,
                                      random_state=0).fit(Xtr, ytr)
    return m.predict(Xte), {"n_features": Xtr.shape[1]}


MODELS = {
    "A current (hand-built facets)": model_A,
    "B selected ridge":           model_B,
    "C PCA + ridge":              model_C,
    "D interactions + ridge":     model_D,
    "E elastic net":              model_E,
    "F gradient boosting":        model_F,
}


def calibration_slope(pred, actual):
    """Regression of actual on predicted. 1.0 means the spread is right."""
    b = np.polyfit(pred, actual, 1)
    return float(b[0])


def main():
    df, cand, bench = load_frames()
    X = df[cand].to_numpy(float)
    B = df[bench].to_numpy(float)
    y = df.y.to_numpy(float)
    seasons = df.index.get_level_values("season").to_numpy()
    print(f"candidates {len(cand)}   benchmark facets {len(bench)}   "
          f"team-seasons with a following year {len(df)}")
    print(f"seasons: {sorted(set(seasons))}\n")

    rows, preds = [], {}
    for label, fn in MODELS.items():
        t0 = time.time()
        P, A_ = np.full(len(y), np.nan), []
        for s in sorted(set(seasons)):
            tr, te = seasons != s, seasons == s
            if tr.sum() < 200:
                continue
            p, info = fn(X[tr], y[tr], X[te], B[tr], B[te], cand)
            P[te] = p
            A_.append(info.get("n_features", np.nan))
        ok = ~np.isnan(P)
        rows.append({
            "model": label,
            "features": float(np.nanmean(A_)),
            "rmse": float(np.sqrt(mean_squared_error(y[ok], P[ok]))),
            "mae": float(mean_absolute_error(y[ok], P[ok])),
            "r2": float(r2_score(y[ok], P[ok])),
            "r": float(np.corrcoef(P[ok], y[ok])[0, 1]),
            "calib": calibration_slope(P[ok], y[ok]),
            "seconds": round(time.time() - t0, 1),
        })
        preds[label] = P
        print(f"  {label:<28} rmse {rows[-1]['rmse']:.4f}  r {rows[-1]['r']:.3f}  "
              f"feat {rows[-1]['features']:.0f}  {rows[-1]['seconds']:.0f}s", flush=True)

    r = pd.DataFrame(rows)
    base = float(r[r.model.str.startswith("A")].rmse.iloc[0])
    r["vs_benchmark"] = (r.rmse - base).round(5)
    print("\n" + "=" * 92)
    print("MODEL COMPARISON — predicting NEXT season's adjusted win pct, season-blocked")
    print("=" * 92)
    print(r[["model", "features", "rmse", "mae", "r2", "r", "calib", "seconds",
             "vs_benchmark"]].round(4).to_string(index=False))
    best = r.loc[r.rmse.idxmin()]
    print(f"\nbest: {best.model}  (rmse {best.rmse:.4f} vs benchmark {base:.4f}, "
          f"{best.vs_benchmark:+.5f})")
    if best.vs_benchmark > -0.0005:
        print("  -> no model beats the current methodology by a material margin")

    r.to_csv(f"{HERE}/model_comparison.csv", index=False)
    json.dump({"rows": json.loads(r.round(5).to_json(orient="records")),
               "benchmark_rmse": base,
               "n_candidates": len(cand), "n_benchmark": len(bench),
               "n_team_seasons": int(len(df))},
              open(f"{HERE}/model_comparison.json", "w"), indent=1)
    print("\n-> model_comparison.csv / .json")


if __name__ == "__main__":
    main()
