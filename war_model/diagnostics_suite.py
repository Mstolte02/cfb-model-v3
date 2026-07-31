"""Stability, sensitivity and importance for the facet weights.

Three separate jobs that all answer "how much should we believe this?":

  stability    repeated k-fold, refitting the weights each time, so every coefficient
               gets a mean, a standard deviation and a 95% interval. A coefficient
               whose sign flips across folds is not a football finding, it is noise
               with a name.

  sensitivity  the assumptions nobody fitted. Replacement level is set by hand at 15%;
               normalization is a within-season z-score by convention; features are in
               because they were chosen. Each gets moved and the damage measured.

  importance   ridge coefficients, permutation importance and SHAP, side by side.
               They disagree when a feature is collinear with others - the coefficient
               splits, permutation does not notice the loss because a twin covers for
               it - and those disagreements are the interesting part.

The model-comparison margin gets a bootstrap here too, because a 1% RMSE improvement
on 1,156 rows needs an interval before it counts as an improvement.

Run: ./rbenv/bin/python diagnostics_suite.py
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

from model_lab import (load_frames, drop_inert, cluster_pick, enet_select,
                       fit_ridge, model_A, model_B, model_C, PCA_VARIANCE)

HERE = os.path.dirname(os.path.abspath(__file__))
REPEATS, FOLDS = 8, 5
REPLACEMENT_LEVELS = [0.10, 0.125, 0.15, 0.175, 0.20]


# ----------------------------------------------------------------- stability
def coefficient_stability(X, y, names):
    """Repeated k-fold ridge; per-coefficient mean, sd and 95% interval."""
    rkf = RepeatedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=0)
    coefs = []
    for tr, _ in rkf.split(X):
        sc = StandardScaler().fit(X[tr])
        m = fit_ridge(sc.transform(X[tr]), y[tr])
        coefs.append(m.coef_)
    C = np.vstack(coefs)
    out = pd.DataFrame({
        "feature": names,
        "mean": C.mean(0),
        "sd": C.std(0),
        "lo95": np.percentile(C, 2.5, axis=0),
        "hi95": np.percentile(C, 97.5, axis=0),
        "pct_positive": (C > 0).mean(0),
    })
    out["sign_stable"] = (out.pct_positive >= 0.95) | (out.pct_positive <= 0.05)
    # |mean| / sd: below 1 the interval straddles zero comfortably
    out["t_like"] = out["mean"].abs() / out.sd.replace(0, np.nan)
    return out.sort_values("mean", key=abs, ascending=False)


# --------------------------------------------------------------- sensitivity
def replacement_sensitivity(levels=REPLACEMENT_LEVELS):
    """Replacement level changes every WAR figure but should not change the order.

    WAA is untouched by it - the level only sets the size of the pool handed out per
    snap - so the test is whether the RANKING moves, and by how much at the top.
    """
    fv = pd.read_parquet(f"{HERE}/hybrid_facet_war.parquet")
    recs = pd.read_csv(f"{HERE}/records.csv")
    w = pd.read_csv(f"{HERE}/hybrid_facet_weights.csv", index_col=0)["rf"]
    key = "player_id" if "player_id" in fv.columns else "uid"

    base_rank = None
    rows = []
    for lv in levels:
        pool_by_season = {}
        for s, r in recs.groupby("season"):
            pool_by_season[s] = float(((0.5 - lv) * r.fbs_games).sum())
        league_snaps = fv.groupby(["season", "facet"]).snaps.sum()
        per_snap = {}
        for (s, f), snaps in league_snaps.items():
            if snaps > 0 and f in w.index:
                per_snap[(s, f)] = pool_by_season[s] * w[f] / snaps
        credit = np.array([per_snap.get((s, f), 0.0)
                           for s, f in zip(fv.season, fv.facet)]) * fv.snaps.to_numpy()
        war = fv.waa.to_numpy() + credit
        d = pd.DataFrame({"season": fv.season, "id": fv[key], "war": war})
        tot = d.groupby(["season", "id"]).war.sum()
        rk = tot.groupby(level="season").rank(ascending=False)
        if base_rank is None:
            base_rank, base_tot = rk, tot
            rows.append({"replacement": lv, "mean_war": float(tot.mean()),
                         "total_war": float(tot.sum()),
                         "spearman_vs_15pct": 1.0, "mean_rank_shift": 0.0,
                         "top50_overlap": 1.0})
            continue
        j = pd.concat([base_rank.rename("a"), rk.rename("b")], axis=1).dropna()
        top_a = set(base_tot.groupby(level="season").nlargest(50).index.get_level_values(-1))
        top_b = set(tot.groupby(level="season").nlargest(50).index.get_level_values(-1))
        rows.append({
            "replacement": lv, "mean_war": float(tot.mean()),
            "total_war": float(tot.sum()),
            "spearman_vs_15pct": float(j.a.corr(j.b, method="spearman")),
            "mean_rank_shift": float((j.a - j.b).abs().mean()),
            "top50_overlap": len(top_a & top_b) / max(1, len(top_a)),
        })
    return pd.DataFrame(rows)


def normalization_sensitivity(df, cand, y, seasons):
    """Within-season z is a choice. Compare it to rolling and global baselines.

    Within-season removes any league-wide drift, which is why it is the default, but
    it also erases genuine year-to-year movement in the whole sport - if passing got
    better everywhere, a within-season z cannot see it.
    """
    raw = pd.read_csv(f"{HERE}/candidate_team_totals.csv", index_col=[0, 1])
    raw = raw.reindex(df.index)[cand]
    out = {}

    def score(Z):
        X = Z.to_numpy(float)
        P = np.full(len(y), np.nan)
        for s in sorted(set(seasons)):
            tr, te = seasons != s, seasons == s
            if tr.sum() < 200:
                continue
            keep = drop_inert(X[tr], cand)
            keep = cluster_pick(X[tr], y[tr], cand, keep)
            keep = enet_select(X[tr], y[tr], keep)
            m = fit_ridge(X[tr][:, keep], y[tr])
            P[te] = m.predict(X[te][:, keep])
        ok = ~np.isnan(P)
        return float(np.sqrt(mean_squared_error(y[ok], P[ok]))), \
               float(np.corrcoef(P[ok], y[ok])[0, 1])

    out["within season (ships)"] = score(
        raw.groupby(level="season").transform(lambda c: (c - c.mean()) / c.std(ddof=0)).fillna(0.0))

    # rolling: standardize against the three seasons up to and including this one, so
    # the baseline moves with the sport but never uses the future
    piece = []
    for s in sorted(raw.index.get_level_values("season").unique()):
        hist = raw[raw.index.get_level_values("season").isin(range(s - 2, s + 1))]
        cur = raw.xs(s, level="season", drop_level=False)
        piece.append((cur - hist.mean()) / hist.std(ddof=0))
    out["rolling 3-season"] = score(pd.concat(piece).reindex(raw.index).fillna(0.0))

    out["global (all seasons)"] = score(
        ((raw - raw.mean()) / raw.std(ddof=0)).fillna(0.0))
    return pd.DataFrame([{"scheme": k, "rmse": v[0], "r": v[1]}
                         for k, v in out.items()])


def leave_one_feature_out(X, y, names, seasons, top=15):
    """Drop each feature, refit, and measure both prediction loss and coefficient drift."""
    sc = StandardScaler().fit(X)
    base_m = fit_ridge(sc.transform(X), y)
    base_coef = pd.Series(base_m.coef_, index=names)

    def cv_rmse(cols):
        P = np.full(len(y), np.nan)
        for s in sorted(set(seasons)):
            tr, te = seasons != s, seasons == s
            if tr.sum() < 200:
                continue
            m = fit_ridge(X[tr][:, cols], y[tr])
            P[te] = m.predict(X[te][:, cols])
        ok = ~np.isnan(P)
        return float(np.sqrt(mean_squared_error(y[ok], P[ok])))

    base_rmse = cv_rmse(list(range(len(names))))
    order = base_coef.abs().sort_values(ascending=False).index[:top]
    rows = []
    for f in order:
        j = names.index(f)
        cols = [i for i in range(len(names)) if i != j]
        r = cv_rmse(cols)
        m = fit_ridge(StandardScaler().fit_transform(X[:, cols]), y)
        c2 = pd.Series(m.coef_, index=[names[i] for i in cols])
        drift = (c2 - base_coef.drop(f)).abs()
        rows.append({"dropped": f, "rmse": r, "delta_rmse": r - base_rmse,
                     "max_coef_drift": float(drift.max()),
                     "drifted_most": str(drift.idxmax()),
                     "mean_coef_drift": float(drift.mean())})
    return base_rmse, pd.DataFrame(rows).sort_values("delta_rmse", ascending=False)


# ---------------------------------------------------------------- importance
def importance_audit(X, y, names):
    """Ridge coefficient, permutation importance and SHAP, normalized to compare."""
    sc = StandardScaler().fit(X)
    S = sc.transform(X)
    m = fit_ridge(S, y)
    coef = pd.Series(np.abs(m.coef_), index=names)

    perm = permutation_importance(m, S, y, n_repeats=15, random_state=0, n_jobs=-1)
    pim = pd.Series(np.clip(perm.importances_mean, 0, None), index=names)

    shap_vals = None
    try:
        import shap
        ex = shap.LinearExplainer(m, S)
        shap_vals = pd.Series(np.abs(ex.shap_values(S)).mean(0), index=names)
    except Exception as e:
        print(f"  [info] SHAP unavailable ({type(e).__name__}); "
              "reporting ridge + permutation only")

    def norm(s):
        return s / s.sum() if s is not None and s.sum() else s

    out = pd.DataFrame({"ridge": norm(coef), "permutation": norm(pim)})
    if shap_vals is not None:
        out["shap"] = norm(shap_vals)
    out["rank_ridge"] = out.ridge.rank(ascending=False)
    out["rank_perm"] = out.permutation.rank(ascending=False)
    out["rank_gap"] = (out.rank_ridge - out.rank_perm).abs()
    return out.sort_values("ridge", ascending=False)


# ------------------------------------------------- is the comparison margin real?
def bootstrap_margin(df, cand, bench, y, seasons, n=200):
    """Bootstrap the RMSE difference between the benchmark and the two best models."""
    X, B = df[cand].to_numpy(float), df[bench].to_numpy(float)
    preds = {}
    for label, fn in (("A", model_A), ("B", model_B), ("C", model_C)):
        P = np.full(len(y), np.nan)
        for s in sorted(set(seasons)):
            tr, te = seasons != s, seasons == s
            if tr.sum() < 200:
                continue
            p, _ = fn(X[tr], y[tr], X[te], B[tr], B[te], cand)
            P[te] = p
        preds[label] = P
    ok = ~np.isnan(preds["A"])
    rng = np.random.default_rng(0)
    idx = np.where(ok)[0]
    rows = []
    for label in ("B", "C"):
        d = []
        for _ in range(n):
            s = rng.choice(idx, size=len(idx), replace=True)
            ra = np.sqrt(mean_squared_error(y[s], preds["A"][s]))
            rb = np.sqrt(mean_squared_error(y[s], preds[label][s]))
            d.append(rb - ra)
        d = np.array(d)
        rows.append({"model": label, "mean_delta_rmse": float(d.mean()),
                     "lo95": float(np.percentile(d, 2.5)),
                     "hi95": float(np.percentile(d, 97.5)),
                     "pct_better_than_A": float((d < 0).mean())})
    return pd.DataFrame(rows)


def main():
    df, cand, bench = load_frames()
    X = df[cand].to_numpy(float)
    y = df.y.to_numpy(float)
    seasons = df.index.get_level_values("season").to_numpy()
    print(f"features {len(cand)}   team-seasons {len(df)}\n")

    print("=" * 78)
    print(f"1. COEFFICIENT STABILITY — {REPEATS}x{FOLDS}-fold, weights refitted each time")
    print("=" * 78)
    st = coefficient_stability(X, y, cand)
    print(st.head(18).round(4).to_string(index=False))
    unstable = st[~st.sign_stable]
    print(f"\n  sign flips across folds: {len(unstable)} of {len(cand)} features")
    if len(unstable):
        print("  worst offenders (largest |mean| with an unstable sign):")
        for _, r in unstable.head(8).iterrows():
            print(f"    {r.feature:<30} mean {r['mean']:+.4f}  sd {r.sd:.4f}  "
                  f"positive in {r.pct_positive*100:.0f}% of folds")

    print("\n" + "=" * 78)
    print("2. IMPORTANCE — three measures, disagreements highlighted")
    print("=" * 78)
    imp = importance_audit(X, y, cand)
    print(imp.head(15).round(4).to_string())
    big = imp[imp.rank_gap >= 15]
    print(f"\n  features ranked 15+ places apart by ridge vs permutation: {len(big)}")
    for f, r in big.head(6).iterrows():
        print(f"    {f:<30} ridge #{r.rank_ridge:.0f}  permutation #{r.rank_perm:.0f}")

    print("\n" + "=" * 78)
    print("3. REPLACEMENT LEVEL")
    print("=" * 78)
    rs = replacement_sensitivity()
    print(rs.round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("4. NORMALIZATION BASELINE")
    print("=" * 78)
    ns = normalization_sensitivity(df, cand, y, seasons)
    print(ns.round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("5. LEAVE-ONE-FEATURE-OUT")
    print("=" * 78)
    base_rmse, lofo = leave_one_feature_out(X, y, cand, seasons)
    print(f"  all features: rmse {base_rmse:.4f}")
    print(lofo.round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("6. IS THE MODEL-COMPARISON MARGIN REAL? (bootstrap, 200 resamples)")
    print("=" * 78)
    bm = bootstrap_margin(df, cand, bench, y, seasons)
    print(bm.round(4).to_string(index=False))
    for _, r in bm.iterrows():
        verdict = ("beats the benchmark" if r.hi95 < 0 else
                   "cannot be distinguished from the benchmark")
        print(f"  model {r.model}: {verdict} "
              f"(delta {r.mean_delta_rmse:+.4f}, 95% [{r.lo95:+.4f}, {r.hi95:+.4f}], "
              f"better in {r.pct_better_than_A*100:.0f}% of resamples)")

    json.dump({
        "stability": json.loads(st.round(5).to_json(orient="records")),
        "n_sign_flips": int(len(unstable)),
        "importance": json.loads(imp.round(5).reset_index()
                                 .rename(columns={"index": "feature"})
                                 .to_json(orient="records")),
        "replacement": json.loads(rs.round(5).to_json(orient="records")),
        "normalization": json.loads(ns.round(5).to_json(orient="records")),
        "leave_one_out": json.loads(lofo.round(5).to_json(orient="records")),
        "lofo_base_rmse": base_rmse,
        "bootstrap_margin": json.loads(bm.round(5).to_json(orient="records")),
        "repeats": REPEATS, "folds": FOLDS,
    }, open(f"{HERE}/diagnostics_suite.json", "w"), indent=1)
    print("\n-> diagnostics_suite.json")


if __name__ == "__main__":
    main()
