"""Does a six-concept model beat the hand-built facets, and is the margin real?

Forward selection by concept stops paying after six: passing, run defense, rushing,
coverage, run blocking, pass rush. Those six reach rmse .1638 where all fifteen
concepts (86 features) reach .1635 - so nine concepts and 48 features are buying
.0003. That is a simpler model than the one that ships, not a more complex one, which
makes it worth testing properly rather than as a footnote.

Bootstraps three contrasts against the benchmark:
  six concepts (38 raw features)
  six concepts, one PCA score each (6 features)
  all fifteen concepts (86 features)

Run: ./rbenv/bin/python concept_verdict.py
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from concepts import CONCEPTS, concept_of, concept_pca, load

HERE = os.path.dirname(os.path.abspath(__file__))
SIX = ["passing", "run_defense", "rushing", "coverage", "run_blocking", "pass_rush"]
N_BOOT = 400


def oof(X, y, seasons):
    """Out-of-fold predictions, season-blocked."""
    P = np.full(len(y), np.nan)
    for s in sorted(set(seasons)):
        tr, te = seasons != s, seasons == s
        if tr.sum() < 200:
            continue
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(-3, 4, 60)).fit(sc.transform(X[tr]), y[tr])
        P[te] = m.predict(sc.transform(X[te]))
    return P


def main():
    fv, df, names = load()
    y = df.y.to_numpy(float)
    seasons = df.index.get_level_values("season").to_numpy()
    Z = df[names]

    # the benchmark: the hand-built facets that ship, on the same rows
    Zb = pd.read_csv(f"{HERE}/hybrid_team_facet_totals.csv", index_col=[0, 1])
    Zb = Zb.groupby(level="season").transform(
        lambda c: (c - c.mean()) / c.std(ddof=0)).fillna(0.0)
    Zb = Zb.reindex(df.index).fillna(0.0)

    six_cols = [n for n in names if concept_of(n) in set(SIX)]
    C1, _ = concept_pca(Z, names, Z.index, n_components=1)
    six_pca = C1[[c for c in C1.columns if c in SIX]]

    variants = {
        "benchmark (hand-built facets)": Zb.to_numpy(float),
        "six concepts, raw features":    Z[six_cols].to_numpy(float),
        "six concepts, one score each":  six_pca.to_numpy(float),
        "all fifteen concepts":          Z.to_numpy(float),
    }

    preds, rmses = {}, {}
    for label, X in variants.items():
        P = oof(X, y, seasons)
        ok = ~np.isnan(P)
        preds[label] = P
        rmses[label] = float(np.sqrt(mean_squared_error(y[ok], P[ok])))
        print(f"  {label:<32} {X.shape[1]:>3} features   rmse {rmses[label]:.4f}")

    base_label = "benchmark (hand-built facets)"
    ok = ~np.isnan(preds[base_label])
    idx = np.where(ok)[0]
    rng = np.random.default_rng(0)

    print(f"\n{'contrast':<32}{'ΔRMSE':>9}{'95% low':>10}{'95% high':>10}"
          f"{'better in':>12}")
    print("-" * 75)
    rows = []
    for label in variants:
        if label == base_label:
            continue
        d = []
        for _ in range(N_BOOT):
            s = rng.choice(idx, size=len(idx), replace=True)
            d.append(np.sqrt(mean_squared_error(y[s], preds[label][s]))
                     - np.sqrt(mean_squared_error(y[s], preds[base_label][s])))
        d = np.array(d)
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append({"variant": label, "n_features": variants[label].shape[1],
                     "rmse": rmses[label], "mean_delta": float(d.mean()),
                     "lo95": float(lo), "hi95": float(hi),
                     "pct_better": float((d < 0).mean()),
                     "significant": bool(hi < 0)})
        print(f"{label:<32}{d.mean():>+9.4f}{lo:>+10.4f}{hi:>+10.4f}"
              f"{(d < 0).mean()*100:>11.0f}%")

    print()
    for r in rows:
        verdict = ("BEATS the benchmark" if r["significant"]
                   else "not distinguishable from the benchmark")
        print(f"  {r['variant']} ({r['n_features']} features): {verdict}")

    json.dump({"rmse": rmses, "bootstrap": rows, "six_concepts": SIX,
               "n_boot": N_BOOT}, open(f"{HERE}/concept_verdict.json", "w"), indent=1)
    print("\n-> concept_verdict.json")


if __name__ == "__main__":
    main()
