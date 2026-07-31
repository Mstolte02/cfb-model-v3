"""Should facet weights be fitted to THIS season's wins, or next season's?

The build fits facet weights by random-forest importance against the same season's
adjusted win percentage. That is the right target for describing a season and the
wrong one for projecting the next, and the audit shows why:

  facet     weight   same-season r   year-over-year stability
  cov_db     .234        +.552              .213
  prsh_dl    .029        +.384              .567

Coverage grade explains the season it happened in better than pass rush does, and
repeats less than half as well. The likely reason is game script - a defense that is
ahead forces obvious passing downs, and PFF coverage grades reward that - which makes
coverage partly an *outcome* of winning rather than a cause of it. Fitting weights on
same-season fit therefore loads the model onto the facets most contaminated by the
result, which is also why the ratings feel like an echo of last season.

This tests four weighting targets and scores them on what the model is actually for:
predicting the following season.

Run: ./rbenv/bin/python weight_target_test.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict

from build_hybrid import unified_facets

PASS_RUSH = ["prsh_dl", "prsh_sec", "cfbd_havoc_dl", "cfbd_havoc_lb", "cfbd_havoc_db"]
COVERAGE = ["cov_db", "cov_lb", "cfbd_cov_db"]


def build():
    fv = unified_facets()
    names = sorted(fv.facet.unique())
    tot = (fv.groupby(["season", "team", "facet"], as_index=False)["value"].sum()
             .pivot(index=["season", "team"], columns="facet", values="value")
             .reindex(columns=names).fillna(0.0))
    Z = tot.groupby(level="season").transform(
        lambda c: (c - c.mean()) / c.std(ddof=0)).fillna(0.0)
    recs = pd.read_csv("records.csv")
    df = Z.join(recs.set_index(["season", "team"]), how="inner").reset_index()

    # next season's result for the same team, which is the projection target
    nxt = recs[["season", "team", "adj_win_pct"]].copy()
    nxt["season"] -= 1
    df = df.merge(nxt.rename(columns={"adj_win_pct": "next_win_pct"}),
                  on=["season", "team"], how="left")
    return df, names


def weights_from(df, names, target, how):
    d = df.dropna(subset=[target])
    X, y = d[names].to_numpy(float), d[target].to_numpy(float)
    if how == "rf":
        m = RandomForestRegressor(n_estimators=600, min_samples_leaf=3,
                                  random_state=0, n_jobs=-1).fit(X, y)
        w = pd.Series(m.feature_importances_, index=names)
    else:
        m = RidgeCV(alphas=np.logspace(-2, 3, 40)).fit(X, y)
        w = pd.Series(m.coef_, index=names).clip(lower=0)
    return w / w.sum()


def score(df, names, w):
    """Team rating from these weights, scored on THIS and NEXT season's wins."""
    f = df[names].to_numpy(float) @ w.reindex(names).to_numpy()
    d = df.assign(f=f)
    now = np.corrcoef(d.f, d.adj_win_pct)[0, 1]
    nx = d.dropna(subset=["next_win_pct"])
    fut = np.corrcoef(nx.f, nx.next_win_pct)[0, 1]
    # season-blocked CV of the forward target, the honest version
    g = nx.season.to_numpy()
    m = RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                              random_state=0, n_jobs=-1)
    p = cross_val_predict(m, nx[names].to_numpy(float),
                          nx.next_win_pct.to_numpy(float),
                          cv=GroupKFold(n_splits=5), groups=g)
    cvf = np.corrcoef(p, nx.next_win_pct)[0, 1]
    return now, fut, cvf


def main():
    df, names = build()
    print(f"team-seasons: {len(df)}   with a following season: "
          f"{df.next_win_pct.notna().sum()}\n")

    variants = {
        "RF on same season (ships)":  ("adj_win_pct", "rf"),
        "RF on NEXT season":          ("next_win_pct", "rf"),
        "Ridge on same season":       ("adj_win_pct", "ridge"),
        "Ridge on NEXT season":       ("next_win_pct", "ridge"),
    }

    print(f"{'facet weighting':<28}{'r vs this yr':>13}{'r vs next yr':>14}"
          f"{'CV next yr':>12}{'rush%':>8}{'cov%':>7}")
    print("-" * 82)
    W = {}
    for nm, (tgt, how) in variants.items():
        w = weights_from(df, names, tgt, how)
        W[nm] = w
        now, fut, cvf = score(df, names, w)
        pr = w.reindex(PASS_RUSH).fillna(0).sum() * 100
        cv_ = w.reindex(COVERAGE).fillna(0).sum() * 100
        print(f"{nm:<28}{now:>13.3f}{fut:>14.3f}{cvf:>12.3f}{pr:>8.1f}{cv_:>7.1f}")

    print("\ntop 10 facets under each weighting:")
    comp = pd.DataFrame(W)
    print(comp.sort_values("RF on same season (ships)", ascending=False)
          .head(12).round(4).to_string())

    print("\nkey movers, same-season RF -> next-season RF:")
    a = comp["RF on same season (ships)"]
    b = comp["RF on NEXT season"]
    mv = (b - a).sort_values()
    for f in list(mv.index[:5]) + list(mv.index[-5:]):
        print(f"  {f:<18} {a[f]:.4f} -> {b[f]:.4f}   {b[f]-a[f]:+.4f}")


if __name__ == "__main__":
    main()
