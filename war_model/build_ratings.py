"""Stage 2: facet weights, the f vector, PFF Massey ratings, and the rating->wins map."""
import json, os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict

from facets import FACETS, YEARS
from build_massey import massey_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM_MAP = json.load(open(f"{HERE}/team_map.json"))
FACET_NAMES = list(FACETS)


def team_facet_matrix(fv):
    """Team-season totals per facet, standardized within season."""
    fv = fv.copy()
    fv["team"] = fv.team_name.map(TEAM_MAP)
    tot = (fv.groupby(["season", "team", "facet"], as_index=False)["value"].sum()
             .pivot(index=["season", "team"], columns="facet", values="value")
             .reindex(columns=FACET_NAMES).fillna(0.0))
    # standardize each facet within season so weights are on a common scale
    z = tot.groupby(level="season").transform(lambda c: (c - c.mean()) / c.std(ddof=0))
    return tot, z.fillna(0.0)


def main():
    fv = pd.read_parquet(f"{HERE}/facet_values.parquet")
    recs = pd.read_csv(f"{HERE}/records.csv")
    sched = pd.read_csv(f"{HERE}/schedule.csv")

    tot, Z = team_facet_matrix(fv)
    df = Z.join(recs.set_index(["season", "team"]), how="inner")
    print(f"modelling team-seasons: {len(df)}  (dropped {len(Z) - len(df)} without an FBS record)")

    X = df[FACET_NAMES].to_numpy(float)
    y = df.adj_win_pct.to_numpy(float)
    groups = df.index.get_level_values("season").to_numpy()

    # direction check: every facet is oriented higher = better, so a negative
    # marginal correlation would make a positive RF weight the wrong sign
    corr = pd.Series({f: np.corrcoef(df[f], y)[0, 1] for f in FACET_NAMES}).sort_values()
    print("\nfacet correlation with adjusted win pct:")
    for f, c in corr.items():
        flag = "  <-- NEGATIVE" if c < 0 else ""
        print(f"  {f:<12} {c:+.3f}{flag}")

    # ---- facet weights: RF variable importance (paper) vs ridge (comparison)
    rf = RandomForestRegressor(n_estimators=800, min_samples_leaf=3, random_state=0, n_jobs=-1)
    rf.fit(X, y)
    w_rf = pd.Series(rf.feature_importances_, index=FACET_NAMES)
    w_rf = w_rf / w_rf.sum()

    ridge = RidgeCV(alphas=np.logspace(-2, 3, 40)).fit(X, y)
    w_ridge = pd.Series(ridge.coef_, index=FACET_NAMES)
    w_ridge = w_ridge.clip(lower=0)
    w_ridge = w_ridge / w_ridge.sum()

    cv = GroupKFold(n_splits=5)
    for nm, mdl in (("random forest", rf), ("ridge", ridge)):
        p = cross_val_predict(mdl, X, y, cv=cv, groups=groups)
        print(f"\nseason-blocked CV r (facets -> adj win pct), {nm}: {np.corrcoef(p, y)[0,1]:.3f}")

    weights = pd.DataFrame({"rf": w_rf, "ridge": w_ridge}).sort_values("rf", ascending=False)
    print("\nfacet weights:")
    print(weights.round(4).to_string())

    # ---- build f and solve the Massey system per season
    ratings, inverses = [], {}
    for season in YEARS:
        zs = Z.xs(season, level="season")
        teams = [t for t in zs.index if (season, t) in recs.set_index(["season", "team"]).index]
        teams = sorted(teams)
        zs = zs.loc[teams]
        f = (zs[FACET_NAMES].to_numpy(float) @ w_rf.to_numpy())
        f = f - f.mean()  # columns of M sum to zero, so f must too
        M, idx = massey_matrix(sched, season, teams)
        A = M.copy(); A[-1, :] = 1.0
        b = f.copy(); b[-1] = 0.0
        r = np.linalg.solve(A, b)
        Ainv = np.linalg.inv(A)
        inverses[season] = (Ainv, teams)
        ratings.append(pd.DataFrame({"season": season, "team": teams, "f": f, "massey": r}))
    ratings = pd.concat(ratings, ignore_index=True)

    out = ratings.merge(recs, on=["season", "team"])
    print("\nMassey rating vs adjusted win pct:")
    for s in YEARS:
        d = out[out.season == s]
        print(f"  {s}: r = {np.corrcoef(d.massey, d.adj_win_pct)[0,1]:.3f}  (n={len(d)})")
    print(f"  all: r = {np.corrcoef(out.massey, out.adj_win_pct)[0,1]:.3f}")

    # ---- rating -> implied wins (per-game rate, so it scales to any schedule)
    slope, intercept = np.polyfit(out.massey, out.adj_win_pct, 1)
    print(f"\nimplied win pct = {slope:.4f} * massey + {intercept:.4f}")

    # paper Table 1: stability and predictive power of the ratings
    nxt = out[["season", "team", "massey", "adj_win_pct"]].copy()
    nxt["season"] -= 1
    j = out.merge(nxt, on=["season", "team"], suffixes=("", "_n1"))
    print("\npaper Table 1 analogue (PFF Massey total):")
    print(f"  Cor(rating, rating year n+1):   {np.corrcoef(j.massey, j.massey_n1)[0,1]:.2f}")
    print(f"  Cor(rating, win pct year n):    {np.corrcoef(out.massey, out.adj_win_pct)[0,1]:.2f}")
    print(f"  Cor(rating, win pct year n+1):  {np.corrcoef(j.massey, j.adj_win_pct_n1)[0,1]:.2f}")

    weights.to_csv(f"{HERE}/facet_weights.csv")
    out.to_csv(f"{HERE}/team_ratings.csv", index=False)
    tot.to_csv(f"{HERE}/team_facet_totals.csv")
    Z.to_csv(f"{HERE}/team_facet_z.csv")
    json.dump({"slope": float(slope), "intercept": float(intercept)},
              open(f"{HERE}/wins_map.json", "w"), indent=1)
    print("\nstage 2 written")


if __name__ == "__main__":
    main()
