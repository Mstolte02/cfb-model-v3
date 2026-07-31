"""Does CFBD data predict team wins better than PFF grades? And where?

Three questions, in order:
  1. head to head - PFF facet set vs CFBD facet set vs the union, same estimator,
     same season-blocked CV, same target.
  2. per position - for each position group, swap just that group's facets from one
     source to the other and see which side of the swap predicts better.
  3. next season - the harder and more honest test, since a WAR model is mostly used
     to project forward. Fit on facets from season n, predict adjusted win pct in n+1.

Everything is evaluated with GroupKFold on season, so no fold ever sees a team-season
from a season it was trained on.
"""
import json, os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict

from facets import FACETS, YEARS
from cfbd_facets import CFBD_FACETS

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM_MAP = json.load(open(f"{HERE}/team_map.json"))

# which position group each facet speaks for, so the per-position swap can be scoped
FACET_GROUP = {
    "pass_qb": "QB", "run_qb": "QB",
    "run_rb": "RB", "recv_rb": "RB",
    "recv_wr": "WR", "recv_te": "TE",
    "pblk_ol": "OL", "rblk_ol": "OL", "pblk_skill": "OL", "rblk_skill": "OL",
    "cov_db": "DB", "rdef_sec": "DB", "prsh_sec": "DB", "tackle_db": "DB",
    "havoc_db": "DB", "cov_lb": "LB", "tackle_lb": "LB", "havoc_lb": "LB",
    "prsh_dl": "DL", "rdef_dl": "DL", "havoc_dl": "DL",
    "tackle": "DL", "pen_off": "OL", "pen_def": "DL", "fumble": "RB",
}


def standardize(tot):
    """Z within season, so weights across facets are on a common scale."""
    return tot.groupby(level="season").transform(
        lambda c: (c - c.mean()) / c.std(ddof=0)).fillna(0.0)


def pff_matrix():
    fv = pd.read_parquet(f"{HERE}/facet_values.parquet")
    fv["team"] = fv.team_name.map(TEAM_MAP)
    tot = (fv.groupby(["season", "team", "facet"], as_index=False)["value"].sum()
             .pivot(index=["season", "team"], columns="facet", values="value")
             .reindex(columns=list(FACETS)).fillna(0.0))
    return standardize(tot).add_prefix("pff_")


def cfbd_matrix():
    fv = pd.read_parquet(f"{HERE}/cfbd_facet_values.parquet")
    tot = (fv.rename(columns={"cfbd_team": "team"})
             .groupby(["season", "team", "facet"], as_index=False)["value"].sum()
             .pivot(index=["season", "team"], columns="facet", values="value")
             .reindex(columns=list(CFBD_FACETS)).fillna(0.0))
    return standardize(tot).add_prefix("cfbd_")


def rf():
    return RandomForestRegressor(n_estimators=800, min_samples_leaf=3,
                                 random_state=0, n_jobs=-1)


def cv_r(X, y, groups):
    """Leave-a-season-out CV. The forward test only has four transitions, so the
    fold count follows the number of distinct seasons rather than being fixed at 5."""
    if X.shape[1] == 0:
        return float("nan")
    n = min(5, len(np.unique(groups)))
    p = cross_val_predict(rf(), X, y, cv=GroupKFold(n_splits=n), groups=groups)
    return float(np.corrcoef(p, y)[0, 1])


def main():
    recs = pd.read_csv(f"{HERE}/records.csv").set_index(["season", "team"])
    P, C = pff_matrix(), cfbd_matrix()
    df = P.join(C, how="inner").join(recs, how="inner")
    print(f"team-seasons with both sources: {len(df)}\n")

    pff_cols = [c for c in df.columns if c.startswith("pff_")]
    cfbd_cols = [c for c in df.columns if c.startswith("cfbd_")]
    y = df.adj_win_pct.to_numpy(float)
    g = df.index.get_level_values("season").to_numpy()

    # ---------------- 1. head to head ---------------------------------------
    print("=" * 68)
    print("1. SAME-SEASON: facets -> adjusted win pct (season-blocked 5-fold CV)")
    print("=" * 68)
    sets = {
        "PFF only": pff_cols,
        "CFBD only": cfbd_cols,
        "both": pff_cols + cfbd_cols,
        "PFF offense only": [c for c in pff_cols if FACET_GROUP.get(c[4:]) in
                             ("QB", "RB", "WR", "TE", "OL")],
        "CFBD offense only": [c for c in cfbd_cols if FACET_GROUP.get(c[5:]) in
                              ("QB", "RB", "WR", "TE")],
    }
    for nm, cols in sets.items():
        print(f"  {nm:<20} n_feat={len(cols):<3} CV r = {cv_r(df[cols].to_numpy(float), y, g):.3f}")

    # ---------------- 2. per-position swap ----------------------------------
    print("\n" + "=" * 68)
    print("2. PER-POSITION SWAP: baseline is the full PFF set. For each group,")
    print("   replace that group's PFF facets with CFBD's and re-score.")
    print("=" * 68)
    base = cv_r(df[pff_cols].to_numpy(float), y, g)
    print(f"  baseline (all PFF): CV r = {base:.3f}\n")
    print(f"  {'group':<6} {'PFF facets':<3} {'CFBD facets':<3}  {'swapped r':>10} {'delta':>8}  {'added r':>9} {'delta':>8}")
    for grp in ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB"]:
        pg = [c for c in pff_cols if FACET_GROUP.get(c[4:]) == grp]
        cg = [c for c in cfbd_cols if FACET_GROUP.get(c[5:]) == grp]
        if not cg:
            print(f"  {grp:<6} {len(pg):<11} {'-- CFBD has no data for this group --':<3}")
            continue
        swapped = [c for c in pff_cols if c not in pg] + cg
        added = pff_cols + cg
        rs = cv_r(df[swapped].to_numpy(float), y, g)
        ra = cv_r(df[added].to_numpy(float), y, g)
        print(f"  {grp:<6} {len(pg):<11} {len(cg):<12} {rs:>10.3f} {rs-base:>+8.3f}  {ra:>9.3f} {ra-base:>+8.3f}")

    # ---------------- 3. next season ----------------------------------------
    print("\n" + "=" * 68)
    print("3. NEXT SEASON: facets in year n -> adjusted win pct in year n+1")
    print("=" * 68)
    nxt = recs.reset_index()[["season", "team", "adj_win_pct"]].copy()
    nxt["season"] -= 1
    fwd = df.reset_index().merge(nxt.rename(columns={"adj_win_pct": "y1"}),
                                 on=["season", "team"], how="inner")
    y1 = fwd.y1.to_numpy(float)
    g1 = fwd.season.to_numpy()
    print(f"  pairs: {len(fwd)}")
    for nm, cols in sets.items():
        print(f"  {nm:<20} CV r = {cv_r(fwd[cols].to_numpy(float), y1, g1):.3f}")

    df.to_csv(f"{HERE}/horse_race_matrix.csv")


if __name__ == "__main__":
    main()
