"""Is the hybrid gain real, and which source produces the more stable player metric?

The horse race compares point estimates from one random seed on 658 team-seasons.
Differences of a hundredth of a correlation do not survive that, so part 1 repeats
the comparison across seeds and reports the spread.

Part 2 asks a different question that matters just as much for a WAR model: a metric
that swings wildly year to year describes what happened, it does not measure a player.
Year-over-year correlation of the same player's facet z-score is the cleanest test,
and it is where an outcome-anchored metric like PPA might beat a grade.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict

from horse_race import pff_matrix, cfbd_matrix, FACET_GROUP

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = range(8)


def cv_r(X, y, groups, seed):
    n = min(5, len(np.unique(groups)))
    m = RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                              random_state=seed, n_jobs=-1)
    p = cross_val_predict(m, X, y, cv=GroupKFold(n_splits=n), groups=groups)
    return float(np.corrcoef(p, y)[0, 1])


def part1():
    recs = pd.read_csv(f"{HERE}/records.csv").set_index(["season", "team"])
    df = pff_matrix().join(cfbd_matrix(), how="inner").join(recs, how="inner")
    pff = [c for c in df.columns if c.startswith("pff_")]
    cfbd = [c for c in df.columns if c.startswith("cfbd_")]
    y = df.adj_win_pct.to_numpy(float)
    g = df.index.get_level_values("season").to_numpy()

    print("=" * 70)
    print("1. SEED STABILITY (8 seeds, same-season target)")
    print("=" * 70)
    cand = {"PFF only": pff, "CFBD only": cfbd, "PFF + all CFBD": pff + cfbd,
            "PFF + CFBD QB only": pff + [c for c in cfbd if FACET_GROUP.get(c[5:]) == "QB"]}
    res = {}
    for nm, cols in cand.items():
        v = np.array([cv_r(df[cols].to_numpy(float), y, g, s) for s in SEEDS])
        res[nm] = v
        print(f"  {nm:<22} r = {v.mean():.3f}  sd {v.std():.4f}  "
              f"[{v.min():.3f}, {v.max():.3f}]")

    print("\n  paired deltas vs PFF only (same seed, so the seed cancels):")
    for nm in list(cand)[1:]:
        d = res[nm] - res["PFF only"]
        beats = (d > 0).sum()
        print(f"    {nm:<22} mean {d.mean():+.4f}  sd {d.std():.4f}  "
              f"wins {beats}/{len(SEEDS)}")


def part2():
    print("\n" + "=" * 70)
    print("2. PLAYER-LEVEL STABILITY: same player, year n vs year n+1 (facet z)")
    print("=" * 70)
    p = pd.read_parquet(f"{HERE}/facet_values.parquet")[
        ["season", "player_id", "facet", "z", "snaps"]]
    c = pd.read_parquet(f"{HERE}/cfbd_facet_values.parquet")[
        ["season", "player_id", "facet", "z", "snaps"]]

    # pair the sources on the facets that exist in both
    pairs = sorted(set(p.facet.unique()) & set(c.facet.unique()))
    print(f"  facets present in both sources: {pairs}\n")
    print(f"  {'facet':<10} {'PFF r':>8} {'n':>6}   {'CFBD r':>8} {'n':>6}   winner")
    for facet in pairs:
        out = {}
        for nm, d in (("PFF", p), ("CFBD", c)):
            f = d[d.facet == facet]
            # require a real workload in both seasons, else the pairing is noise
            cut = f.snaps.quantile(0.5)
            f = f[f.snaps >= cut]
            nxt = f[["season", "player_id", "z"]].copy()
            nxt["season"] -= 1
            j = f.merge(nxt.rename(columns={"z": "z1"}), on=["season", "player_id"])
            out[nm] = (np.corrcoef(j.z, j.z1)[0, 1], len(j)) if len(j) > 30 else (np.nan, len(j))
        pr, pn = out["PFF"]
        cr, cn = out["CFBD"]
        win = "CFBD" if (cr == cr and pr == pr and cr > pr) else "PFF"
        print(f"  {facet:<10} {pr:>8.3f} {pn:>6}   {cr:>8.3f} {cn:>6}   {win}")


if __name__ == "__main__":
    part1()
    part2()
