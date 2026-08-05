"""Should a WR3 be graded against WR3s, or against every receiver in the country?

Every candidate is normalized by pooling all players at a position: a receiver's z is
his route grade against the snap-weighted mean of all receivers. Role bias in that
pool is large - WR1 route grade averages 73.9 and WR5 58.7, and 88% of WR5s sit below
the pooled mean against 20% of WR1s - so a fifth receiver scores negative largely for
being a fifth receiver.

Whether that is a BUG depends on something the grade cannot tell us:

  if the spread is ROLE          a WR3 who is excellent at being a WR3 is being
                                 punished for his depth-chart slot, and the facet is
                                 partly measuring how concentrated a team's targets
                                 are rather than how good its receivers are.
  if the spread is SELECTION     better receivers earn more targets, the WR1 really is
                                 better, and normalizing it away destroys real signal.

Both are true to some degree and no amount of staring at grades separates them. What
can be tested is which normalization predicts NEXT season's wins better, which is the
question the model actually needs answered.

Three schemes, per position group:
  pooled   z against every player at the position          (ships today)
  tier     z within depth-chart tier, ranked by opportunity
  partial  pooled z with the tier mean subtracted, which removes role bias while
           keeping the part of the tier gap that varies within a tier

Run: ./rbenv/bin/python role_normalization.py
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.metrics import mean_squared_error

# add_tiers/TIER_BY/MAX_TIER moved into candidates.py when `partial` was promoted
# from experiment to shipped default. They live where the facet values are built so
# that this comparison and the build cannot disagree about what a tier is.
from candidates import CATALOGUE, MIN_DENOM, LOWER_IS_BETTER, MAX_TIER, TIER_BY, \
    add_tiers, build_catalogue, group_of, load_players

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM_MAP = json.load(open(f"{HERE}/team_map.json"))


def facet_frame(players, scheme):
    """Player-level values under one normalization scheme."""
    rows = []
    for name, metric, denom, positions, kind in build_catalogue():
        if metric not in players.columns or denom not in players.columns:
            continue
        d = players[players.position.isin(positions)][
            ["season", "player_id", "team_name", "tier"]].copy()
        d["metric"] = pd.to_numeric(players.loc[d.index, metric], errors="coerce")
        d["snaps"] = pd.to_numeric(players.loc[d.index, denom], errors="coerce")
        d = d[(d.snaps >= MIN_DENOM[kind]) & d.metric.notna()]
        if len(d) < 200:
            continue
        if metric in LOWER_IS_BETTER:
            d["metric"] = -d["metric"]
        out = []
        for season, g in d.groupby("season"):
            wt, x = g.snaps.to_numpy(float), g.metric.to_numpy(float)
            if scheme == "pooled":
                mu = np.average(x, weights=wt)
                sd = np.sqrt(np.average((x - mu) ** 2, weights=wt))
                z = (x - mu) / sd if sd else np.zeros_like(x)
            elif scheme == "tier":
                z = np.zeros_like(x)
                for t in np.unique(g.tier):
                    m = (g.tier == t).to_numpy()
                    if m.sum() < 30:
                        m2 = np.ones_like(m, bool)      # too few: fall back to pooled
                    else:
                        m2 = m
                    mu = np.average(x[m2], weights=wt[m2])
                    sd = np.sqrt(np.average((x[m2] - mu) ** 2, weights=wt[m2]))
                    z[m] = (x[m] - mu) / sd if sd else 0.0
            else:  # partial: pooled scale, tier mean removed
                mu = np.average(x, weights=wt)
                sd = np.sqrt(np.average((x - mu) ** 2, weights=wt))
                tier_mu = pd.Series(x).groupby(g.tier.to_numpy()).transform("mean").to_numpy()
                z = (x - tier_mu) / sd if sd else np.zeros_like(x)
            z = np.clip(z, -6, 6)
            out.append(pd.DataFrame({"season": season, "team_name": g.team_name.to_numpy(),
                                     "facet": name, "value": z * wt}))
        rows.extend(out)
    fv = pd.concat(rows, ignore_index=True)
    fv["team"] = fv.team_name.map(TEAM_MAP)
    names = sorted(fv.facet.unique())
    tot = (fv.groupby(["season", "team", "facet"], as_index=False)["value"].sum()
             .pivot(index=["season", "team"], columns="facet", values="value")
             .reindex(columns=names).fillna(0.0))
    Z = tot.groupby(level="season").transform(
        lambda c: (c - c.mean()) / c.std(ddof=0)).fillna(0.0)
    return Z, names


def score(Z, names, lam=1000.0):
    """Season-blocked next-season RMSE under non-negative ridge, plus WR/RB shares."""
    recs = pd.read_csv(f"{HERE}/records.csv")
    nxt = recs[["season", "team", "adj_win_pct"]].copy()
    nxt["season"] -= 1
    nxt = nxt.rename(columns={"adj_win_pct": "y"}).set_index(["season", "team"])
    df = Z.join(nxt, how="inner").dropna(subset=["y"])
    X, y = df[names].to_numpy(float), df.y.to_numpy(float)
    seasons = df.index.get_level_values("season").to_numpy()

    def nnr(A, b):
        n = A.shape[1]
        c, _ = nnls(np.vstack([A, np.sqrt(lam) * np.eye(n)]),
                    np.concatenate([b - b.mean(), np.zeros(n)]))
        return c, b.mean()

    P = np.full(len(y), np.nan)
    for s in sorted(set(seasons)):
        tr, te = seasons != s, seasons == s
        if tr.sum() < 200:
            continue
        c, mu = nnr(X[tr], y[tr])
        P[te] = X[te] @ c + mu
    ok = ~np.isnan(P)
    rmse = float(np.sqrt(mean_squared_error(y[ok], P[ok])))
    c, _ = nnr(X, y)
    w = pd.Series(c, index=names)
    share = w.groupby([n.split("_")[0] for n in names]).sum()
    share = (share / share.sum() * 100)
    return rmse, share


def main():
    print("loading players ...")
    players = add_tiers(load_players())
    wr = players[players.grp == "WR"]
    print(f"  tiers assigned; WR tier counts: "
          f"{wr.tier.value_counts().sort_index().to_dict()}\n")

    results = {}
    print(f"{'scheme':<10}{'rmse (next season)':>20}{'WR share':>11}{'RB share':>11}"
          f"{'TE':>7}{'QB':>7}")
    print("-" * 66)
    for scheme in ("pooled", "tier", "partial"):
        Z, names = facet_frame(players, scheme)
        rmse, share = score(Z, names)
        results[scheme] = {"rmse": rmse,
                           "shares": {k: round(float(v), 2) for k, v in share.items()}}
        print(f"{scheme:<10}{rmse:>20.4f}{share.get('WR', 0):>10.1f}%"
              f"{share.get('RB', 0):>10.1f}%{share.get('TE', 0):>6.1f}%"
              f"{share.get('QB', 0):>6.1f}%", flush=True)

    base = results["pooled"]["rmse"]
    best = min(results, key=lambda k: results[k]["rmse"])
    print(f"\nbest: {best} ({results[best]['rmse']:.4f} vs pooled {base:.4f}, "
          f"{results[best]['rmse'] - base:+.4f})")
    if results[best]["rmse"] > base - 0.0005:
        print("  -> no material gain; the tier gap is mostly SELECTION, not role bias")
    else:
        print("  -> role-relative normalization predicts better; the tier gap was bias")

    json.dump(results, open(f"{HERE}/role_normalization.json", "w"), indent=1)
    print("\n-> role_normalization.json")


if __name__ == "__main__":
    main()
