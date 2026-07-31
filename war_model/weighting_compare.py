"""Flat NNLS vs two-level weights: prediction, weight stability, ranking stability.

The claim two_level_weights.py makes is narrow and should be checked as such. It is NOT
that the model predicts wins better - the concept-level fit throws away degrees of
freedom the flat fit has, so if anything it should predict slightly worse, and
concept_verdict.json already found the difference immaterial. The claim is that the
player rankings stop depending on which team-seasons happened to land in the sample.

So all three are measured: what prediction costs, how much the weights move under
resampling, and how much the PLAYER RANKINGS move under resampling. The third is the
one that matters, because WAR is an attribution and attribution is what was broken.

Run: ./rbenv/bin/python weighting_compare.py
"""
import json, os

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr

import two_level_weights as tlw
from build_massey import massey_matrix
from build_war import REPL_WIN_PCT
from facets import YEARS

HERE = os.path.dirname(os.path.abspath(__file__))
FLAT_LAM = 1000.0
N_BOOT = 200
N_BOOT_RANK = 40
RANK_SEASON = 2025
# Bagging is PART of the two-level estimator, so a bootstrap replicate of it has to be
# bagged too - otherwise this measures the stability of an estimator nobody ships, and
# understates the one being proposed. Nested, hence the smaller inner count.
INNER_BOOT = 50


# ------------------------------------------------------------------------ data
def load():
    tot = pd.read_csv(f"{HERE}/hybrid_team_facet_totals.csv", index_col=[0, 1])
    Z = tot.groupby(level="season").transform(
        lambda c: (c - c.mean()) / c.std(ddof=0)).fillna(0.0)
    recs = pd.read_csv(f"{HERE}/records.csv")
    nxt = recs[["season", "team", "adj_win_pct"]].copy()
    nxt["season"] -= 1
    fwd = (Z.join(recs.set_index(["season", "team"]), how="inner").reset_index()
            .merge(nxt.rename(columns={"adj_win_pct": "next_win_pct"}),
                   on=["season", "team"], how="inner"))
    rel = tlw.load_reliability(pd.read_parquet(f"{HERE}/hybrid_facet_war.parquet"),
                               rebuild=True)
    return Z, tot, recs, fwd, rel, list(tot.columns)


def flat_weights(X, y, facets, lam=FLAT_LAM):
    """The current production fit, verbatim from build_hybrid.py."""
    A = np.vstack([X, np.sqrt(lam) * np.eye(len(facets))])
    b = np.concatenate([y - y.mean(), np.zeros(len(facets))])
    w = pd.Series(nnls(A, b)[0], index=facets)
    return w / w.sum()


# ------------------------------------------------- WAR under an arbitrary weighting
def wins_slope(Z, recs, sched, w):
    """The Massey rating -> win pct slope, which sets the wins scale."""
    rows = []
    for season in YEARS:
        zs = Z.xs(season, level="season")
        teams = sorted(t for t in zs.index
                       if ((recs.season == season) & (recs.team == t)).any())
        f = zs.loc[teams, w.index].to_numpy(float) @ w.to_numpy()
        f = f - f.mean()
        M, _ = massey_matrix(sched, season, teams)
        A = M.copy(); A[-1, :] = 1.0
        b = f.copy(); b[-1] = 0.0
        rows.append(pd.DataFrame({"season": season, "team": teams,
                                  "massey": np.linalg.solve(A, b)}))
    out = pd.concat(rows, ignore_index=True).merge(recs, on=["season", "team"])
    slope, _ = np.polyfit(out.massey, out.adj_win_pct, 1)
    return float(slope)


def player_war(fv, w, slope, pool, league_snaps):
    """WAR per player for one season, given a weight vector. Mirrors build_hybrid."""
    wf = fv.facet.map(w).to_numpy()
    contrib = wf * fv.value.to_numpy() / fv.sigma.to_numpy()
    waa = fv.games.to_numpy() * slope * fv.c_t.to_numpy() * contrib
    per_snap = pool * wf / fv.facet.map(league_snaps).to_numpy()
    war = waa + np.nan_to_num(per_snap) * fv.snaps.to_numpy()
    return (pd.DataFrame({"player_id": fv.player_id, "player": fv.player,
                          "position": fv.position, "team": fv.team, "war": war})
            .groupby(["player_id", "player", "position", "team"], as_index=False)
            .war.sum())


# ---------------------------------------------------------------------- reporting
def block_of(f):
    return "CFBD" if f.startswith("cfbd_") else f.split("_")[0]


def main():
    Z, tot, recs, fwd, rel, facets = load()
    sched = pd.read_csv(f"{HERE}/schedule.csv")
    X = fwd[facets].to_numpy(float)
    y = fwd.next_win_pct.to_numpy(float)
    seasons = fwd.season.to_numpy()

    flat = flat_weights(X, y, facets)
    two, info = tlw.build(fwd.set_index(["season", "team"])[facets], y, rel,
                          facets=facets, n_boot=N_BOOT)
    print(f"two-level: {info['n_concepts']} concepts in {info['n_groups']} groups, "
          f"lam={info['lam']} (flat fit uses {len(facets)} features, lam={FLAT_LAM:g})")
    print(f"\nzeros:  flat {int((flat == 0).sum())}/{len(facets)}   "
          f"two-level {int((two == 0).sum())}/{len(facets)}")
    print(f"collinear concepts grouped: {info['groups'] or 'none'}")

    print("\ngroup weights (fit against wins):")
    for c, v in info["group_weights"].items():
        print(f"  {c:<20} {v*100:5.1f}%")
    grouped = list(info["groups"])
    if grouped:
        print("\nsplit inside the grouped concepts (by univariate r):")
        for c in grouped:
            print(f"  {c:<20} {info['within_group'][c]*100:5.1f}%")

    # ---- 1. what does it cost to predict? --------------------------------
    print("\n" + "=" * 72)
    print("1. PREDICTION - season-blocked, next season's adjusted win pct")
    print("=" * 72)
    rows = []
    for name, wv in (("flat NNLS", flat), ("two-level", two)):
        preds, actuals = [], []
        for s in np.unique(seasons):
            tr, te = seasons != s, seasons == s
            if name == "flat NNLS":
                wb = flat_weights(X[tr], y[tr], facets)
            else:
                idx = fwd.index[tr]
                wb, _ = tlw.build(fwd.loc[idx].set_index(["season", "team"])[facets],
                                  y[tr], rel, facets=facets, n_boot=40)
            preds.append(X[te] @ wb.reindex(facets).to_numpy())
            actuals.append(y[te])
        p, a = np.concatenate(preds), np.concatenate(actuals)
        p = (p - p.mean()) / p.std() * a.std() + a.mean()   # common scale, weights are relative
        rows.append({"weights": name, "r": np.corrcoef(p, a)[0, 1],
                     "rmse": float(np.sqrt(np.mean((p - a) ** 2)))})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # ---- 2. do the weights hold still? -----------------------------------
    print("\n" + "=" * 72)
    print("2. WEIGHT STABILITY - 200 bootstrap resamples of team-seasons")
    print("=" * 72)
    rng = np.random.default_rng(0)
    draws = [rng.integers(0, len(X), len(X)) for _ in range(N_BOOT)]
    Bf = pd.DataFrame([flat_weights(X[i], y[i], facets) for i in draws])
    Bt = pd.DataFrame([tlw.build(fwd.iloc[i].set_index(["season", "team"])[facets],
                                 y[i], rel, facets=facets, n_boot=INNER_BOOT, seed=k)[0]
                       for k, i in enumerate(draws)])

    QB = [f for f in facets if f.startswith("QB_")]
    for name, B in (("flat NNLS", Bf), ("two-level", Bt)):
        sh = B[QB].div(B[QB].sum(axis=1), axis=0) * 100
        top = B[QB].idxmax(axis=1)
        print(f"\n  {name}")
        print(f"    QB_twp_rate share of QB block: "
              f"{sh.QB_twp_rate.quantile(.05):5.1f} - {sh.QB_twp_rate.quantile(.95):.1f}%")
        print(f"    QB_pass    share of QB block: "
              f"{sh.QB_pass.quantile(.05):5.1f} - {sh.QB_pass.quantile(.95):.1f}%   "
              f"(zeroed in {(B.QB_pass == 0).mean()*100:.0f}% of fits)")
        print(f"    facets topping the QB block: {top.nunique()}   "
              f"modal {top.value_counts(normalize=True).iloc[0]*100:.0f}%")
        print(f"    facets zeroed in >=25% of fits, all blocks: "
              f"{int((B == 0).mean().ge(.25).sum())}/{len(facets)}")

    # ---- 3. do the PLAYER RANKINGS hold still? ---------------------------
    print("\n" + "=" * 72)
    print(f"3. RANKING STABILITY - {RANK_SEASON} players, {N_BOOT_RANK} resamples")
    print("=" * 72)
    fv = pd.read_parquet(f"{HERE}/hybrid_facet_war.parquet")
    fv = fv[fv.season == RANK_SEASON].reset_index(drop=True)
    r = recs[recs.season == RANK_SEASON]
    pool = float(((0.5 - REPL_WIN_PCT) * r.fbs_games).sum())
    league_snaps = fv.groupby("facet").snaps.sum()

    base, boot_ranks = {}, {}
    for name, wv in (("flat NNLS", flat), ("two-level", two)):
        base[name] = player_war(fv, wv, wins_slope(Z, recs, sched, wv),
                                pool, league_snaps).set_index("player_id")
        ranks = []
        for k, i in enumerate(draws[:N_BOOT_RANK]):
            if name == "flat NNLS":
                wb = flat_weights(X[i], y[i], facets)
            else:
                wb, _ = tlw.build(fwd.iloc[i].set_index(["season", "team"])[facets],
                                  y[i], rel, facets=facets, n_boot=INNER_BOOT, seed=k)
            pw = player_war(fv, wb, wins_slope(Z, recs, sched, wb), pool, league_snaps)
            ranks.append(pw.set_index("player_id").war.rank(ascending=False))
        boot_ranks[name] = pd.DataFrame(ranks)

    print(f"\n{'':<12}{'rank sd':>10}{'top-50 churn':>15}{'mean rho':>11}")
    for name in ("flat NNLS", "two-level"):
        R = boot_ranks[name]
        b = base[name]
        played = b[b.war.abs() > 0].index
        sd = R[[c for c in R.columns if c in played]].std().median()
        top50 = set(b.nlargest(50, "war").index)
        churn = np.mean([len(top50 - set(R.columns[np.argsort(rr.to_numpy())[:50]]))
                         for _, rr in R.iterrows()])
        rho = np.mean([spearmanr(rr, b.war.reindex(rr.index).rank(ascending=False),
                                 nan_policy="omit").statistic for _, rr in R.iterrows()])
        print(f"{name:<12}{sd:>10.1f}{churn:>15.1f}{rho:>11.4f}")
    print("\n  rank sd     = median across players of the sd of their rank over resamples")
    print("  top-50 churn = how many of the base top 50 fall out, per resample")

    # ---- what actually changes -------------------------------------------
    print("\n" + "=" * 72)
    print("4. WHAT MOVES - 2025 quarterbacks")
    print("=" * 72)
    a = base["flat NNLS"]; b = base["two-level"]
    qb = a[a.position == "QB"].nlargest(15, "war")[["player", "team", "war"]]
    qb["two_level"] = b.war.reindex(qb.index)
    qb["rank_flat"] = a[a.position == "QB"].war.rank(ascending=False).reindex(qb.index)
    qb["rank_two"] = b[b.position == "QB"].war.rank(ascending=False).reindex(qb.index)
    qb["move"] = (qb.rank_flat - qb.rank_two).astype(int)
    print(qb.round(3).to_string(index=False))

    both = a.join(b.war.rename("war2"), how="inner")
    print(f"\noverall Spearman, flat vs two-level ({len(both)} players): "
          f"{spearmanr(both.war, both.war2).statistic:.4f}")

    two.rename("rf").to_frame().to_csv(f"{HERE}/two_level_facet_weights.csv")
    json.dump({"concept_weights": info["concept_weights"].round(5).to_dict(),
               "lam": info["lam"]},
              open(f"{HERE}/two_level_meta.json", "w"), indent=1)
    print("\n-> two_level_facet_weights.csv, two_level_meta.json")


if __name__ == "__main__":
    main()
