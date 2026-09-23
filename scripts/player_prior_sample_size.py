"""In season: should a player's prior be harder to move the more it rests on?

`scripts/player_prior_stabilisation.py` showed that shrinking an in-season player
value toward his own last-season value beats shrinking toward the league mean for
receivers (and for quarterbacks early). It used ONE prior strength for everybody,
alpha = 10, so a prior built on a full 13-game season and one built on three
snaps-in-garbage-time appearances pulled equally hard.

The Gaussian answer says they should not. Last season's estimate carries its own
uncertainty, sigma^2 / (n_prev + alpha), and the player's true level drifts between
seasons by d^2, so the prior variance is tau_i^2 = d^2 + sigma^2 / (n_prev_i + alpha)
and the right ridge penalty is

    alpha_i = sigma^2 / tau_i^2 = 1 / (c + 1 / (n_prev_i + alpha)),   c = d^2 / sigma^2

which rises with n_prev and flattens at 1/c. One parameter, c. Arms, same data,
same folds, same target as the stabilisation script:

  flat          shrink to the league mean, alpha 10               (production today)
  prior_const   shrink to own prior, alpha 10 for all             (implement/player-prior-updates)
  prior_n       shrink to own prior, alpha_i from n_prev          (this hypothesis)
  decay_h       prior_const plus a recency weight 0.5^((cut-week)/h) on the games,
                the alternative a cloud session proposed

c and h are chosen on EARLIER test seasons only; 2022 has none, so the honest
comparison is 2023-25. The target is the rest of the season, opponent-adjusted the
same way but nearly unshrunk (alpha 0.1), so its noise is independent of every arm;
a shrunk target rewards whichever arm shrinks hardest. Graded by MSE weighted by
the games behind the target, on the value scale (level matters for WAR, no z-scoring)
and by correlation; 95% intervals by bootstrap over player-seasons.

    venv/Scripts/python -m scripts.player_prior_sample_size
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import ARTIFACTS, ROOT
from scripts.player_prior_stabilisation import (ALPHA, CUTS, MIN_AFTER, MIN_BEFORE,
                                                POSITIONS, PRIOR_SEASONS,
                                                TEST_SEASONS)
from src.availability import WEEKLY_DIR
from src.qbwar import fit_season_values

sys.path.insert(0, str(ROOT / "war_model"))
from build_roster_2026 import norm_name  # noqa: E402

RAW = ROOT / "data" / "raw"

OUT_JSON = ARTIFACTS / "player_prior_sample_size.json"
C_GRID = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2]
H_GRID = [2, 3, 5, 8]
# prior_best: ONE stiffness for everybody, chosen honestly. prior_n must beat this,
# not just alpha 10, or its gain is "stiffer overall" rather than "stiffer for the
# players we know more about".
K_GRID = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]


TARGET_ALPHA = 0.1     # near-unshrunk: the target's noise must not depend on any arm


def fit(frame, alpha=ALPHA, **kw):
    if len(frame) < 20 or frame.id.nunique() < 5:
        return pd.Series(dtype=float)
    val, _ = fit_season_values(frame, alpha=alpha, **kw)
    return val.set_index("id")["qb_value"]


def weekly_snaps(year: int) -> pd.DataFrame:
    """PFF offensive snaps per (week, team, normalised name). Weeks line up with
    CFBD's exactly (a +-1 shift drops the match rate from ~92% to ~65%)."""
    out = []
    for f in sorted(WEEKLY_DIR.glob(f"offense_{year}_w*.csv")):
        d = pd.read_csv(f, usecols=["player", "team", "snap_counts_total"])
        d["week"] = int(f.stem.split("_w")[1])
        d["key"] = d.player.map(norm_name)
        out.append(d)
    if not out:
        return pd.DataFrame(columns=["week", "team", "key", "snaps"])
    s = pd.concat(out).groupby(["week", "team", "key"], as_index=False).snap_counts_total.sum()
    return s.rename(columns={"snap_counts_total": "snaps"})


def load_games(year, position):
    """As player_prior_stabilisation.load_games, plus that game's PFF snaps.

    CFBD's per-game feed carries an average and no play count, so a one-target game
    and a twelve-target game look alike. PFF's weekly snap report is joined on
    (week, team, name); unmatched rows (~8%) get the position's median, so they
    count as a typical game rather than being dropped or zeroed."""
    fbs = {t["school"] for t in json.loads((RAW / f"teams_{year}.json").read_text())}
    rows = []
    for wk in range(1, 16):
        f = RAW / f"ppa_players_games_{year}_wk{wk}.json"
        if not f.exists():
            continue
        for p in json.loads(f.read_text()):
            if p.get("position") != position:
                continue
            a = p.get("averagePPA") or {}
            if a.get("all") is None or not p.get("opponent"):
                continue
            if p["team"] not in fbs or p["opponent"] not in fbs:
                continue
            rows.append({"week": wk, "id": str(p["id"]), "team": p["team"],
                         "key": norm_name(p.get("name", "")),
                         "opponent": p["opponent"], "ppa": float(a["all"])})
    g = pd.DataFrame(rows)
    if g.empty:
        return g
    g = g.merge(weekly_snaps(year), on=["week", "team", "key"], how="left")
    g["snap_matched"] = g.snaps.notna()
    g["snaps"] = g.snaps.fillna(g.snaps.median())
    return g


def strength_for(n_prev: dict, c: float) -> dict:
    return {i: (1.0 / (c + 1.0 / (n + ALPHA))) / ALPHA for i, n in n_prev.items()}


def main():
    rows = []
    for pos in POSITIONS:
        by_year = {y: load_games(y, pos) for y in [*PRIOR_SEASONS, *TEST_SEASONS]}
        priors, n_prev = {}, {}
        for y in PRIOR_SEASONS:
            priors[y + 1] = fit(by_year[y]).to_dict()
            n_prev[y + 1] = by_year[y].groupby("id").size().to_dict()
        # snaps in "typical games": last season's PFF snaps over the median game's
        med = float(pd.concat([by_year[y] for y in PRIOR_SEASONS]).snaps.median())
        snap_prev = {y + 1: (by_year[y].groupby("id").snaps.sum() / med).to_dict()
                     for y in PRIOR_SEASONS}
        print(pos, "snap match", {y: round(float(v.snap_matched.mean()), 3)
                                  for y, v in by_year.items()},
              "median snaps/game", med, flush=True)
        for y in TEST_SEASONS:
            g = by_year[y]
            for cut in CUTS:
                before, after = g[g.week <= cut], g[g.week > cut]
                nb, na = before.groupby("id").size(), after.groupby("id").size()
                keep = (set(nb[nb >= MIN_BEFORE].index) & set(na[na >= MIN_AFTER].index)
                        & set(priors[y]))
                if len(keep) < 10:
                    continue
                est = {"flat": fit(before),
                       "prior_const": fit(before, prior=priors[y])}
                for c in C_GRID:
                    est[f"prior_n:{c}"] = fit(before, prior=priors[y],
                                              strength=strength_for(n_prev[y], c))
                wts = np.clip(before.snaps.to_numpy() / med, .1, 3.0)
                for c in C_GRID:
                    st = strength_for(snap_prev[y], c)
                    est[f"prior_snap:{c}"] = fit(before, prior=priors[y], strength=st)
                    est[f"prior_snap_w:{c}"] = fit(before, prior=priors[y], strength=st,
                                                   sample_weight=wts)
                for k in K_GRID:
                    est[f"prior_best:{k}"] = fit(before, prior=priors[y],
                                                 alpha=ALPHA * k)
                for h in H_GRID:
                    w = 0.5 ** ((cut - before.week.to_numpy()) / h)
                    est[f"decay:{h}"] = fit(before, prior=priors[y], sample_weight=w)
                tgt = fit(after, alpha=TARGET_ALPHA)
                for pid in keep:
                    if pid not in tgt.index or any(pid not in e.index for e in est.values()):
                        continue
                    rows.append(dict(pos=pos, season=y, cut=cut, id=pid,
                                     n_prev=n_prev[y].get(pid, 0), n_before=int(nb[pid]),
                                     n_after=int(na[pid]),
                                     target=float(tgt[pid]),
                                     **{k: float(e[pid]) for k, e in est.items()}))
        print(f"{pos} done", flush=True)

    res = pd.DataFrame(rows)
    mse = lambda df, a: float(np.average((df[a] - df.target) ** 2, weights=df.n_after))

    # honest choice of c and h: per position, per test season, using earlier seasons
    def honest(prefix, grid):
        out = pd.Series(np.nan, index=res.index)
        chosen = {}
        for (pos, y), idx in res.groupby(["pos", "season"]).groups.items():
            past = res[(res.pos == pos) & (res.season < y)]
            if past.empty:
                continue
            best = min(grid, key=lambda v: mse(past, f"{prefix}:{v}"))
            chosen[f"{pos}/{y}"] = best
            out.loc[idx] = res.loc[idx, f"{prefix}:{best}"]
        return out, chosen

    res["prior_n"], chosen_c = honest("prior_n", C_GRID)
    res["decay"], chosen_h = honest("decay", H_GRID)
    res["prior_best"], chosen_k = honest("prior_best", K_GRID)
    res["prior_snap"], chosen_cs = honest("prior_snap", C_GRID)
    res["prior_snap_w"], chosen_csw = honest("prior_snap_w", C_GRID)
    arms = ["flat", "prior_const", "prior_best", "prior_n", "prior_snap",
            "prior_snap_w", "decay"]
    ev = res[res.season >= 2023].copy()

    rng = np.random.default_rng(0)
    keys = ev[["pos", "season", "id"]].astype(str).agg("|".join, axis=1)
    groups = {k: v.index.to_numpy() for k, v in ev.groupby(keys)}
    names = np.array(list(groups))

    def ci(a, b, df=ev, n=2000):
        d = (((df[a] - df.target) ** 2 - (df[b] - df.target) ** 2) * df.n_after).to_numpy()
        pos_of = {k: i for i, k in enumerate(df.index)}
        sums = np.array([d[[pos_of[j] for j in groups[k]]].sum() for k in names])
        cnts = np.array([df.n_after.to_numpy()[[pos_of[j] for j in groups[k]]].sum()
                         for k in names])
        draws = []
        for _ in range(n):
            pick = rng.integers(0, len(names), len(names))
            draws.append(sums[pick].sum() / cnts[pick].sum())
        return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]

    table = {}
    for pos, df in ev.groupby("pos"):
        table[pos] = {
            "n_rows": int(len(df)),
            "mse": {a: mse(df, a) for a in arms},
            "r": {a: float(np.corrcoef(df[a], df.target)[0, 1]) for a in arms},
            "by_cut": {int(c): {a: mse(v, a) for a in arms} | {"n": int(len(v))}
                       for c, v in df.groupby("cut")},
        }
    pooled = {a: mse(ev, a) for a in arms}
    intervals = {"prior_n_vs_prior_const": ci("prior_n", "prior_const"),
                 "prior_n_vs_prior_best": ci("prior_n", "prior_best"),
                 "prior_snap_vs_prior_best": ci("prior_snap", "prior_best"),
                 "prior_snap_w_vs_prior_best": ci("prior_snap_w", "prior_best"),
                 "decay_vs_prior_const": ci("decay", "prior_const"),
                 "prior_n_vs_flat": ci("prior_n", "flat")}
    for pos in table:
        sub = ev[ev.pos == pos]
        for arm in ("prior_n", "prior_snap", "prior_snap_w", "flat"):
            d = ((sub[arm] - sub.target) ** 2 - (sub.prior_best - sub.target) ** 2) * sub.n_after
            per = pd.DataFrame({"sum": d, "count": sub.n_after}).groupby(keys[sub.index]).sum()
            draws = []
            for _ in range(2000):
                pk = per.iloc[rng.integers(0, len(per), len(per))]
                draws.append(pk["sum"].sum() / pk["count"].sum())
            table[pos][f"{arm}_vs_prior_best_ci"] = [float(np.percentile(draws, 2.5)),
                                                     float(np.percentile(draws, 97.5))]
    ev["prev_bin"] = pd.cut(ev.n_prev, [0, 4, 8, 11, 99], labels=["1-4", "5-8", "9-11", "12+"])
    by_prev = {str(b): {a: mse(v, a) for a in arms} | {"n": int(len(v))}
               for b, v in ev.groupby("prev_bin", observed=True)}
    out = dict(alpha=ALPHA, c_grid=C_GRID, h_grid=H_GRID, chosen_c=chosen_c,
               chosen_h=chosen_h, chosen_k=chosen_k, chosen_c_snap=chosen_cs,
               chosen_c_snap_w=chosen_csw, eval_seasons=[2023, 2024, 2025],
               pooled_mse=pooled, ci_95=intervals, by_position=table,
               by_prior_games=by_prev,
               full_grid_mse={k: mse(res, k) for k in res.columns
                              if k.startswith(("prior_n:", "decay:", "prior_best:", "prior_snap"))
                              or k in ("flat", "prior_const")})
    OUT_JSON.write_text(json.dumps(out, indent=2))
    res.to_csv(OUT_JSON.with_name("player_prior_sample_size_predictions.csv"), index=False)
    print(json.dumps({k: out[k] for k in ("pooled_mse", "ci_95", "by_prior_games",
                                          "chosen_c", "chosen_c_snap", "chosen_c_snap_w", "chosen_k")}, indent=1))
    for pos, t in table.items():
        print(pos, t["n_rows"], {a: round(v, 5) for a, v in t["mse"].items()},
              "r", {a: round(v, 3) for a, v in t["r"].items()},
              "\n   ci vs prior_best", {a: [round(x, 5) for x in t[f"{a}_vs_prior_best_ci"]]
                                      for a in ("prior_n", "prior_snap", "prior_snap_w", "flat")})
        for c, v in t["by_cut"].items():
            print("   cut", c, {a: round(v[a], 5) for a in arms}, v["n"])


if __name__ == "__main__":
    main()
