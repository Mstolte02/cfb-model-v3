"""In season, on PFF WAR at every position: should the update depend on sample size?

Mark's hypothesis: the more we know about a player, the harder it should be for him to
move off his prior - and, the other way round, a week-3 value built on 40 snaps
should move nobody. The alternative a cloud session proposed was a recency decay that
treats everyone alike. This tests them on the quantity the site actually shows,
player WAR from PFF, for every position including the offensive line and defence.

Data (all through src/war_window.py, so every number is on the production facet path
and weights):

  history   full-season facet-WAR per 1,000 snaps, 2014-25 less 2020, from the same
            files the production build reads
  window    weeks 1-W of the test season, from PFF API week-window pulls
  target    weeks W+1-16 of the test season, likewise; graded weighted by its snaps

Prior: a per-player Kalman filter over the player's full seasons
(scripts/war_credibility_backtest.py), giving a mean m and a variance P that shrinks
the more snaps sit behind it. Arms:

  prior_only            m
  window_only           the week 1-W value
  same_for_all          (1 - lam) * m_fixed + lam * window, lam one number per position
                        and cut; m_fixed is the fixed-year-weights prior. Nobody's
                        update depends on how much is known about him.
  same_for_all_kalman   the same one-lambda blend, starting from the Kalman mean m, so
                        bayes vs this isolates the update rule from the better prior
  bayes                 precision-weighted: m with variance P, window with variance
                        s2 / snaps_w. Both halves depend on sample size.
  bayes_flat_prior      as bayes, P replaced by its position mean (prior depth ignored)
  bayes_flat_window     as bayes, window snaps replaced by the position mean

Every arm gets the same linear calibration a + b * estimate, and every free parameter
(lam, s2, the window scale g, a, b) is fitted per position and cut on EARLIER test
seasons only. 2022 is therefore training only; graded seasons are 2023-25.

    python -m scripts.war_inseason_backtest
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from config import ARTIFACTS  # noqa: E402
from scripts.sync_pff_war_windows import LEGACY, POSITION, window_dir  # noqa: E402
from scripts.war_credibility_backtest import (MIN_OBS_SNAPS, fit_fixed,  # noqa: E402
                                              fit_kalman, fixed_features, kalman)
from src import war_window as ww  # noqa: E402

HIST = ARTIFACTS / "war_facet_history.parquet"
OUT = ARTIFACTS / "war_inseason_backtest.json"
SEASONS_ALL = [y for y in range(2014, 2026) if y != 2020]
TEST = [2022, 2023, 2024, 2025]
GRADED = [2023, 2024, 2025]
CUTS = {3: ((1, 3), (4, 16)), 6: ((1, 6), (7, 16))}
MIN_WINDOW = 20
MIN_TARGET = 100
PREFIX = {"passing": "pass", "rushing": "rush", "receiving": "recv",
          "blocking": "blk", "defense": "def", "pass_blocking": "pblk",
          "run_blocking": "rblk", "pass_rush": "prsh", "run_defense": "rdef",
          "coverage": "cov"}


def history() -> pd.DataFrame:
    if HIST.exists():
        return pd.read_parquet(HIST)
    parts = []
    for s in SEASONS_ALL:
        p = ww.load_players_from(ww.season_files(s), s)
        parts.append(ww.facet_contrib(p, s))
        print(f"history {s}", flush=True)
    h = pd.concat(parts, ignore_index=True)
    h.to_parquet(HIST)
    return h


def window(season: int, a: int, b: int) -> pd.DataFrame:
    d = window_dir(season, a, b)
    files = {PREFIX[n]: d / f"{n}.csv" for n in (*LEGACY, *POSITION)}
    missing = [f for f in files.values() if not f.exists()]
    if missing:
        raise FileNotFoundError(f"window not staged: {missing[0]}")
    return ww.facet_contrib(ww.load_players_from(files, season), season)


def rate(df):
    return df.fc / df.snaps * ww.SCALE


def priors(hist: pd.DataFrame, test: int) -> pd.DataFrame:
    """Kalman (m, P) and fixed-weights prior for every player, from seasons < test."""
    h = hist[hist.snaps > 0].copy()
    h["rate"] = rate(h)
    h = h.sort_values("snaps").drop_duplicates(["season", "player_id"], keep="last")
    seasons = [s for s in SEASONS_ALL if s < test] + [test]
    t = len(seasons) - 1
    out = []
    for g in sorted(h.group.dropna().unique()):
        sub = h[(h.group == g) & h.season.isin(seasons[:-1])]
        ids = np.sort(sub.player_id.unique())
        pos = pd.Series(np.arange(len(ids)), index=ids)
        col = {s: i for i, s in enumerate(seasons)}
        R = np.full((len(ids), len(seasons)), np.nan)
        S = np.zeros((len(ids), len(seasons)))
        R[pos[sub.player_id], sub.season.map(col)] = sub.rate
        S[pos[sub.player_id], sub.season.map(col)] = sub.snaps
        train = (S[:, :t] >= 100) & np.isfinite(R[:, :t])
        mu = float(np.average(R[:, :t][train], weights=S[:, :t][train]))
        pk = fit_kalman(R, S, t, mu)
        m, P = kalman(R, S, pk, mu, t)
        beta = fit_fixed(R, S, t, mu)
        X, avail = fixed_features(R, S, t)
        out.append(pd.DataFrame({
            "player_id": ids, "group": g, "mu": mu, "m": m[:, t], "P": P[:, t],
            "m_fixed": mu + np.where(avail, X - mu, 0.0) @ beta,
            "prior_snaps": np.where(S[:, :t] >= MIN_OBS_SNAPS, S[:, :t], 0).sum(1),
            "sigma2": pk[2], "omega2": pk[3], "tau2": pk[1] / max(1 - pk[0] ** 2, 1e-6)}))
    return pd.concat(out, ignore_index=True)


def frame(hist) -> pd.DataFrame:
    rows = []
    for test in TEST:
        pr = priors(hist, test)
        for cut, (wa, wb) in CUTS.items():
            w = window(test, *wa)
            r = window(test, *wb)
            d = (w[["player_id", "group", "snaps", "fc"]]
                 .merge(r[["player_id", "snaps", "fc"]], on="player_id",
                        suffixes=("_w", "_t")))
            d = d[(d.snaps_w >= MIN_WINDOW) & (d.snaps_t >= MIN_TARGET) & d.group.notna()]
            d["obs"] = d.fc_w / d.snaps_w * ww.SCALE
            d["target"] = d.fc_t / d.snaps_t * ww.SCALE
            d = d.merge(pr, on=["player_id", "group"], how="left")
            # newcomers: population prior
            gp = pr.groupby("group").agg(mu=("mu", "first"), tau2=("tau2", "first"))
            new = d.m.isna()
            d.loc[new, "mu"] = d.loc[new, "group"].map(gp.mu)
            d.loc[new, "m"] = d.loc[new, "mu"]
            d.loc[new, "m_fixed"] = d.loc[new, "mu"]
            d.loc[new, "P"] = d.loc[new, "group"].map(gp.tau2)
            d.loc[new, "prior_snaps"] = 0
            d["season"], d["cut"] = test, cut
            rows.append(d)
        print(f"frame {test}", flush=True)
    return pd.concat(rows, ignore_index=True)


def wmse(y, p, w):
    return float(np.average((p - y) ** 2, weights=w))


def calib(train, col):
    X = np.c_[np.ones(len(train)), train[col]]
    sw = np.sqrt(train.snaps_t.to_numpy())
    return np.linalg.lstsq(X * sw[:, None], train.target.to_numpy() * sw, rcond=None)[0]


def bayes(df, g, s2, flat_prior=False, flat_window=False, P_mean=None, n_mean=None):
    P = np.full(len(df), P_mean) if flat_prior else df.P.to_numpy()
    n = np.full(len(df), n_mean) if flat_window else df.snaps_w.to_numpy()
    v = s2 / n
    k = P / (P + v)
    return df.m.to_numpy() + k * (g * df.obs.to_numpy() - df.m.to_numpy())


G_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
S_GRID = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
LAM_GRID = np.linspace(0, 1, 21)


def evaluate(fr: pd.DataFrame):
    preds = []
    chosen = {}
    for (g, cut, season), te in fr.groupby(["group", "cut", "season"]):
        if season not in GRADED:
            continue
        tr = fr[(fr.group == g) & (fr.cut == cut) & (fr.season < season)]
        if len(tr) < 50:
            continue
        P_mean, n_mean = float(tr.P.mean()), float(tr.snaps_w.mean())
        est_tr, est_te = {}, {}
        est_tr["prior_only"], est_te["prior_only"] = tr.m.to_numpy(), te.m.to_numpy()
        est_tr["window_only"], est_te["window_only"] = tr.obs.to_numpy(), te.obs.to_numpy()
        lam = min(LAM_GRID, key=lambda l: wmse(
            tr.target, (1 - l) * tr.m_fixed + l * tr.obs, tr.snaps_t))
        est_tr["same_for_all"] = ((1 - lam) * tr.m_fixed + lam * tr.obs).to_numpy()
        est_te["same_for_all"] = ((1 - lam) * te.m_fixed + lam * te.obs).to_numpy()
        lk = min(LAM_GRID, key=lambda l: wmse(
            tr.target, (1 - l) * tr.m + l * tr.obs, tr.snaps_t))
        est_tr["same_for_all_kalman"] = ((1 - lk) * tr.m + lk * tr.obs).to_numpy()
        est_te["same_for_all_kalman"] = ((1 - lk) * te.m + lk * te.obs).to_numpy()
        pick = {"lam": float(lam), "lam_kalman": float(lk)}
        for arm, kw in (("bayes", {}), ("bayes_flat_prior", {"flat_prior": True}),
                        ("bayes_flat_window", {"flat_window": True})):
            kw = kw | {"P_mean": P_mean, "n_mean": n_mean}
            best = min(((gg, s) for gg in G_GRID for s in S_GRID),
                       key=lambda p: wmse(tr.target, bayes(tr, *p, **kw), tr.snaps_t))
            est_tr[arm], est_te[arm] = bayes(tr, *best, **kw), bayes(te, *best, **kw)
            pick[arm] = best
        out = te[["season", "cut", "group", "player_id", "snaps_w", "snaps_t", "target",
                  "prior_snaps"]].copy()
        for arm in est_te:
            t2 = tr.assign(_e=est_tr[arm])
            a, b = calib(t2, "_e")
            out[arm] = a + b * est_te[arm]
        preds.append(out)
        chosen[f"{g}/{cut}/{season}"] = pick
    return pd.concat(preds, ignore_index=True), chosen


ARMS = ["prior_only", "window_only", "same_for_all", "same_for_all_kalman", "bayes",
        "bayes_flat_prior", "bayes_flat_window"]


def boot_ci(df, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = ((df[a] - df.target) ** 2 - (df[b] - df.target) ** 2) * df.snaps_t
    per = pd.DataFrame({"s": d, "w": df.snaps_t}).groupby(df.player_id.astype(str)).sum()
    s, w = per.s.to_numpy(), per.w.to_numpy()
    draws = []
    for _ in range(n):
        i = rng.integers(0, len(s), len(s))
        draws.append(s[i].sum() / w[i].sum())
    base = float(np.average((df[b] - df.target) ** 2, weights=df.snaps_t))
    # as a percentage of arm b's error, which is what the tables report
    return [round(100 * float(np.percentile(draws, 2.5)) / base, 2),
            round(100 * float(np.percentile(draws, 97.5)) / base, 2)]


def main():
    hist = history()
    fr = frame(hist)
    res, chosen = evaluate(fr)
    res.to_csv(OUT.with_name("war_inseason_backtest_predictions.csv"), index=False)
    m = lambda df: {a: wmse(df.target, df[a], df.snaps_t) for a in ARMS}
    rel = lambda d: {a: round(100 * (v / d["same_for_all"] - 1), 2) for a, v in d.items()}
    out = {"n": int(len(res)), "pooled": m(res), "pooled_pct_vs_same_for_all": rel(m(res)),
           "ci_95": {"bayes_vs_same_for_all": boot_ci(res, "bayes", "same_for_all"),
                     "bayes_vs_same_for_all_kalman": boot_ci(res, "bayes",
                                                             "same_for_all_kalman"),
                     "bayes_vs_flat_prior": boot_ci(res, "bayes", "bayes_flat_prior"),
                     "bayes_vs_flat_window": boot_ci(res, "bayes", "bayes_flat_window")},
           "by_cut": {int(c): m(v) | {"n": int(len(v))} for c, v in res.groupby("cut")},
           "by_season": {int(s): m(v) for s, v in res.groupby("season")},
           "by_group": {}, "chosen": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                                          for kk, vv in v.items()}
                                      for k, v in chosen.items()}}
    for g, v in res.groupby("group"):
        mm = m(v)
        out["by_group"][g] = {"n": int(len(v)), "pct_vs_same_for_all": rel(mm),
                              "bayes_vs_same_for_all_ci": boot_ci(v, "bayes", "same_for_all"),
                              "bayes_vs_same_kalman_ci": boot_ci(v, "bayes",
                                                                 "same_for_all_kalman"),
                              "by_cut": {int(c): rel(m(vv)) for c, vv in v.groupby("cut")}}
    res["depth"] = pd.cut(res.prior_snaps, [-1, 0, 300, 800, 1600, 1e9],
                          labels=["newcomer", "<300", "300-800", "800-1600", "1600+"])
    out["by_prior_depth"] = {str(k): rel(m(v)) | {"n": int(len(v))}
                             for k, v in res.groupby("depth", observed=True)}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("n", "pooled", "pooled_pct_vs_same_for_all",
                                          "ci_95", "by_cut", "by_season",
                                          "by_prior_depth")}, indent=1))
    for g, v in out["by_group"].items():
        print(g, v["n"], v["pct_vs_same_for_all"], "ci %", v["bayes_vs_same_for_all_ci"],
              "vs kalman blend ci %", v["bayes_vs_same_kalman_ci"])


if __name__ == "__main__":
    main()
