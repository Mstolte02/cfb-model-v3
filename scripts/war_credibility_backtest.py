"""Should a player's WAR prior be harder to move the more we know about him?

The cross-season question. A player's per-snap value (WAA per 1,000 snaps) is
predicted for season N from what he did before N, three ways:

  fixed      One weight per lag season (N-1, N-2, N-3), fitted by least squares per
             position group and applied to everybody alike. This is the "last year
             54%, the year before 33%, three back 12%" reading: sample-size agnostic,
             a 60-snap season counts as much as a 900-snap one.

  kalman     A per-player state-space filter. True value drifts year to year
             (theta_t = mu + rho (theta_{t-1} - mu) + eta, var d2); each season is a
             noisy look at it with variance sigma2 / snaps + omega2. A season's
             weight therefore grows with its snaps, and a player with a long, heavy
             record has a tight posterior that one season cannot move far. That is
             the hypothesis being tested.

  kalman_flat  The same filter with sigma2 = 0, so every season carries the same
             noise whatever its snaps. The difference kalman - kalman_flat is the
             value of the sample-size channel and nothing else.

Parameters are fitted by one-step-ahead Gaussian likelihood on seasons BEFORE the
test season only (expanding window), per position group. Graded on season N players
with >= 100 snaps and at least one prior season of >= 20 snaps in the same group.

    python -m scripts.war_credibility_backtest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC = ROOT / "war_model" / "hybrid_player_war.csv"
OUT = ROOT / "artifacts" / "war_credibility_backtest.json"

GROUP = {"QB": "QB", "HB": "RB", "FB": "RB", "RB": "RB", "WR": "WR", "TE": "TE",
         "T": "OT", "G": "IOL", "C": "IOL", "DI": "DT", "DL": "DT", "DT": "DT",
         "ED": "EDGE", "DE": "EDGE", "EDGE": "EDGE", "LB": "LB", "CB": "CB",
         "DB": "CB", "S": "SAF"}
MIN_OBS_SNAPS = 20      # a season below this is not a look at the player at all
MIN_TARGET_SNAPS = 100
TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]
SCALE = 1000.0          # WAA per 1,000 snaps


def load() -> pd.DataFrame:
    d = pd.read_csv(SRC, dtype={"player_id": str})
    d["group"] = d.position.map(GROUP)
    d = d[d.group.notna() & (d.snaps > 0)].copy()
    d["rate"] = d.waa / d.snaps * SCALE
    # one row per player-season (a mid-season transfer can appear twice): pool it
    g = d.groupby(["season", "player_id", "group"], as_index=False).agg(
        waa=("waa", "sum"), snaps=("snaps", "sum"))
    g["rate"] = g.waa / g.snaps * SCALE
    # keep the player's main group that season
    g = g.sort_values("snaps").drop_duplicates(["season", "player_id"], keep="last")
    return g


def panel(d: pd.DataFrame, group: str, seasons: list[int]):
    """Players x seasons arrays of rate and snaps, masked to this group."""
    sub = d[d.group == group]
    ids = np.sort(sub.player_id.unique())
    col = {s: i for i, s in enumerate(seasons)}
    R = np.full((len(ids), len(seasons)), np.nan)
    S = np.zeros((len(ids), len(seasons)))
    pos = pd.Series(np.arange(len(ids)), index=ids)
    sub = sub[sub.season.isin(seasons)]
    r, c = pos[sub.player_id].to_numpy(), sub.season.map(col).to_numpy()
    R[r, c] = sub.rate.to_numpy()
    S[r, c] = sub.snaps.to_numpy()
    return ids, R, S


def kalman(R, S, params, mu, upto):
    """Run the filter over columns [0, upto); return one-step predictions for every
    column in [0, upto] as (mean, state variance) BEFORE that column is seen."""
    rho, d2, sigma2, omega2 = params
    tau2 = d2 / max(1 - rho ** 2, 1e-6)
    n, T = R.shape
    m = np.full(n, mu)
    P = np.full(n, tau2)
    pm = np.empty((n, upto + 1))
    pP = np.empty((n, upto + 1))
    for t in range(upto + 1):
        if t > 0:
            m = mu + rho * (m - mu)
            P = rho ** 2 * P + d2
        pm[:, t], pP[:, t] = m, P
        if t == upto:
            break
        obs = (S[:, t] >= MIN_OBS_SNAPS) & np.isfinite(R[:, t])
        v = sigma2 / np.maximum(S[:, t], 1) + omega2
        k = np.where(obs, P / (P + v), 0.0)
        m = m + k * np.where(obs, R[:, t] - m, 0.0)
        P = (1 - k) * P
    return pm, pP


def fit_kalman(R, S, upto, mu, flat=False):
    """Max one-step likelihood over target columns 1..upto-1 (training seasons)."""
    tgt = (S >= MIN_TARGET_SNAPS) & np.isfinite(R)

    def unpack(x):
        rho = 1 / (1 + np.exp(-x[0]))
        d2, sigma2, omega2 = np.exp(x[1]), (0.0 if flat else np.exp(x[2])), np.exp(x[3])
        return rho, d2, sigma2, omega2

    def nll(x):
        p = unpack(x)
        pm, pP = kalman(R, S, p, mu, upto - 1)
        ll = 0.0
        for t in range(1, upto):
            ok = tgt[:, t]
            if not ok.any():
                continue
            var = pP[ok, t] + p[2] / S[ok, t] + p[3]
            e = R[ok, t] - pm[ok, t]
            ll += np.sum(np.log(var) + e ** 2 / var)
        return ll

    v0 = np.nanvar(R[tgt]) if tgt.any() else 1.0
    x0 = np.array([1.0, np.log(v0 * .3), np.log(v0 * 50), np.log(v0 * .3)])
    res = minimize(nll, x0, method="Nelder-Mead",
                   options=dict(maxiter=4000, xatol=1e-4, fatol=1e-4))
    return unpack(res.x)


def fixed_features(R, S, t):
    X = np.zeros((R.shape[0], 3))
    for k in (1, 2, 3):
        if t - k >= 0:
            ok = (S[:, t - k] >= MIN_OBS_SNAPS) & np.isfinite(R[:, t - k])
            X[ok, k - 1] = R[ok, t - k]
    avail = np.zeros_like(X, dtype=bool)
    for k in (1, 2, 3):
        if t - k >= 0:
            avail[:, k - 1] = (S[:, t - k] >= MIN_OBS_SNAPS) & np.isfinite(R[:, t - k])
    return X, avail


def fit_fixed(R, S, upto, mu):
    Xs, ys, ws = [], [], []
    for t in range(1, upto):
        X, avail = fixed_features(R, S, t)
        ok = (S[:, t] >= MIN_TARGET_SNAPS) & np.isfinite(R[:, t]) & avail.any(1)
        Xs.append(np.where(avail[ok], X[ok] - mu, 0.0))
        ys.append(R[ok, t] - mu)
        ws.append(S[ok, t])
    X, y, w = np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return beta


def main():
    d = load()
    seasons = sorted(d.season.unique())
    rows, params_out = [], {}
    for test in TEST_SEASONS:
        t = seasons.index(test)
        for g in sorted(d.group.unique()):
            ids, R, S = panel(d, g, seasons)
            train = (S[:, :t] >= MIN_TARGET_SNAPS) & np.isfinite(R[:, :t])
            mu = float(np.average(R[:, :t][train], weights=S[:, :t][train]))
            pk = fit_kalman(R, S, t, mu)
            pf = fit_kalman(R, S, t, mu, flat=True)
            beta = fit_fixed(R, S, t, mu)
            params_out[f"{test}/{g}"] = dict(
                mu=mu, kalman=dict(zip(["rho", "d2", "sigma2", "omega2"], pk)),
                kalman_flat=dict(zip(["rho", "d2", "sigma2", "omega2"], pf)),
                fixed_weights=beta.tolist())
            m_k, P_k = kalman(R, S, pk, mu, t)
            m_f, _ = kalman(R, S, pf, mu, t)
            X, avail = fixed_features(R, S, t)
            p_fix = mu + np.where(avail, X - mu, 0.0) @ beta
            ok = (S[:, t] >= MIN_TARGET_SNAPS) & np.isfinite(R[:, t]) & avail.any(1)
            prior_snaps = np.where(S[:, :t] >= MIN_OBS_SNAPS, S[:, :t], 0).sum(1)
            lag1 = np.where(avail[:, 0], S[:, t - 1], 0)
            for i in np.flatnonzero(ok):
                rows.append(dict(season=test, group=g, player_id=str(ids[i]),
                                 snaps=S[i, t], actual=R[i, t], mu=mu,
                                 fixed=p_fix[i], kalman=m_k[i, t],
                                 kalman_flat=m_f[i, t], post_sd=np.sqrt(P_k[i, t]),
                                 prior_snaps=prior_snaps[i], lag1_snaps=lag1[i]))
        print(f"{test} done", flush=True)

    res = pd.DataFrame(rows)
    arms = ["fixed", "kalman_flat", "kalman"]
    rng = np.random.default_rng(0)

    def wmse(df, arm):
        return float(np.average((df[arm] - df.actual) ** 2, weights=df.snaps))

    def waa_mse(df, arm):   # error in the season's WAA total, snaps known
        return float(np.mean(((df[arm] - df.actual) * df.snaps / SCALE) ** 2))

    def boot(df, a, b, fn, n=2000):
        ids = df.player_id.unique()
        by = {k: v for k, v in df.groupby("player_id")}
        out = []
        for _ in range(n):
            pick = rng.choice(ids, len(ids))
            s = pd.concat([by[k] for k in pick])
            out.append(fn(s, a) - fn(s, b))
        return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]

    summary = {}
    for name, fn in (("snap_weighted_mse_rate", wmse), ("season_waa_mse", waa_mse)):
        summary[name] = {
            "pooled": {a: fn(res, a) for a in arms},
            "by_season": {int(s): {a: fn(v, a) for a in arms}
                          for s, v in res.groupby("season")},
        }
    ci = {"kalman_vs_fixed": boot(res, "kalman", "fixed", wmse, 500),
          "kalman_vs_flat": boot(res, "kalman", "kalman_flat", wmse, 500)}
    res["info_bin"] = pd.cut(res.prior_snaps, [0, 300, 800, 1600, 99999],
                             labels=["<300", "300-800", "800-1600", "1600+"])
    by_info = {str(b): {a: wmse(v, a) for a in arms} | {"n": int(len(v))}
               for b, v in res.groupby("info_bin", observed=True)}
    by_group = {g: {a: wmse(v, a) for a in arms} | {"n": int(len(v))}
                for g, v in res.groupby("group")}
    corr = {a: float(np.corrcoef(res[a], res.actual)[0, 1]) for a in arms}
    out = dict(n=int(len(res)), arms=arms, summary=summary, ci_95=ci,
               by_prior_snaps=by_info, by_group=by_group, correlation=corr,
               params=params_out)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    res.to_csv(OUT.with_name("war_credibility_backtest_predictions.csv"), index=False)
    print(json.dumps({k: out[k] for k in
                      ("n", "summary", "ci_95", "by_prior_snaps", "by_group",
                       "correlation")}, indent=1, default=float))


if __name__ == "__main__":
    main()
