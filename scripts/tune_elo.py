"""Optimize the in-season Elo parameters by backtest Brier (2022-25):

  K-factor   : constant vs decaying over the season (more confident later -> smaller
               updates as the pecking order establishes).
  Home field : constant vs team-specific (HFA varies by venue/program), estimated
               from each team's historical home-minus-away margin, shrunk to the mean.

No market data used. Run: ./venv/bin/python -m scripts.tune_elo
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import GAME_YEARS, OPP_ADJ_ALPHA, UNCERTAINTY_LAMBDA
from src.data import load, pff
from src import matchup as MU, oppadj as OA, model as M, elo as E
from scripts.train import load_bundle, raw_returning, blended_talent

SEASONS = [y for y in GAME_YEARS if y >= 2022]


def team_hfa_z(games):
    """z-scored per-team home advantage (home-minus-away margin), shrunk by games."""
    margins = {}
    for y in GAME_YEARS:
        for _, g in games[y].iterrows():
            if g.get("neutral_site", False) or pd.isna(g["home_points"]):
                continue
            m = g["home_points"] - g["away_points"]
            margins.setdefault(g["home_team"], []).append(("H", m))
            margins.setdefault(g["away_team"], []).append(("A", -m))
    raw = {}
    for t, vs in margins.items():
        hm = [m for v, m in vs if v == "H"]; am = [m for v, m in vs if v == "A"]
        if len(hm) >= 4 and len(am) >= 4:
            n = min(len(hm), len(am))
            raw[t] = ((np.mean(hm) - np.mean(am)) / 2) * (n / (n + 6))  # shrink
    s = pd.Series(raw)
    return (s - s.mean()) / (s.std(ddof=0) or 1.0)


def prepare(season, std, cfbd_tal, ret, games, pyth, ret_raw, talent, od):
    tr = [g for g in GAME_YEARS if g != season]
    b_o, b_d = MU.fit_talent_od_slopes(tr, std, talent, od_by_year=od)
    u = (1.0 - ret_raw[season]).clip(0, 1) if season in ret_raw else None
    unc = (UNCERTAINTY_LAMBDA, b_o, b_d, u) if u is not None else None
    fr = MU.team_frame(season, std, pyth, talent, ret, uncertainty=unc, od_by_year=od)
    parts = MU.assemble(tr, std, pyth, talent, ret, games, lam=UNCERTAINTY_LAMBDA,
                        b_o=b_o, b_d=b_d, ret_raw_by_year=ret_raw, od_by_year=od)
    mdl, _ = M.train(np.vstack([parts[g][0] for g in tr]),
                     np.concatenate([parts[g][1] for g in tr]),
                     np.concatenate([parts[g][2] for g in tr]))
    seed = {t: E.to_elo(mdl.win_prob(MU.vs_average_vector(fr, t), 0.0)) for t in fr.index}
    gl = []
    for _, gm in games[season].dropna(subset=["week"]).sort_values("week").iterrows():
        h, a = gm["home_team"], gm["away_team"]
        if h not in seed or a not in seed or gm["home_points"] == gm["away_points"]:
            continue
        gl.append({"week": int(gm["week"]), "h": h, "a": a,
                   "hp": gm["home_points"], "ap": gm["away_points"],
                   "neutral": bool(gm.get("neutral_site", False)),
                   "y": 1 if gm["home_points"] > gm["away_points"] else 0})
    return seed, gl


def walk(seed, gl, k_fn, hfa_fn):
    """k_fn(week, n_avg) where n_avg = avg games already played by the two teams."""
    R = dict(seed); gp = {}; rec = []
    for g in gl:
        nh, na = gp.get(g["h"], 0), gp.get(g["a"], 0)
        hfa = 0.0 if g["neutral"] else hfa_fn(g["h"])
        e = E.update(R, g["h"], g["a"], g["hp"], g["ap"], hfa,
                     k=k_fn(g["week"], (nh + na) / 2), neutral=g["neutral"])
        rec.append((g["week"], e, g["y"]))
        gp[g["h"]] = nh + 1; gp[g["a"]] = na + 1
    return pd.DataFrame(rec, columns=["week", "p", "y"])


def main():
    load.require_key()
    print("Preparing seasons (training seed models) ...")
    std, cfbd_tal, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()
    talent = blended_talent(cfbd_tal, pff.build_roster_talent())
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    prepared = {s: prepare(s, std, cfbd_tal, ret, games, pyth, ret_raw, talent, od)
                for s in SEASONS}
    hfa_z = team_hfa_z(games)
    team_hfa = lambda t: 65 + 25 * hfa_z.get(t, 0.0)   # tuned HFA, held fixed for K

    def preds(k_fn, hfa_fn=team_hfa):
        return pd.concat([walk(prepared[s][0], prepared[s][1], k_fn, hfa_fn)
                          for s in SEASONS], ignore_index=True)

    def brier(d):
        return np.mean((d["p"] - d["y"]) ** 2)

    def buckets(d):
        d = d.assign(b=pd.cut(d.week, [0, 4, 9, 99], labels=["wk1-4", "wk5-9", "wk10+"]))
        return {str(k): brier(v) for k, v in d.groupby("b", observed=True)}

    # K families: constant, week-decay (near optimum), games-played decay (principled)
    k_opts = [(f"const {k}", (lambda k: lambda w, n: k)(k)) for k in (30, 35, 40, 45, 50)]
    k_opts += [(f"wk-decay {a}->{b}",
                (lambda a, b: lambda w, n: max(b, a - (a - b) * (w - 1) / 11.0))(a, b))
               for a, b in [(50, 30), (45, 30), (50, 25)]]
    # games-played decay: K = K0 * n0/(n0 + games_played)  (Bayesian learning-rate)
    k_opts += [(f"gp-decay K0={K0},n0={n0}",
                (lambda K0, n0: lambda w, n: K0 * n0 / (n0 + n))(K0, n0))
               for K0, n0 in [(60, 6), (70, 5), (55, 8), (80, 4)]]

    print(f"\n{'K schedule':<22}{'pooled':>9}{'wk1-4':>9}{'wk5-9':>9}{'wk10+':>9}")
    print("-" * 58)
    results = []
    for lab, fn in k_opts:
        d = preds(fn); bk = buckets(d); pooled = brier(d)
        results.append((lab, pooled, fn))
        print(f"{lab:<22}{pooled:>9.4f}{bk['wk1-4']:>9.4f}{bk['wk5-9']:>9.4f}{bk['wk10+']:>9.4f}")
    best = min(results, key=lambda r: r[1])
    print(f"\nBest K: {best[0]} (pooled {best[1]:.4f})")


if __name__ == "__main__":
    main()
