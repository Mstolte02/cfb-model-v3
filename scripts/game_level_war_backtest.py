"""Game-by-game PFF grades against the cumulative weeks 1..N window.

Mark's question (4 October 2026): what would in-season WAR look like built from each
game's grades rather than one season-to-date report?

Every game is scored on the production facet path (src/war_window.py on a one-week
PFF window, so z is taken among that week's players) and adjusted for THAT game's
opponent, rather than for the average of the window. Three ways of combining a
player's games through cut c are compared, each run through the same shipped
sample-weighted update (src/inseason_war.fit_rule / updated_rate) and graded on the
same schedule-neutral rest-of-season target as scripts/war_inseason_backtest.py:

  cumulative     the production input: one weeks 1..c report, window-average opponent
  game_equal     per-game rates, per-game opponent adjustment, snap-weighted mean
  game_recency   as game_equal, each game also weighted by delta ** (c - week), delta
                 fitted per position on earlier seasons

Parameters (opponent slope, delta, the update rule) are fitted on earlier seasons
only; 2023-25 are graded.

    python -m scripts.game_level_war_backtest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ARTIFACTS  # noqa: E402
from scripts import war_inseason_backtest as B  # noqa: E402
from src import inseason_war as IW  # noqa: E402
from src import opponent_strength as O  # noqa: E402

OUT = ARTIFACTS / "game_level_war_backtest.json"
GAMES = ARTIFACTS / "war_game_level_rows.parquet"
WEEKS = range(1, 16)
DELTAS = (1.0, .95, .9, .85, .8, .7, .6, .5)


def game_rows() -> pd.DataFrame:
    """One row per (season, week, player): game facet rate, snaps, opponent difficulty."""
    if GAMES.exists():
        return pd.read_parquet(GAMES)
    parts = []
    for s in B.TEST:
        g = O.games(s)
        st = O.strength(s)
        g = g[g.opponent.isin(st.index)].assign(
            off_diff=lambda x: -x.opponent.map(st.dfn), def_diff=lambda x: x.opponent.map(st.off))
        for w in WEEKS:
            try:
                fc = B.window(s, w, w)
            except FileNotFoundError:
                continue
            fc = fc[fc.group.notna() & (fc.snaps > 0)].copy()
            fc["team"] = fc.team_name.map(B._canon)
            gw = g[g.week == w].drop_duplicates("team").set_index("team")
            off = fc.team.map(gw.off_diff)
            dfn = fc.team.map(gw.def_diff)
            fc["diff"] = np.where(fc.group.isin(O.OFFENCE), off, dfn)
            fc["obs"] = fc.fc / fc.snaps * 1000
            parts.append(fc[["player_id", "group", "team", "snaps", "obs", "diff"]]
                         .assign(season=s, week=w))
            print(f"games {s} w{w}: {len(fc)}", flush=True)
    rows = pd.concat(parts, ignore_index=True)
    rows["player_id"] = rows.player_id.astype(str)
    rows.to_parquet(GAMES)
    return rows


def game_beta(rows: pd.DataFrame) -> dict:
    """Per group: slope of game rate on game difficulty, within player-season."""
    out = {}
    d = rows.dropna(subset=["diff"])
    d = d[d.snaps >= 10]
    for g, x in d.groupby("group"):
        key = [x.season, x.player_id]
        w = x.snaps.to_numpy()
        yd = x.obs - x.obs.groupby(key).transform("mean")
        xd = x["diff"] - x["diff"].groupby(key).transform("mean")
        out[g] = float(np.sum(w * xd * yd) / max(np.sum(w * xd * xd), 1e-9))
    return out


def aggregate(rows: pd.DataFrame, cut: int, beta: dict, delta: dict | float) -> pd.DataFrame:
    d = rows[rows.week <= cut].copy()
    b = d.group.map(beta).fillna(0.0)
    d["adj"] = d.obs - b * d["diff"].fillna(0.0)
    dl = d.group.map(delta) if isinstance(delta, dict) else pd.Series(delta, index=d.index)
    d["w"] = d.snaps * dl ** (cut - d.week)
    g = d.groupby(["season", "player_id"])
    return pd.DataFrame({"obs_game": g.apply(lambda x: np.average(x.adj, weights=x.w)),
                         "snaps_game": g.snaps.sum()}).reset_index()


def score(train, test, obs_col, snaps_col):
    tr = train.assign(obs=train[obs_col], snaps_w=train[snaps_col])
    te = test.assign(obs=test[obs_col], snaps_w=test[snaps_col])
    rule = IW.fit_rule(tr)
    out = []
    for (g, c), t in te.groupby(["group", "cut"]):
        p = rule.get(g, {}).get(str(int(c)))
        if p is None:
            continue
        pred = IW.updated_rate(p, t.m, t.P, t.obs, t.snaps_w)
        out.append(pd.DataFrame({"group": g, "cut": c, "err": (pred - t.target) ** 2,
                                 "w": t.snaps_t}))
    return pd.concat(out)


def main():
    rows = game_rows()
    fr = B.frame(B.history())
    fr = B.apply_opponent(fr, B.fit_opponent_beta(fr))
    fr["player_id"] = fr.player_id.astype(str)
    results = []
    for T in B.GRADED:
        train_rows = rows[rows.season < T]
        beta = game_beta(train_rows)
        aggs = {}
        for cut in B.CUTS:
            for name, delta in [("equal", 1.0)] + [(f"d{d}", d) for d in DELTAS[1:]]:
                a = aggregate(rows[rows.season <= T], cut, beta, delta).assign(cut=cut)
                aggs.setdefault(name, []).append(a)
        frames = {}
        for name, parts in aggs.items():
            a = pd.concat(parts)
            frames[name] = fr[fr.season <= T].merge(a, on=["season", "player_id", "cut"],
                                                    how="inner")
        base = frames["equal"]
        tr, te = base[base.season < T], base[base.season == T]
        results.append(score(tr, te, "obs", "snaps_w").assign(arm="cumulative", season=T))
        results.append(score(tr, te, "obs_game", "snaps_game").assign(arm="game_equal", season=T))
        # recency: choose delta per group on the training seasons
        best = {}
        for g in base.group.unique():
            errs = {}
            for name, f in frames.items():
                d = f[(f.season < T) & (f.group == g)]
                if len(d) < 50:
                    continue
                inner = d[d.season < T - 1]
                val = d[d.season == T - 1]
                if len(inner) < 50 or len(val) < 20:
                    errs[name] = 0.0 if name == "equal" else np.inf
                    continue
                e = score(inner, val, "obs_game", "snaps_game")
                errs[name] = float(np.average(e.err, weights=e.w))
            best[g] = min(errs, key=errs.get) if errs else "equal"
        rec = pd.concat([frames[best.get(g, "equal")].query("group == @g")
                         for g in base.group.unique()])
        tr, te = rec[rec.season < T], rec[rec.season == T]
        results.append(score(tr, te, "obs_game", "snaps_game").assign(arm="game_recency", season=T))
        print(f"graded {T}; recency picks {best}", flush=True)
    res = pd.concat(results, ignore_index=True)
    pooled = res.groupby("arm").apply(lambda x: np.average(x.err, weights=x.w))
    base = pooled["cumulative"]
    by = res.groupby(["group", "arm"]).apply(lambda x: np.average(x.err, weights=x.w)).unstack()
    report = {"pooled_pct_vs_cumulative": {a: round(100 * (v / base - 1), 2)
                                           for a, v in pooled.items()},
              "by_group_pct_vs_cumulative": {g: {a: round(100 * (r[a] / r["cumulative"] - 1), 2)
                                                 for a in r.index} for g, r in by.iterrows()}}
    OUT.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
