"""Deployment in WAR units: assignment difficulty and replacement level, per position.

Uses the per-game facet-WAR rows of scripts/game_level_war_backtest.py (every
2022-25 game scored on the production facet path), adjusted for each game's
opponent, joined to that game's PFF alignment snaps. Two estimates per group, both
continuous in the alignment shares, both shrunk (ridge, penalty by leave-one-season-
out), no significance gates:

  difficulty   within player-season: how his per-snap WAR rate moves with his
               alignment mix. A positive slope on "wide" means the same receiver
               produces more in wide-heavy games, so wide snaps are easier; the
               WAR adjustment judges a player's rate against the difficulty of the
               mix he actually played.
  replacement  games a regular missed: his own rate minus the rate of the players
               at his position who played instead, on his alignment shares. A
               positive slope means a missing player from that alignment is harder
               to replace, i.e. replacement level is lower there.

Validation: difficulty is graded on predicting each player's later games from his
earlier ones (raw vs difficulty-adjusted, as scripts/deployment_production_backtest);
replacement on predicting held-out seasons' replacement gaps. Both report the
percentage change in error.

    python -m scripts.deployment_rate_backtest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import ARTIFACTS  # noqa: E402
from src import deployment as DP  # noqa: E402

OUT = ARTIFACTS / "deployment_rate_backtest.json"
ROWS = ARTIFACTS / "war_game_level_rows.parquet"
SEASONS = [2022, 2023, 2024, 2025]
LAMBDAS = [0, .3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
# alignment shares tested per group; the omitted alignment is the base
ALIGN = {
    "WR": ["wide"], "TE": ["inline", "wide"], "OT": ["rt"], "IOL": ["c"],
    "DT": ["b_gap", "over_t", "outside_t"], "EDGE": ["outside_t", "off_ball"],
    "LB": ["box", "slot_d"], "CB": ["slot_d"], "SAF": ["deep", "box"],
}
SOURCES = {
    "receiving": {"wide_snaps": "wide", "slot_snaps": "slot", "inline_snaps": "inline"},
    "defense": {"snap_counts_box": "box", "snap_counts_slot": "slot_d", "snap_counts_fs": "deep",
                "snap_counts_corner": "outside_cb", "snap_counts_dl_a_gap": "a_gap",
                "snap_counts_dl_b_gap": "b_gap", "snap_counts_dl_over_t": "over_t",
                "snap_counts_dl_outside_t": "outside_t"},
    "pass_blocking": {"snap_counts_lt": "lt", "snap_counts_lg": "lg", "snap_counts_ce": "c",
                      "snap_counts_rg": "rg", "snap_counts_rt": "rt"},
}


def alignment_rows() -> pd.DataFrame:
    """Per (season, week, player_id): alignment snap counts from the one-week reports."""
    parts = []
    for s in SEASONS:
        for w in range(1, 16):
            frames = []
            for f, cols in SOURCES.items():
                d = DP._week(s, w, f)
                if d is None:
                    continue
                have = [c for c in cols if c in d.columns]
                x = d[["player_id", *have]].copy()
                for c in have:
                    x[c] = pd.to_numeric(x[c], errors="coerce")
                frames.append(x.groupby("player_id")[have].sum().rename(columns=cols))
            if frames:
                a = pd.concat(frames, axis=1).fillna(0.0)
                a = a.T.groupby(level=0).sum().T          # dedupe any shared label
                parts.append(a.reset_index().assign(season=s, week=w))
    return pd.concat(parts, ignore_index=True)


def shares(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Shares within the alignments that define the job at that position."""
    pools = {"WR": ["wide", "slot", "inline"], "TE": ["wide", "slot", "inline"],
             "OT": ["lt", "rt"], "IOL": ["lg", "c", "rg"],
             "DT": ["a_gap", "b_gap", "over_t", "outside_t"],
             "EDGE": ["b_gap", "over_t", "outside_t", "box"],
             "LB": ["box", "slot_d", "a_gap", "b_gap", "over_t", "outside_t"],
             "CB": ["slot_d", "outside_cb"], "SAF": ["deep", "box", "slot_d"]}
    pool = [c for c in pools[group] if c in df.columns]
    tot = df[pool].sum(axis=1).replace(0, np.nan)
    out = pd.DataFrame(index=df.index)
    for k in ALIGN[group]:
        if k == "off_ball":
            out[k] = df["box"] / tot
        elif k == "on_line":
            out[k] = df[["a_gap", "b_gap", "over_t", "outside_t"]].sum(axis=1) / tot
        else:
            out[k] = df[k] / tot
    return out


def ridge_fit(X, y, w, lam):
    sw = np.sqrt(w)
    A = np.c_[np.ones(len(X)), X] * sw[:, None]
    P = np.diag(np.r_[0.0, np.full(X.shape[1], lam)])
    return np.linalg.solve(A.T @ A + P, A.T @ (y * sw))


def main():
    rows = pd.read_parquet(ROWS)
    params = json.loads((ROOT / "data" / "live" / "inseason_war_params.json").read_text())
    beta = params.get("opp_beta", {})
    rows["adj"] = rows.obs - rows.group.map(beta).fillna(0) * rows["diff"].fillna(0)
    al = alignment_rows()
    al["player_id"] = al.player_id.astype(str)
    r = rows.merge(al, on=["season", "week", "player_id"], how="left")
    report = {}
    for g, ks in ALIGN.items():
        d = r[(r.group == g) & (r.snaps >= 10)].copy()
        sh = shares(d, g)
        d = d.drop(columns=[k for k in ks if k in d.columns]).join(sh)
        d = d.dropna(subset=ks)
        if len(d) < 1000:
            continue
        means = {k: float(np.average(d[k], weights=d.snaps)) for k in ks}

        # ---- difficulty: within player-season, ridge, LOSO over the prediction task
        key = [d.season, d.player_id]
        yd = (d.adj - d.groupby(key).adj.transform("mean")).to_numpy()
        Xd = np.column_stack([(d[k] - d.groupby(key)[k].transform("mean")).to_numpy() for k in ks])
        wv = d.snaps.to_numpy(float)
        best = None
        for lam in LAMBDAS:
            err = {"raw": 0.0, "adj": 0.0}
            for T in SEASONS:
                tr = (d.season != T).to_numpy()
                b = ridge_fit(Xd[tr], yd[tr], wv[tr], lam * len(yd) / 1000)[1:]
                te = d[d.season == T]
                for c in (3, 6, 9):
                    e = te[te.week <= c]; l = te[te.week > c]
                    if e.empty or l.empty:
                        continue
                    def agg(x):
                        w_ = x.snaps.to_numpy(float)
                        return pd.Series({"y": np.average(x.adj, weights=w_), "w": w_.sum(),
                                          **{k: np.average(x[k], weights=w_) for k in ks}})
                    E = e.groupby("player_id").apply(agg); L = l.groupby("player_id").apply(agg)
                    j = E.join(L, lsuffix="_e", rsuffix="_l", how="inner")
                    dsh = np.column_stack([j[f"{k}_e"] - j[f"{k}_l"] for k in ks])
                    err["raw"] += float(np.sum(j.w_l * (j.y_e - j.y_l) ** 2))
                    err["adj"] += float(np.sum(j.w_l * (j.y_e - dsh @ b - j.y_l) ** 2))
            ch = 100 * (err["adj"] / err["raw"] - 1)
            if best is None or ch < best[1]:
                best = (lam, ch)
        bfull = ridge_fit(Xd, yd, wv, best[0] * len(yd) / 1000)[1:]

        # ---- replacement: regular's rate minus his fill-ins' rate on his absences
        g_rate = d.groupby(["season", "team", "player_id"]).apply(
            lambda x: pd.Series({"rate": np.average(x.adj, weights=x.snaps), "sn": x.snaps.sum(),
                                 "gp": x.week.nunique(),
                                 **{k: np.average(x[k], weights=x.snaps) for k in ks}})).reset_index()
        tw = r[["season", "team", "week"]].drop_duplicates()
        top = g_rate.groupby(["season", "team"]).sn.transform("max")
        regs = g_rate[(g_rate.sn >= .6 * top) & (g_rate.gp >= 4)]
        unit = r[(r.group == g) & (r.snaps >= 1)]
        gaps = []
        for reg in regs.itertuples():
            weeks = set(tw[(tw.season == reg.season) & (tw.team == reg.team)].week)
            played = set(d[(d.season == reg.season) & (d.player_id == reg.player_id)].week)
            for wk_ in weeks - played:
                fill = unit[(unit.season == reg.season) & (unit.team == reg.team) & (unit.week == wk_)]
                if fill.snaps.sum() < 20:
                    continue
                gaps.append({"season": reg.season, "gap": reg.rate - np.average(fill.adj, weights=fill.snaps),
                             "w": min(reg.sn / reg.gp, fill.snaps.sum()),
                             **{k: getattr(reg, k) for k in ks}})
        gp = pd.DataFrame(gaps)
        rep = None
        if len(gp) >= 60:
            Xg = np.column_stack([gp[k] - means[k] for k in ks])
            bestg = None
            for lam in LAMBDAS:
                e0 = e1 = 0.0
                for T in SEASONS:
                    tr, te = gp.season != T, gp.season == T
                    if te.sum() == 0:
                        continue
                    b = ridge_fit(Xg[tr], gp.gap[tr].to_numpy(), gp.w[tr].to_numpy(float), lam)
                    p0 = np.average(gp.gap[tr], weights=gp.w[tr])
                    p1 = b[0] + Xg[te] @ b[1:]
                    e0 += float(np.sum(gp.w[te] * (p0 - gp.gap[te]) ** 2))
                    e1 += float(np.sum(gp.w[te] * (p1 - gp.gap[te]) ** 2))
                ch = 100 * (e1 / e0 - 1)
                if bestg is None or ch < bestg[1]:
                    bestg = (lam, ch)
            bg = ridge_fit(Xg, gp.gap.to_numpy(), gp.w.to_numpy(float), bestg[0])
            rep = {"absences": int(len(gp)), "lambda": bestg[0],
                   "error_change_pct": round(bestg[1], 3),
                   "mean_gap": round(float(bg[0]), 4),
                   "slope": {k: round(float(v), 4) for k, v in zip(ks, bg[1:])}}
        report[g] = {"mean_share": {k: round(v, 3) for k, v in means.items()},
                     "difficulty": {"lambda": best[0], "error_change_pct": round(best[1], 3),
                                    "slope": {k: round(float(v), 4) for k, v in zip(ks, bfull)}},
                     "replacement": rep, "player_games": int(len(d))}
        print(f"{g:5} difficulty {best[1]:+.3f}% slope={report[g]['difficulty']['slope']}  |  "
              f"replacement {rep and rep['error_change_pct']}% n={rep and rep['absences']} "
              f"slope={rep and rep['slope']}", flush=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
