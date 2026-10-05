"""Parts 2 and 3: is WAR earned in a deployment worth more, and is it harder to replace?

No significance gates. Every effect is estimated with ridge regression whose penalty
is chosen by leave-one-season-out cross-validation, so an effect the data cannot
support is shrunk toward zero and one it can enters at its supported size.

Part 2 - value (team-seasons 2021-25). The position-appropriate, opponent-adjusted
unit outcome (src/unit_outcomes: pass offence for WR/TE/OL, rush offence for RB,
pass defence for CB/S/LB/EDGE, rush defence for DT) on
    W   the unit's production WAR
    D_k sum of WAR_i * (share_ik - league mean share_k), one per alignment k
plus the other units of that side and the team's pass rate. D_k credits WAR earned
in alignment k, never the alignment itself. relative value of alignment k
= coef(D_k) / coef(W): +0.5 means a WAR earned one full share further into k is
worth 50% more.

Part 3 - replacement (team-games 2022-25). A regular (60%+ of his team's snaps in the
games he played) who misses a game takes his per-game WAR with him. The game's unit
outcome on
    M   per-game WAR of the unit's missing regulars
    E_k sum of missing WAR * (share_k - mean share_k)
coef(M) is the loss per missing WAR; coef(E_k)/coef(M) says whether a missing player
from alignment k is harder (positive) or easier to replace.

    python -m scripts.deployment_value_backtest
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
from scripts import deployment_war_backtest as DW  # noqa: E402
from src import unit_outcomes as U  # noqa: E402
from src.war_window import GROUP  # noqa: E402

OUT = ARTIFACTS / "deployment_value_backtest.json"
SEASONS = [2021, 2022, 2023, 2024, 2025]
LAMBDAS = [0, .1, .3, 1, 3, 10, 30, 100, 300, 1000]
DEPLOY = {
    "WR": ["wide", "slot"], "TE": ["inline", "slot", "wide"],
    "OT": ["left_tackle"], "IOL": ["center"],
    "CB": ["cb_slot"], "SAF": ["deep", "box", "dslot"], "LB": ["box", "dslot", "on_line"],
    "EDGE": ["outside_t", "over_t", "on_line"], "DT": ["a_gap", "b_gap", "over_t"],
    "RB": ["route_rate"],
}
OFF = ["QB", "RB", "WR", "TE", "OT", "IOL"]
DEF = ["DT", "EDGE", "LB", "CB", "SAF"]


def ridge_loso(X, y, seasons, free):
    """LOSO RMSE and full-sample coefficients for the best lambda; columns in `free`
    are unpenalised (the unit totals and controls), the rest are shrunk."""
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu) / sd
    pen = np.array([0.0 if i in free else 1.0 for i in range(X.shape[1])])

    def fit(Zt, yt, lam):
        A = np.c_[np.ones(len(Zt)), Zt]
        P = np.diag(np.r_[0.0, pen * lam])
        return np.linalg.solve(A.T @ A + P, A.T @ yt)

    best = None
    for lam in LAMBDAS:
        e = []
        for s in np.unique(seasons):
            tr, te = seasons != s, seasons == s
            b = fit(Z[tr], y[tr], lam)
            e.append((np.c_[np.ones(te.sum()), Z[te]] @ b - y[te]) ** 2)
        rmse = float(np.sqrt(np.concatenate(e).mean()))
        if best is None or rmse < best[1]:
            best = (lam, rmse)
    b = fit(Z, y, best[0])
    return best[1], best[0], b[1:] / sd          # coefficients on the raw scale


def base_rmse(X, y, seasons):
    e = []
    for s in np.unique(seasons):
        tr, te = seasons != s, seasons == s
        A = np.c_[np.ones(tr.sum()), X[tr]]
        b = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        e.append((np.c_[np.ones(te.sum()), X[te]] @ b - y[te]) ** 2)
    return float(np.sqrt(np.concatenate(e).mean()))


def load():
    war = pd.read_csv(ROOT / "war_model" / "hybrid_player_war.csv", dtype={"player_id": str})
    war["group"] = war.position.map(GROUP)
    war = war[war.group.notna() & war.season.isin(SEASONS)]
    dep = pd.concat([DW.deployment(s) for s in SEASONS], ignore_index=True)
    dep["player_id"] = dep.player_id.astype(str)
    return war.merge(dep, on=["season", "player_id"], how="left")


def part2(w, env):
    out = {}
    y_all = U.season_outcomes(SEASONS)
    totals = w.groupby(["season", "team", "group"]).war.sum().unstack(fill_value=0.0)
    for g, ks in DEPLOY.items():
        side = OFF if g in OFF else DEF
        target = U.OUTCOME[g]
        d = w[w.group == g]
        cols = {}
        means = {}
        for k in ks:
            dk = d.dropna(subset=[k])
            means[k] = float(np.average(dk[k], weights=dk.snaps.clip(lower=1)))
            cols[f"D_{k}"] = (dk.war * (dk[k] - means[k])).groupby([dk.season, dk.team]).sum()
        f = totals[side].join(pd.DataFrame(cols)).fillna(0.0).join(env).join(y_all[[target]], how="inner")
        f = f.dropna()
        base_cols = side + ["pass_rate"]
        X0 = f[base_cols].to_numpy()
        X1 = f[base_cols + list(cols)].to_numpy()
        yv, ss = f[target].to_numpy(), f.index.get_level_values(0).to_numpy()
        r0 = base_rmse(X0, yv, ss)
        r1, lam, coef = ridge_loso(X1, yv, ss, free=set(range(len(base_cols))))
        cw = coef[side.index(g)]
        out[g] = {"outcome": target, "n": int(len(f)), "lambda": lam,
                  "rmse_change_pct": round(100 * (r1 / r0 - 1), 3),
                  "war_coef": round(float(cw), 4),
                  "relative_value": {k: round(float(coef[len(base_cols) + i] / cw), 3) if cw else None
                                     for i, k in enumerate(ks)},
                  "mean_share": means}
        print(f"P2 {g:5} {target:9} n={len(f):4} lam={lam:<6} rmse {out[g]['rmse_change_pct']:+.3f}%"
              f"  rel={out[g]['relative_value']}", flush=True)
    return out


def part3(w):
    from scripts.playing_time_backtest import weekly
    out = {}
    rows = []
    for s in [x for x in SEASONS if x >= 2022]:
        wk = weekly(s)
        wk["player_id"] = wk.player_id.astype(str)
        team_weeks = wk[["team", "week"]].drop_duplicates()
        played = wk.groupby(["team", "player_id"]).agg(g_played=("week", "nunique"),
                                                        sn=("snaps", "sum"))
        team_games = team_weeks.groupby("team").size()
        unit_snaps = wk.groupby(["team", "week", "group"]).snaps.sum()
        share = (wk.set_index(["team", "week", "group"]).snaps /
                 unit_snaps.reindex(wk.set_index(["team", "week", "group"]).index).values)
        wk["share"] = share.values
        reg = wk.groupby(["team", "player_id", "group"]).share.mean().reset_index()
        reg = reg[reg.share >= .6 / pd.Series({"QB": 1, "RB": 1, "TE": 1}).reindex(reg.group).fillna(3).values * 1.0]
        ws = w[w.season == s][["player_id", "team", "war", *sum(DEPLOY.values(), [])]]
        reg = reg.merge(ws, on=["player_id", "team"], how="inner")
        reg = reg.merge(played.reset_index(), on=["team", "player_id"])
        reg["war_pg"] = reg.war / reg.g_played
        present = set(zip(wk.team, wk.week, wk.player_id))
        for r in reg.itertuples():
            for wkk in team_weeks[team_weeks.team == r.team].week:
                if (r.team, wkk, r.player_id) not in present:
                    rows.append({"season": s, "team": r.team, "week": wkk, "group": r.group,
                                 "war_pg": r.war_pg,
                                 **{k: getattr(r, k) for k in sum(DEPLOY.values(), [])}})
    miss = pd.DataFrame(rows)
    games = pd.concat([U.games(s) for s in SEASONS if s >= 2022], ignore_index=True)
    for g, ks in DEPLOY.items():
        target = U.OUTCOME[g] + "_adj"
        m = miss[miss.group == g]
        means = {k: float(w[w.group == g][k].mean()) for k in ks}
        agg = m.groupby(["season", "team", "week"]).apply(lambda x: pd.Series({
            "M": x.war_pg.sum(),
            **{f"E_{k}": (x.war_pg * (x[k].fillna(means[k]) - means[k])).sum() for k in ks}}))
        f = games.set_index(["season", "team", "week"])[[target]].join(agg).fillna(0.0)
        f = f[f.index.get_level_values(1).isin(set(w.team))]
        X0 = f[["M"]].to_numpy()
        X1 = f[["M"] + [f"E_{k}" for k in ks]].to_numpy()
        yv, ss = f[target].to_numpy(), f.index.get_level_values(0).to_numpy()
        r0 = base_rmse(X0, yv, ss)
        r1, lam, coef = ridge_loso(X1, yv, ss, free={0})
        out[g] = {"outcome": target, "games": int(len(f)), "games_with_absence": int((f.M > 0).sum()),
                  "lambda": lam, "rmse_change_pct": round(100 * (r1 / r0 - 1), 4),
                  "loss_per_missing_war": round(float(coef[0]), 4),
                  "relative_replacement": {k: round(float(coef[1 + i] / coef[0]), 3) if coef[0] else None
                                           for i, k in enumerate(ks)}}
        print(f"P3 {g:5} absences={out[g]['games_with_absence']:5} lam={lam:<6} "
              f"rmse {out[g]['rmse_change_pct']:+.4f}%  loss/WAR={out[g]['loss_per_missing_war']}"
              f"  rel={out[g]['relative_replacement']}", flush=True)
    return out


def main():
    w = load()
    env = DW.environment(SEASONS).rename("pass_rate").to_frame()
    report = {"part2_value": part2(w, env), "part3_replacement": part3(w)}
    OUT.write_text(json.dumps(report, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
