"""How fast should a player's playing time move from its prior to this season's snaps?

Mark's request (4 October 2026): in-season WAR must use the snaps a player actually
plays in 2026, not the preseason projection's playing time. This fits the rule.

For a cut c (weeks 1..c played), each player's snaps per team game is estimated as a
Bayesian average of a prior and what he has played so far:

    share_now = (n0 * prior + G * observed) / (n0 + G)

G is the games counted so far and n0 is how many games the prior is worth. n0 is fitted
per position group on EARLIER seasons only and graded on the rest of the season
(weeks c+1..end, snaps per team game), weighted toward players who play.

Two denominators are compared for `observed`:
  team    snaps / team games (a game he missed counts as zero)
  played  snaps / games he appeared in (a missed game is left to the availability
          layer, which is how the live site handles injuries)

The backtest prior is last season's snaps per team game, because no historical
preseason projection exists at the weekly grain. The live prior is the preseason
projection, which is at least as good, so the fitted n0 is if anything too small and
the live rule leans toward 2026 snaps slightly harder than the evidence strictly asks.

    python -m scripts.playing_time_backtest
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
from src.war_window import GROUP  # noqa: E402

WEEKLY = ROOT.parent / "source-data" / "pff_api" / "player_weekly"
OUT = ARTIFACTS / "playing_time_backtest.json"
SEASONS = [2021, 2022, 2023, 2024, 2025]
GRADED = [2023, 2024, 2025]
CUTS = [2, 3, 4, 5, 6, 9]
N0_GRID = np.array([0, .5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 30, 50, 1e6])


def weekly(season: int) -> pd.DataFrame:
    """One row per (player, team, week) with snaps; offense and defense combined."""
    frames = []
    for side, col in (("offense", "snap_counts_total"), ("defense", "snap_counts_defense")):
        for f in sorted(WEEKLY.glob(f"{side}_{season}_w*.csv")):
            d = pd.read_csv(f)
            if col not in d.columns:
                continue
            d = d.rename(columns={col: "snaps"})
            d["week"] = int(f.stem.split("_w")[1])
            frames.append(d[["player_id", "player", "position", "team", "week", "snaps"]])
    w = pd.concat(frames, ignore_index=True)
    w["group"] = w.position.map(GROUP)
    w = w[w.group.notna() & w.team.notna()]
    # A two-way player appears on both sides; keep his busier side's group.
    w = (w.sort_values("snaps", ascending=False)
          .drop_duplicates(["player_id", "week"]).reset_index(drop=True))
    w["season"] = season
    return w


def team_weeks(w: pd.DataFrame) -> pd.DataFrame:
    return w[["team", "week"]].drop_duplicates()


def full_time_per_game(w: pd.DataFrame) -> dict:
    """Median over teams of the busiest player's snaps per team game, per group."""
    tg = team_weeks(w).groupby("team").size()
    tot = w.groupby(["group", "team", "player_id"]).snaps.sum().reset_index()
    tot["pg"] = tot.snaps / tot.team.map(tg)
    return tot.groupby(["group", "team"]).pg.max().groupby(level=0).median().to_dict()


def rows_for(season: int, W: dict, cut: int) -> pd.DataFrame:
    w = W[season]
    tw = team_weeks(w)
    g_team_c = tw[tw.week <= cut].groupby("team").size()
    g_team_r = tw[tw.week > cut].groupby("team").size()
    early = w[w.week <= cut]
    late = w[w.week > cut]
    key = ["player_id", "team", "group"]
    e = early.groupby(key).agg(sn=("snaps", "sum"), gp=("week", "nunique")).reset_index()
    e["G_team"] = e.team.map(g_team_c)
    e["obs_team"] = e.sn / e.G_team
    e["obs_played"] = e.sn / e.gp
    r = late.groupby(["player_id", "team"]).snaps.sum().rename("late_sn").reset_index()
    e = e.merge(r, on=["player_id", "team"], how="left")
    e["late_sn"] = e.late_sn.fillna(0.0)
    e["target"] = e.late_sn / e.team.map(g_team_r)
    prev = W.get(season - 1)
    if prev is not None:
        tg = team_weeks(prev).groupby("team").size()
        p = prev.groupby(["player_id", "team"]).snaps.sum().reset_index()
        p["pg"] = p.snaps / p.team.map(tg)
        # Last season's busiest team for the player (transfers keep their own role).
        p = p.sort_values("pg", ascending=False).drop_duplicates("player_id")
        e = e.merge(p[["player_id", "pg"]].rename(columns={"pg": "prior"}),
                    on="player_id", how="left")
    else:
        e["prior"] = np.nan
    e = e[e.target.notna() & e.G_team.notna()]
    return e.assign(season=season, cut=cut)


def fit_n0(d: pd.DataFrame, obs: str, gcol: str) -> float:
    w = d.target + d[obs] + 1.0          # weight toward players who play
    best, err = None, np.inf
    for n0 in N0_GRID:
        est = (n0 * d.prior + d[gcol] * d[obs]) / (n0 + d[gcol])
        e = np.average((est - d.target) ** 2, weights=w)
        if e < err:
            best, err = float(n0), e
    return best


def mse(d, est):
    w = d.target + d[["obs_team", "obs_played"]].max(axis=1) + 1.0
    return float(np.average((est - d.target) ** 2, weights=w))


def main():
    W = {s: weekly(s) for s in SEASONS}
    rows = pd.concat([rows_for(s, W, c) for s in SEASONS[1:] for c in CUTS],
                     ignore_index=True)
    rows = rows[rows.prior.notna()]        # the fit needs a prior to blend
    rows["gp_c"] = rows.gp
    report = {"cuts": CUTS, "graded": GRADED, "by_group": {}, "pooled": {}}
    pooled = {k: [] for k in ("prior", "obs_team", "obs_played", "blend_team",
                              "blend_played")}
    chosen = {}
    for g, d in rows.groupby("group"):
        res = {}
        for T in GRADED:
            tr, te = d[d.season < T], d[d.season == T]
            for c in CUTS:
                trc, tec = tr[tr.cut == c], te[te.cut == c]
                if len(trc) < 50 or len(tec) < 20:
                    continue
                n_t = fit_n0(trc, "obs_team", "G_team")
                n_p = fit_n0(trc, "obs_played", "gp_c")
                est = {
                    "prior": tec.prior,
                    "obs_team": tec.obs_team,
                    "obs_played": tec.obs_played,
                    "blend_team": (n_t * tec.prior + tec.G_team * tec.obs_team)
                                  / (n_t + tec.G_team),
                    "blend_played": (n_p * tec.prior + tec.gp_c * tec.obs_played)
                                    / (n_p + tec.gp_c),
                }
                for k, v in est.items():
                    e = mse(tec, v)
                    res.setdefault(k, []).append(e * len(tec))
                    pooled[k].append(e * len(tec))
                res.setdefault("n", []).append(len(tec))
        n = sum(res.get("n", [0]))
        if not n:
            continue
        report["by_group"][g] = {k: sum(v) / n for k, v in res.items() if k != "n"}
        report["by_group"][g]["n"] = n
        # The live rule: n0 per cut fitted on every season, for the better denominator.
        chosen[g] = {str(c): {"team": fit_n0(d[d.cut == c], "obs_team", "G_team"),
                              "played": fit_n0(d[d.cut == c], "obs_played", "gp_c")}
                     for c in CUTS}
    tot = sum(r["n"] for r in report["by_group"].values())
    report["pooled"] = {k: sum(v) / tot for k, v in pooled.items()}
    report["chosen_n0"] = chosen
    full = full_time_per_game(W[2025])
    report["full_time_per_game_2025"] = full
    OUT.write_text(json.dumps(report, indent=1))
    base = report["pooled"]["prior"]
    print("pooled weighted MSE (relative to prior only):")
    for k, v in report["pooled"].items():
        print(f"  {k:14} {v:9.2f}  {100 * (v / base - 1):+6.1f}%")
    print("\nby group, blend_team / blend_played vs prior:")
    for g, r in sorted(report["by_group"].items()):
        print(f"  {g:5} n={r['n']:6}  team {100*(r['blend_team']/r['prior']-1):+6.1f}%"
              f"  played {100*(r['blend_played']/r['prior']-1):+6.1f}%"
              f"  obs_team {100*(r['obs_team']/r['prior']-1):+6.1f}%")
    print("\nchosen n0 (games the prior is worth), team denominator:")
    for g, c in sorted(chosen.items()):
        print(f"  {g:5} " + "  ".join(f"c{k}:{v['team']:g}" for k, v in c.items()))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
