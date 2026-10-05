"""Deployment-specific production from PFF single-week reports.

A player's alignment mix changes from game to game: a receiver is 80% wide one week
and 50% the next; a corner moves into the slot against 11 personnel. Comparing a
player with himself across those games estimates how much an alignment changes
production for the same player - the difficulty of the assignment - without
confusing it with who gets put there. That slope lets production be judged relative
to its assignment: production adjusted to the league's average alignment mix.

Rows are player-games from source-data/pff_api/war_windows/{season}_wWW-WW (pulled
for 2022-26). Shares are of the alignment snaps PFF reports for that job.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT.parent / "source-data" / "pff_api" / "war_windows"
GROUP = {"WR": "WR", "TE": "TE", "HB": "RB", "FB": "RB", "QB": "QB", "T": "OT", "G": "IOL",
         "C": "IOL", "DI": "DT", "ED": "EDGE", "LB": "LB", "CB": "CB", "S": "SAF"}


def _share(df, cols):
    tot = df[cols].sum(axis=1)
    return df[cols].div(tot.replace(0, np.nan), axis=0)


# job -> (file, production metric, weight column, {share name: alignment columns},
#         higher is better, groups, minimum weight per game)
SPECS = {
    "route":   ("receiving", "yprr", "routes",
                {"wide": ["wide_snaps"], "slot": ["slot_snaps"], "inline": ["inline_snaps"]},
                True, ("WR", "TE"), 5),
    "route_grade": ("receiving", "grades_pass_route", "routes",
                {"wide": ["wide_snaps"], "slot": ["slot_snaps"], "inline": ["inline_snaps"]},
                True, ("WR", "TE"), 5),
    "coverage": ("coverage", "yards_per_coverage_snap", "snap_counts_coverage",
                {"slot": ["def__snap_counts_slot"], "outside": ["def__snap_counts_corner"],
                 "box": ["def__snap_counts_box"], "deep": ["def__snap_counts_fs"]},
                False, ("CB", "SAF", "LB"), 10),
    "coverage_grade": ("coverage", "grades_coverage_defense", "snap_counts_coverage",
                {"slot": ["def__snap_counts_slot"], "outside": ["def__snap_counts_corner"],
                 "box": ["def__snap_counts_box"], "deep": ["def__snap_counts_fs"]},
                True, ("CB", "SAF", "LB"), 10),
    "pass_rush": ("pass_rush", "pass_rush_win_rate", "snap_counts_pass_rush",
                {"a_gap": ["def__snap_counts_dl_a_gap"], "b_gap": ["def__snap_counts_dl_b_gap"],
                 "over_t": ["def__snap_counts_dl_over_t"], "outside_t": ["def__snap_counts_dl_outside_t"],
                 "off_ball": ["def__snap_counts_box"]},
                True, ("DT", "EDGE", "LB"), 8),
    "run_stop": ("run_defense", "stop_percent", "snap_counts_run",
                {"a_gap": ["def__snap_counts_dl_a_gap"], "b_gap": ["def__snap_counts_dl_b_gap"],
                 "over_t": ["def__snap_counts_dl_over_t"], "outside_t": ["def__snap_counts_dl_outside_t"],
                 "box": ["def__snap_counts_box"]},
                True, ("DT", "EDGE", "LB"), 8),
    "pass_pro": ("pass_blocking", "pressure_rate_allowed", "snap_counts_pass_block",
                {"lt": ["snap_counts_lt"], "lg": ["snap_counts_lg"], "c": ["snap_counts_ce"],
                 "rg": ["snap_counts_rg"], "rt": ["snap_counts_rt"]},
                False, ("OT", "IOL"), 10),
}


@lru_cache(maxsize=None)
def _week(season: int, week: int, name: str) -> pd.DataFrame | None:
    p = WINDOWS / f"{season}_w{week:02d}-{week:02d}" / f"{name}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p, low_memory=False)
    d["player_id"] = d.player_id.astype(str)
    return d


def player_games(job: str, seasons, weeks=range(1, 16)) -> pd.DataFrame:
    """Player-game rows for one job: production, weight, alignment shares."""
    file, metric, weight, shares, _, groups, floor = SPECS[job]
    parts = []
    for s in seasons:
        for w in weeks:
            d = _week(s, w, file)
            if d is None or metric not in d or weight not in d:
                continue
            d = d[["player_id", "position", "team_name", metric, weight] +
                  [c for cs in shares.values() for c in cs if c in d.columns]].copy()
            need = [c for cs in shares.values() for c in cs if c.startswith("def__")]
            if need:   # defensive alignment lives in the defense report
                df = _week(s, w, "defense")
                if df is None:
                    continue
                df = df[["player_id"] + [c.split("__", 1)[1] for c in need]]
                df.columns = ["player_id"] + need
                d = d.merge(df, on="player_id", how="left")
            for c in d.columns:
                if c not in ("player_id", "position", "team_name"):
                    d[c] = pd.to_numeric(d[c], errors="coerce")
            d["season"], d["week"] = s, w
            parts.append(d)
    g = pd.concat(parts, ignore_index=True)
    g["group"] = g.position.map(GROUP)
    g = g[g.group.isin(groups) & (g[weight] >= floor) & g[metric].notna()].copy()
    cols = {k: [c for c in cs if c in g.columns] for k, cs in shares.items()}
    tot = sum(g[cs].fillna(0).sum(axis=1) for cs in cols.values())
    for k, cs in cols.items():
        g[f"sh_{k}"] = g[cs].fillna(0).sum(axis=1) / tot.replace(0, np.nan)
    g = g[tot > 0]
    return g.rename(columns={metric: "y", weight: "wt"})[
        ["season", "week", "player_id", "team_name", "group", "y", "wt",
         *[f"sh_{k}" for k in cols]]]


def within_slopes(g: pd.DataFrame, shares: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Within player-season WLS of production on alignment shares (one share dropped as
    the base). Returns (beta, se) for the kept shares."""
    key = [g.season, g.player_id]
    y = g.y - g.groupby(key).y.transform("mean")
    X = np.column_stack([g[s] - g.groupby(key)[s].transform("mean") for s in shares])
    w = g.wt.to_numpy(float)
    sw = np.sqrt(w)
    b, *_ = np.linalg.lstsq(X * sw[:, None], y.to_numpy() * sw, rcond=None)
    r = y.to_numpy() - X @ b
    dof = max(len(y) - len(b) - g.groupby(key).ngroups, 1)
    s2 = float(np.sum(w * r ** 2) / dof)
    cov = np.linalg.pinv((X * w[:, None]).T @ X) * s2
    return b, np.sqrt(np.diag(cov))
