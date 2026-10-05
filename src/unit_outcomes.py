"""Opponent-adjusted, position-appropriate team outcomes per game and per season.

From CFBD per-game advanced stats (data/raw/game_advanced_{season}.json, 2020-25),
four outcomes per team-game: pass offence, rush offence, pass defence, rush defence
(PPA per play; defence signed so higher is better). Each is adjusted for the
opponent by subtracting what that opponent allowed (or produced) in its OTHER games,
so a game against a strong pass defence is not read as a weak passing game.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTCOME = {"WR": "pass_off", "TE": "pass_off", "QB": "pass_off", "OT": "pass_off",
           "IOL": "pass_off", "RB": "rush_off", "CB": "pass_def", "SAF": "pass_def",
           "LB": "pass_def", "EDGE": "pass_def", "DT": "rush_def"}


@lru_cache(maxsize=None)
def games(season: int) -> pd.DataFrame:
    raw = json.loads((ROOT / "data" / "raw" / f"game_advanced_{season}.json").read_text())
    rows = []
    for r in raw:
        o, d = r.get("offense") or {}, r.get("defense") or {}
        rows.append({"season": season, "week": r["week"], "game": r.get("gameId"),
                     "team": r["team"], "opponent": r["opponent"],
                     "pass_off": (o.get("passingPlays") or {}).get("ppa"),
                     "rush_off": (o.get("rushingPlays") or {}).get("ppa"),
                     "pass_def_raw": (d.get("passingPlays") or {}).get("ppa"),
                     "rush_def_raw": (d.get("rushingPlays") or {}).get("ppa")})
    g = pd.DataFrame(rows).dropna(subset=["pass_off", "rush_off", "pass_def_raw", "rush_def_raw"])
    # opponent context: the opponent's mean in its other games (leave this game out)
    for side, opp_col in (("pass_off", "pass_def_raw"), ("rush_off", "rush_def_raw"),
                          ("pass_def_raw", "pass_off"), ("rush_def_raw", "rush_off")):
        tot = g.groupby("team")[opp_col].agg(["sum", "count"])
        s = g.opponent.map(tot["sum"]); n = g.opponent.map(tot["count"])
        # the opponent's value in THIS game is the mirror row; remove it
        mirror = g.set_index(["game", "team"])[opp_col]
        this = pd.Series([mirror.get((gm, op)) for gm, op in zip(g.game, g.opponent)],
                         index=g.index)
        g[f"ctx_{side}"] = (s - this.fillna(0)) / (n - this.notna().astype(int))
    g["pass_off_adj"] = g.pass_off - g.ctx_pass_off
    g["rush_off_adj"] = g.rush_off - g.ctx_rush_off
    g["pass_def_adj"] = -(g.pass_def_raw - g.ctx_pass_def_raw)
    g["rush_def_adj"] = -(g.rush_def_raw - g.ctx_rush_def_raw)
    return g


def season_outcomes(seasons) -> pd.DataFrame:
    g = pd.concat([games(s) for s in seasons], ignore_index=True)
    cols = ["pass_off_adj", "rush_off_adj", "pass_def_adj", "rush_def_adj"]
    return g.groupby(["season", "team"])[cols].mean().rename(
        columns=lambda c: c.replace("_adj", ""))
