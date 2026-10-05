"""Opponent difficulty for a team over a window of weeks.

PFF grades and rates are not adjusted for the opponent: a 75 against a weak defence
counts the same as a 75 against a strong one. This module says how hard each team's
schedule was for its offence and for its defence over weeks a..b, so a player's rate
can be adjusted to an average opponent.

Strength comes from CFBD per-game PPA (data/raw/game_advanced_{season}.json for past
seasons, data/live/game_advanced_2026.json for the live one):

  offence faces the opponent's DEFENCE   difficulty = -z(opponent def PPA allowed)
  defence faces the opponent's OFFENCE   difficulty = +z(opponent off PPA)

so higher is always harder. FCS opponents are in the same feed whenever they play an
FBS team, and their poor PPA against FBS opposition makes them easy, which is the
competition-level part of the adjustment. A team's strength is its mean over the games
in the feed, shrunk toward the league mean by SHRINK games so a one-game FCS sample
cannot look elite. For a past season the whole season is used (a schedule fact, not
the player's own outcome); for the live season, games through the cut only.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SHRINK = 2.0
OFFENCE = {"QB", "RB", "WR", "TE", "OT", "IOL"}


@lru_cache(maxsize=None)
def games(season: int) -> pd.DataFrame:
    """One row per (team, game): week, team, opponent, off_ppa, def_ppa."""
    live = ROOT / "data" / "live" / f"game_advanced_{season}.json"
    raw = ROOT / "data" / "raw" / f"game_advanced_{season}.json"
    path = live if live.exists() else raw
    d = json.loads(path.read_text())
    if isinstance(d, dict) and "rows" in d:
        g = pd.DataFrame(d["rows"], columns=d["fields"])
    else:   # the CFBD /stats/game/advanced shape
        g = pd.DataFrame([{"week": r["week"], "team": r["team"], "opponent": r["opponent"],
                           "off_ppa": (r.get("offense") or {}).get("ppa"),
                           "def_ppa": (r.get("defense") or {}).get("ppa")} for r in d])
    g = g[["week", "team", "opponent", "off_ppa", "def_ppa"]].dropna()
    return g.astype({"week": int})


def strength(season: int, through: int | None = None) -> pd.DataFrame:
    """Per team: shrunk mean off and def PPA, z-scored across teams."""
    g = games(season)
    if through is not None:
        g = g[g.week <= through]
    t = g.groupby("team").agg(off=("off_ppa", "mean"), dfn=("def_ppa", "mean"),
                              n=("week", "size"))
    lam = t.n / (t.n + SHRINK)
    for c in ("off", "dfn"):
        mu = np.average(t[c], weights=t.n)
        t[c] = mu + lam * (t[c] - mu)
        t[c] = (t[c] - mu) / t[c].std(ddof=0)
    return t


def window_difficulty(season: int, a: int, b: int,
                      through: int | None = None) -> pd.DataFrame:
    """Per team over weeks a..b: mean difficulty for its offence and its defence."""
    st = strength(season, through)
    g = games(season)
    g = g[(g.week >= a) & (g.week <= b)]
    g = g[g.opponent.isin(st.index)]
    g = g.assign(off_diff=-g.opponent.map(st.dfn), def_diff=g.opponent.map(st.off))
    return g.groupby("team").agg(off_diff=("off_diff", "mean"),
                                 def_diff=("def_diff", "mean"), games=("week", "size"))


def player_difficulty(team: pd.Series, group: pd.Series, diff: pd.DataFrame) -> pd.Series:
    """The difficulty each player's unit faced: offence vs defence by position group."""
    off = team.map(diff.off_diff)
    dfn = team.map(diff.def_diff)
    return pd.Series(np.where(group.isin(OFFENCE), off, dfn), index=team.index).fillna(0.0)
