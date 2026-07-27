"""Team talent from the player WAR build in ~/Downloads/rb-win-model.

The existing PFF talent signal is a position-weighted average of last year's grades
over this year's roster. WAR is the same idea carried further: it is already
denominated in wins, so depth weighting falls out of snap counts rather than being
imposed, and the relative worth of a corner against a guard comes from a random
forest fitted to team wins instead of a hand-set weight vector. It also carries the
CFBD PPA signal, which measurably adds to the PFF grades on their own.

Leakage discipline matches src/data/pff.py exactly: for season N a team's roster is
the set of players PFF graded at that team in N, and each carries his season N-1 WAR.
Nothing from season N's results enters season N's talent.

2026 has no results to look back on, so it uses the projected WAR from
projections_2026_v2.csv, which is what that model exists to produce.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

WAR_DIR = Path(os.environ.get("WAR_DIR", Path.home() / "Downloads" / "rb-win-model"))
PLAYER_WAR = WAR_DIR / "hybrid_player_war.csv"
PROJECTIONS = WAR_DIR / "projections_2026_v2.csv"

# rb-win-model already emits CFBD-style school names, but a handful of its own
# spellings differ from the CFBD FBS set the model indexes on.
TEAM_FIX = {"Massachusetts": "Massachusetts", "Hawai'i": "Hawai'i",
            "San José State": "San José State", "Miami (OH)": "Miami (OH)"}


def available() -> bool:
    return PLAYER_WAR.exists()


def _load() -> pd.DataFrame:
    w = pd.read_csv(PLAYER_WAR)
    w["team"] = w.team.replace(TEAM_FIX)
    return w


def team_war_by_year() -> dict[int, pd.Series]:
    """{season: Series(team -> summed prior-year WAR of this season's roster)}.

    A player who transferred shows up at his new team carrying his old value, and a
    player who left simply is not on the roster - the same transfer handling the PFF
    talent signal gets, for the same reason.
    """
    w = _load()
    prior = (w.groupby(["season", "player_id"], as_index=False).war.sum()
              .rename(columns={"war": "prior_war"}))
    prior["season"] += 1  # carry season N-1 value forward to season N

    roster = w[["season", "player_id", "team"]].drop_duplicates()
    j = roster.merge(prior, on=["season", "player_id"], how="left")
    j["prior_war"] = j.prior_war.fillna(0.0)

    out = {}
    for season, g in j.groupby("season"):
        s = g.groupby("team").prior_war.sum()
        if len(s) > 50:  # a season with almost no carry-forward is the first year
            out[int(season)] = s
    return out


def projected_team_war(year: int = 2026) -> pd.Series | None:
    """Projected 2026 team WAR, summed over every two-deep slot."""
    if not PROJECTIONS.exists():
        return None
    p = pd.read_csv(PROJECTIONS)
    p["team"] = p.team.replace(TEAM_FIX)
    return p.groupby("team").proj_war.sum()


def talent_by_year(index_by_year: dict[int, pd.Index],
                   projection_year: int = 2026) -> dict[int, pd.Series]:
    """WAR talent standardized within season, on the same index as the other signals.

    The model's other talent inputs are z-scores within season; WAR is on a wins
    scale, so it is standardized the same way before it can be swapped in.
    """
    raw = team_war_by_year()
    proj = projected_team_war(projection_year)
    if proj is not None:
        raw[projection_year] = proj

    out = {}
    for year, idx in index_by_year.items():
        s = raw.get(year)
        if s is None:
            continue
        s = s.reindex(idx)
        mu, sd = s.mean(), s.std(ddof=0)
        out[year] = (s - mu) / sd if sd and np.isfinite(sd) else s * 0.0
    return out


def player_contributions(year: int = 2026) -> pd.DataFrame | None:
    """Per-player projected WAR, for the team breakdown view."""
    if not PROJECTIONS.exists():
        return None
    p = pd.read_csv(PROJECTIONS)
    p["team"] = p.team.replace(TEAM_FIX)
    cols = ["team", "player", "broad_group", "roster_position", "depth", "class",
            "is_starter", "is_transfer", "proj_war", "imputed", "stars",
            "snaps_2025", "war_2025"]
    return p[[c for c in cols if c in p.columns]].copy()
