"""The FBS membership set, per season.

This project models FBS football. Nothing else belongs in a rating, a rate, a league
mean or a standardisation, and non-FBS rows leak in more easily than they look:

* CFBD's `classification=fbs` filters *games an FBS team played*, not participants, so
  every FBS-versus-FCS game arrives with an FCS team attached to it.
* Anything that groups by team - a per-team pace, a scoring level, a play-call profile
  - will silently mint a row for that FCS visitor.
* Anything that centres or standardises across teams then computes its mean and SD
  over a population that is part FCS, which moves every FBS team's z-score.

The last one is the dangerous one because it produces no obviously wrong row. A
worked example: the per-team home-field study estimated Susquehanna and CSU Pueblo at
plus and minus twelve points of home advantage off a handful of road games, and those
estimates were inside the league mean that all 136 real teams were shrunk toward.

Membership is per season, not a fixed list. Teams move up - Jacksonville State, Sam
Houston, Kennesaw State and Delaware all did inside the window this repo covers - so
asking "is this team FBS" without a year is the wrong question.
"""
from __future__ import annotations

import functools

import pandas as pd

from config import GAME_YEARS
from src.data import cfbd_client


@functools.lru_cache(maxsize=None)
def teams(year: int) -> frozenset[str]:
    """FBS member schools for one season."""
    return frozenset(t["school"] for t in cfbd_client.fbs_teams(year)
                     if t.get("school"))


@functools.lru_cache(maxsize=None)
def any_year(years: tuple[int, ...] = tuple(GAME_YEARS)) -> frozenset[str]:
    """Schools that were FBS in any of the given seasons.

    Use this only where a season is genuinely unavailable. Prefer `teams(year)`.
    """
    out: set[str] = set()
    for year in years:
        out |= teams(year)
    return frozenset(out)


def filter_frame(frame: pd.DataFrame, year_column: str = "season",
                 *team_columns: str) -> pd.DataFrame:
    """Keep rows whose named team columns are all FBS in that row's season."""
    if not team_columns:
        raise ValueError("name at least one team column")
    keep = pd.Series(True, index=frame.index)
    for year, block in frame.groupby(year_column):
        members = teams(int(year))
        mask = pd.Series(True, index=block.index)
        for column in team_columns:
            mask &= block[column].isin(members)
        keep.loc[block.index] = mask
    return frame[keep]


def filter_games(games: list[dict]) -> list[dict]:
    """Keep only games where CFBD marks both participants FBS.

    Uses the classification the payload carries rather than the membership table, so
    a game is judged by what it was at the time.
    """
    return [g for g in games
            if g.get("homeClassification") == "fbs"
            and g.get("awayClassification") == "fbs"]
