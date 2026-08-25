"""Slim play-by-play cache for the decision-profile experiments.

CFBD's ``/plays`` is week-scoped and verbose: a season is ~250k plays carrying play
text, drive ids and conference labels that no decision metric reads. This module
reduces each week to the dozen columns a play-call decision actually depends on -
down, distance, field position, score, clock, timeouts and what the offense did -
and caches one gzipped CSV per season.

The reduction is the reproducible artifact. The fat weekly JSON that ``cfbd_client``
writes on the way through is a convenience cache only; deleting
``data/raw/plays_*.json`` costs API calls on the next rebuild and nothing else.

Play classification note: a sack is a called pass, so it counts as a dropback in
every rate below. Fumble rows carry no play type that distinguishes run from pass,
so they fall back to the play text and are left unclassified when it is silent.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from config import DATA_RAW, GAME_YEARS
from src.data import cfbd_client

# Weeks 1-15 cover the regular season including conference title week. Asking for a
# week a season never played returns an empty list, which is not cached, so the
# upper bound is free to be generous.
WEEKS = range(1, 16)

# CFBD's play-type vocabulary is not stable across seasons. 2025 introduced
# "Pass Completion" alongside "Pass Reception" and "Punt Return" alongside "Punt";
# treating either as unknown silently deleted 1,431 real snaps from the holdout
# season before this was caught. ``audit_play_types`` below fails loudly rather than
# letting the next relabel pass unnoticed.
PASS_TYPES = {
    "Pass Reception", "Pass Completion", "Pass Incompletion", "Passing Touchdown",
    "Sack", "Interception", "Pass Interception Return",
    "Interception Return Touchdown",
}
RUSH_TYPES = {"Rush", "Rushing Touchdown"}
PUNT_TYPES = {
    "Punt", "Punt Return", "Blocked Punt", "Blocked Punt Touchdown",
    "Punt Return Touchdown",
}
FG_TYPES = {
    "Field Goal Good", "Field Goal Missed", "Blocked Field Goal",
    "Missed Field Goal Return", "Blocked Field Goal Touchdown",
}
FUMBLE_TYPES = {
    "Fumble", "Fumble Recovery (Own)", "Fumble Recovery (Opponent)",
    "Fumble Return Touchdown", "Safety", "Uncategorized",
}
# Rows that are not a scrimmage snap by the listed offense at the listed down.
DEAD_TYPES = {
    "Kickoff", "Kickoff Return (Offense)", "Kickoff Return Touchdown", "Timeout",
    "End Period", "End of Half", "End of Game", "End of Regulation", "Penalty",
    "placeholder",
}

_PASS_TEXT = re.compile(r"\bpass(?:es|ed|ing)?\b|\bsack(?:ed)?\b", re.I)
_RUSH_TEXT = re.compile(r"\brun\b|\brush(?:es|ed)?\b|\bkeeper\b|\bscramble", re.I)

SLIM_COLUMNS = [
    "season", "week", "game_id", "offense", "defense", "home", "away",
    "period", "seconds_left", "down", "distance", "yards_to_goal",
    "offense_score", "defense_score", "score_diff",
    "offense_timeouts", "defense_timeouts", "kind", "yards_gained", "ppa",
]


def _kind(play_type: str, text: str) -> str:
    if play_type in PASS_TYPES:
        return "pass"
    if play_type in RUSH_TYPES:
        return "rush"
    if play_type in PUNT_TYPES:
        return "punt"
    if play_type in FG_TYPES:
        return "fg"
    if play_type in FUMBLE_TYPES:
        text = text or ""
        if _PASS_TEXT.search(text):
            return "pass"
        if _RUSH_TEXT.search(text):
            return "rush"
    return "other"


def _seconds_left(period, clock: dict) -> float:
    """Seconds remaining in regulation. Overtime is pinned at zero."""
    clock = clock or {}
    period = int(period or 0)
    within = float((clock.get("minutes") or 0) * 60 + (clock.get("seconds") or 0))
    if period < 1 or period > 4:
        return 0.0
    return float((4 - period) * 900 + within)


def _reduce(raw: list[dict], year: int, week: int) -> pd.DataFrame:
    rows = []
    for p in raw:
        play_type = str(p.get("playType") or "")
        if play_type in DEAD_TYPES:
            continue
        down = p.get("down")
        if down is None or not 1 <= int(down) <= 4:
            continue
        offense_score = float(p.get("offenseScore") or 0)
        defense_score = float(p.get("defenseScore") or 0)
        rows.append({
            "season": int(year), "week": int(week),
            "game_id": int(p.get("gameId") or 0),
            "offense": p.get("offense"), "defense": p.get("defense"),
            "home": p.get("home"), "away": p.get("away"),
            "period": int(p.get("period") or 0),
            "seconds_left": _seconds_left(p.get("period"), p.get("clock")),
            "down": int(down),
            "distance": float(p.get("distance") if p.get("distance") is not None
                              else np.nan),
            "yards_to_goal": float(p.get("yardsToGoal")
                                   if p.get("yardsToGoal") is not None else np.nan),
            "offense_score": offense_score, "defense_score": defense_score,
            "score_diff": offense_score - defense_score,
            "offense_timeouts": float(p.get("offenseTimeouts")
                                      if p.get("offenseTimeouts") is not None
                                      else np.nan),
            "defense_timeouts": float(p.get("defenseTimeouts")
                                      if p.get("defenseTimeouts") is not None
                                      else np.nan),
            "kind": _kind(play_type, p.get("playText")),
            "yards_gained": float(p.get("yardsGained")
                                  if p.get("yardsGained") is not None else np.nan),
            "ppa": float(p.get("ppa")) if p.get("ppa") is not None else np.nan,
        })
    return pd.DataFrame(rows, columns=SLIM_COLUMNS)


def _path(year: int):
    return DATA_RAW / f"plays_slim_{year}.csv.gz"


def build(year: int, refresh=False) -> pd.DataFrame:
    """Return the slim play frame for one season, building the cache if needed."""
    path = _path(year)
    if path.exists() and not refresh:
        return pd.read_csv(path)
    weeks = []
    for week in WEEKS:
        raw = cfbd_client.plays(year, week)
        if not raw:
            continue
        weeks.append(_reduce(raw, year, week))
    if not weeks:
        raise RuntimeError(f"CFBD returned no plays for {year}")
    frame = pd.concat(weeks, ignore_index=True)
    frame.to_csv(path, index=False, compression="gzip")
    return frame


def load(years=GAME_YEARS) -> pd.DataFrame:
    """Slim plays for several seasons, concatenated."""
    return pd.concat([build(year) for year in years], ignore_index=True)


def scrimmage(frame: pd.DataFrame) -> pd.DataFrame:
    """Run/pass snaps only - the denominator for any play-call rate."""
    return frame[frame.kind.isin(("pass", "rush"))]


# A relabel large enough to matter shows up as a jump in the unclassified share.
# 2021-2024 all sit near .0008; 2025 hit .0125 before the vocabulary was fixed.
MAX_UNCLASSIFIED_SHARE = 0.004


def audit_play_types(years=GAME_YEARS) -> pd.DataFrame:
    """Per-season unclassified share, raising when a season looks relabelled."""
    rows = []
    for year in years:
        frame = build(year)
        share = float((frame.kind == "other").mean())
        rows.append({"season": year, "plays": int(len(frame)),
                     "unclassified": int((frame.kind == "other").sum()),
                     "share": share})
    report = pd.DataFrame(rows)
    bad = report[report.share > MAX_UNCLASSIFIED_SHARE]
    if not bad.empty:
        raise RuntimeError(
            "play-type vocabulary drifted; unclassified share exceeds "
            f"{MAX_UNCLASSIFIED_SHARE:.3f} in "
            f"{bad.season.tolist()} -> {bad.share.round(4).tolist()}. "
            "Inspect raw playType values before trusting any decision metric.")
    return report
