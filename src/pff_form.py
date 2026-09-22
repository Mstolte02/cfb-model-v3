"""PFF season-to-date team tables as pregame in-season features.

For a week-W game the tables cover weeks 1..W-1 only
(``source-data/pff_api/team_stats_weekly/{category}_{season}_thru_wNN.csv``, staged by
``scripts/sync_pff_api.py``), so nothing from week W or later is visible. Each weekly
cross-section is standardized on its own, like every other in-season input here.

Three families, all home minus away so they negate when the teams swap:

* **outcome form**: PFF's EPA and success rate for and against - a second
  measurement of the form CFBD's PPA already supplies;
* **process form**: pressure, sacks, tackling and yards before/after contact - how the
  results were produced, which CFBD cannot see;
* **scheme clash**: pass protection against the opponent's pass rush and blocking
  against run defense, each side's unit against the unit it will face.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import PFF_API_DIR
from src.data.pff_team import CATEGORY_PREFIX, TEAM_ALIASES

WEEKLY = Path(PFF_API_DIR) / "team_stats_weekly"
SEASON = Path(PFF_API_DIR) / "team_stats"
KEYS = ["season", "week", "home_team", "away_team"]

OUTCOME = {"off_epa": "pff_off_epa_per_play", "off_sr": "pff_off_success_rate",
           "def_epa": "pff_def_epa_per_play_allowed",
           "def_sr": "pff_def_success_rate_allowed"}
PROCESS = {"press_against": "pff_off_pass_pressure_rate_against",
           "sack_against": "pff_off_pass_sack_rate_against",
           "ybc": "pff_off_rush_yards_before_contact_per_carry",
           "yac": "pff_off_rush_yards_after_contact_per_carry",
           "press": "pff_def_pass_pressure_rate",
           "missed_tackle": "pff_def_missed_tackle_rate",
           "ybc_allowed": "pff_def_rush_yards_before_contact_per_carry_allowed"}

FAMILIES = {
    "pff_outcome_form": ["pff_off_epa_diff", "pff_def_epa_diff",
                         "pff_off_sr_diff", "pff_def_sr_diff"],
    "pff_process_form": ["pff_press_against_diff", "pff_sack_against_diff",
                         "pff_ybc_diff", "pff_yac_diff", "pff_press_diff",
                         "pff_missed_tackle_diff", "pff_ybc_allowed_diff"],
    "pff_scheme_clash": ["pff_pass_pro_clash", "pff_run_block_clash"],
    # One offence and one defence number: the mean of the EPA and success-rate z's.
    # The two correlate .8-.9, so four separate columns mostly trade weight
    # between near-duplicates.
    "pff_outcome_composite": ["pff_O_diff", "pff_D_diff"],
}


def _directory(season: int) -> pd.Series:
    d = pd.read_csv(SEASON / f"team_directory_{season}.csv", low_memory=False)
    d["team"] = d.city.replace(TEAM_ALIASES)
    return (d.dropna(subset=["franchise_id", "team"]).drop_duplicates("franchise_id")
            .set_index("franchise_id").team)


def load_through(season: int, thru: int, teams=None) -> pd.DataFrame | None:
    """Standardized team metrics for weeks 1..thru, or None if not fully staged.

    With ``teams`` the table is cut to that universe BEFORE standardizing, so the
    FCS schools PFF also charts do not compress the FBS spread."""
    paths = {c: WEEKLY / f"{c}_{season}_thru_w{thru:02d}.csv" for c in CATEGORY_PREFIX}
    if not all(p.exists() for p in paths.values()):
        return None
    ids = _directory(season)
    merged = None
    for category, path in paths.items():
        raw = pd.read_csv(path, low_memory=False)
        if raw.empty or "team_id" not in raw:
            return None
        raw["team"] = raw.team_id.map(ids)
        raw = raw.dropna(subset=["team"]).drop_duplicates("team")
        numeric = [c for c in raw.select_dtypes(include=[np.number]).columns
                   if c != "team_id" and not c.endswith("_rank")]
        part = raw.set_index("team")[numeric].rename(
            columns=lambda c: f"{CATEGORY_PREFIX[category]}_{c}")
        merged = part if merged is None else merged.join(part, how="outer")
    if teams is not None:
        merged = merged[merged.index.isin(set(teams))]
    scale = merged.std(ddof=0).replace(0.0, np.nan)
    return (merged - merged.mean()) / scale


def _value(table, team, column) -> float:
    if table is None or team not in table.index or column not in table.columns:
        return 0.0
    v = table.at[team, column]
    return 0.0 if pd.isna(v) else float(v)


def build_pff_form(games: pd.DataFrame, fbs_only: bool = True,
                   _leak_placebo: bool = False) -> pd.DataFrame:
    """``_leak_placebo`` reads the table INCLUDING the game's own week. It exists only
    to show what a leak would look like, and must never feed a real prediction."""
    universe = {}
    for season, g in games.groupby("season"):
        universe[int(season)] = set(g.home_team) | set(g.away_team)
    cache: dict = {}
    rows = []
    for g in games[KEYS].drop_duplicates().itertuples(index=False):
        season, week = int(g.season), int(g.week)
        cut = week if _leak_placebo else week - 1
        key = (season, cut)
        if key not in cache:
            cache[key] = (load_through(season, cut,
                                       universe[season] if fbs_only else None)
                          if cut >= 1 else None)
        t = cache[key]
        h, a = g.home_team, g.away_team
        row = {**g._asdict(), "pff_form_available": float(t is not None)}
        for name, col in {**OUTCOME, **PROCESS}.items():
            row[f"pff_{name}_diff"] = _value(t, h, col) - _value(t, a, col)
        row["pff_O_diff"] = (row["pff_off_epa_diff"] + row["pff_off_sr_diff"]) / 2
        row["pff_D_diff"] = -(row["pff_def_epa_diff"] + row["pff_def_sr_diff"]) / 2
        # Each offence's protection against the rush it will face, minus the reverse.
        # Pressure allowed is bad for the offence and pressure generated good for the
        # defence, so a positive clash favours the home side.
        row["pff_pass_pro_clash"] = (
            (-_value(t, h, PROCESS["press_against"]) - _value(t, a, PROCESS["press"]))
            - (-_value(t, a, PROCESS["press_against"]) - _value(t, h, PROCESS["press"])))
        row["pff_run_block_clash"] = (
            (_value(t, h, PROCESS["ybc"]) - _value(t, a, PROCESS["ybc_allowed"]))
            - (_value(t, a, PROCESS["ybc"]) - _value(t, h, PROCESS["ybc_allowed"])))
        rows.append(row)
    return pd.DataFrame(rows)
