"""Leakage-safe team features staged from the PFF Developer API.

Every frame entering season N is built from PFF's completed season N-1 table.
Rank columns are discarded because they duplicate the underlying metric and make
historical expansion/contraction of the NCAA population part of the signal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import PFF_API_DIR


CATEGORY_PREFIX = {
    "offense_overall_success": "pff_off",
    "offense_passing": "pff_off_pass",
    "offense_rushing": "pff_off_rush",
    "defense_overall_success": "pff_def",
    "defense_passing": "pff_def_pass",
    "defense_rushing": "pff_def_rush",
    "defense_opponent_tendencies": "pff_opp",
}

# PFF's directory ``city`` is close to CFBD's canonical team name.  Keep the
# exceptions explicit and testable rather than fuzzy-matching licensed data.
TEAM_ALIASES = {
    "Appalachian State": "App State",
    "Connecticut": "UConn",
    "Hawaii": "Hawai'i",
    "Louisiana-Monroe": "UL Monroe",
    "Miami (FL)": "Miami",
    "Mississippi": "Ole Miss",
    "North Carolina State": "NC State",
    "Sam Houston State": "Sam Houston",
    "San Jose State": "San José State",
    "USF": "South Florida",
}

OUTCOME_FEATURES = [
    "pff_off_epa_per_play", "pff_off_success_rate",
    "pff_off_explosive_play_rate", "pff_off_points_per_drive",
    "pff_def_epa_per_play_allowed", "pff_def_success_rate_allowed",
    "pff_def_explosive_play_rate_allowed", "pff_def_points_per_drive_allowed",
]

PROCESS_FEATURES = [
    "pff_off_pass_pressure_rate_against", "pff_off_pass_sack_rate_against",
    "pff_off_rush_yards_after_contact_per_carry",
    "pff_off_rush_yards_before_contact_per_carry",
    "pff_def_pass_pressure_rate", "pff_def_pass_sack_rate",
    "pff_def_missed_tackle_rate",
    "pff_def_rush_yards_after_contact_per_carry_allowed",
    "pff_def_rush_yards_before_contact_per_carry_allowed",
]

STYLE_FEATURES = [
    "pff_off_pass_play_action_rate", "pff_off_pass_screen_rate",
    "pff_off_pass_adot", "pff_off_pass_time_to_throw",
    "pff_off_rush_zone_run_pct", "pff_off_rush_gap_run_pct",
    "pff_def_pass_man_coverage_pct", "pff_def_pass_zone_coverage_pct",
]

ALL_EXPERIMENT_FEATURES = [*OUTCOME_FEATURES, *PROCESS_FEATURES, *STYLE_FEATURES]


def _paths(season: int, root: Path) -> tuple[Path, dict[str, Path]]:
    folder = Path(root) / "team_stats"
    directory = folder / f"team_directory_{season}.csv"
    tables = {category: folder / f"{category}_{season}.csv"
              for category in CATEGORY_PREFIX}
    return directory, tables


def load_season(season: int, root: Path = PFF_API_DIR) -> pd.DataFrame:
    """Return standardized PFF team metrics indexed by CFBD team name."""
    directory_path, table_paths = _paths(season, root)
    missing = [path for path in [directory_path, *table_paths.values()]
               if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "PFF team API cache is incomplete for "
            f"{season}: {', '.join(path.name for path in missing)}. "
            f"Run python -m scripts.sync_pff_api --seasons {season} "
            "--team-stats-only.")

    directory = pd.read_csv(directory_path, low_memory=False)
    directory["team"] = directory.city.replace(TEAM_ALIASES)
    id_to_team = (directory.dropna(subset=["franchise_id", "team"])
                  .drop_duplicates("franchise_id")
                  .set_index("franchise_id").team)

    merged = None
    for category, path in table_paths.items():
        raw = pd.read_csv(path, low_memory=False)
        raw["team"] = raw.team_id.map(id_to_team)
        raw = raw.dropna(subset=["team"]).drop_duplicates("team")
        numeric = [column for column in raw.select_dtypes(include=[np.number]).columns
                   if column != "team_id" and not column.endswith("_rank")]
        prefix = CATEGORY_PREFIX[category]
        part = raw.set_index("team")[numeric].rename(
            columns=lambda column: f"{prefix}_{column}")
        merged = part if merged is None else merged.join(part, how="outer")

    assert merged is not None
    # Standardize independently within each completed season.  Missing values stay
    # missing until attachment, where neutral zero is used and coverage is recorded.
    scale = merged.std(ddof=0).replace(0.0, np.nan)
    return (merged - merged.mean()) / scale


def build_lagged(years, root: Path = PFF_API_DIR) -> dict[int, pd.DataFrame]:
    """Map entering season N to standardized completed-season N-1 features."""
    return {int(year): load_season(int(year) - 1, root=root) for year in years}


def attach(frame: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Attach API features without treating missing licensed data as weak play."""
    out = frame.copy()
    aligned = features.reindex(out.index)
    out[features.columns] = aligned.fillna(0.0)
    out.attrs.update(frame.attrs)
    out.attrs["pff_team_coverage"] = float(aligned.notna().any(axis=1).mean())
    return out
