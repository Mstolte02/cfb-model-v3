"""Team components from leakage-safe player production forecasts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = ROOT / "war_model" / "preseason_player_components.csv"
FEATURES = ["player_prod_off", "player_prod_def", "player_prod_war"]


def by_year(index_by_year: dict[int, pd.Index] | None = None) -> dict[int, pd.DataFrame]:
    """Return season/team components without silently synthesizing missing years."""
    if not COMPONENTS.exists():
        raise FileNotFoundError(
            f"player production components not found at {COMPONENTS}; run "
            "python -m war_model.player_production_forecast"
        )
    raw = pd.read_csv(COMPONENTS)
    missing = [column for column in ["season", "team", *FEATURES]
               if column not in raw]
    if missing:
        raise ValueError(f"player production component file is missing {missing}")
    out = {}
    for season, rows in raw.groupby("season"):
        frame = rows.set_index("team")[FEATURES].astype(float)
        if index_by_year is not None and int(season) in index_by_year:
            frame = frame.reindex(index_by_year[int(season)])
        out[int(season)] = frame
    return out
