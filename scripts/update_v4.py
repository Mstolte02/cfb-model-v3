"""Replay newly completed 2026 games into the production v4 rating state.

Predictions are permanently recorded before each weekly update. Re-running is safe:
processed game keys are ignored rather than counted twice.

Run: python -m scripts.update_v4
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import ARTIFACTS, PROJECTION_YEAR
from src import v4 as V4
from src.data import load
from src.dynamic import WeeklyRatingState, current_power_ratings


MODEL_PATH = ARTIFACTS / "model_v4.json"
FRAME_PATH = ARTIFACTS / f"{PROJECTION_YEAR}_v4_team_frame.csv"
STATE_PATH = ARTIFACTS / f"{PROJECTION_YEAR}_dynamic_state.json"
PREDICTIONS_PATH = ARTIFACTS / f"{PROJECTION_YEAR}_v4_predictions.csv"
RATINGS_PATH = ARTIFACTS / f"{PROJECTION_YEAR}_power_ratings.csv"


def main():
    load.require_key()
    if not MODEL_PATH.exists() or not FRAME_PATH.exists() or not STATE_PATH.exists():
        raise FileNotFoundError("v4 production artifacts are missing; run "
                                "python -m scripts.train_v4 first")
    model = V4.ReciprocalTeamModel.load(MODEL_PATH)
    frame = pd.read_csv(FRAME_PATH, index_col="team")
    state = WeeklyRatingState.load(STATE_PATH)

    # Current-season results must bypass the ordinary immutable training cache.
    # Otherwise the first Week 1 pull would freeze the live model for the year.
    games = load.games(PROJECTION_YEAR, refresh=True)
    completed = games.dropna(subset=["home_points", "away_points"]).copy()
    completed = completed[
        completed.home_team.isin(frame.index) & completed.away_team.isin(frame.index)]
    new = state.replay(model, frame, completed)
    state.save(STATE_PATH)

    if len(new):
        history = (pd.read_csv(PREDICTIONS_PATH)
                   if PREDICTIONS_PATH.exists() else pd.DataFrame())
        history = pd.concat([history, new], ignore_index=True)
        history = history.drop_duplicates(
            ["season", "week", "home_team", "away_team"], keep="first")
        history.to_csv(PREDICTIONS_PATH, index=False)
        brier = float(((history.p_home - history.y) ** 2).mean())
        print(f"Applied {len(new)} new completed games; "
              f"season pregame Brier={brier:.4f} over {len(history)} games.")
    else:
        print("No new completed games to apply.")

    ratings = current_power_ratings(model, frame, state)
    ratings.to_csv(RATINGS_PATH, index=False)
    print(f"State -> {STATE_PATH}")
    print(f"Ratings -> {RATINGS_PATH}")


if __name__ == "__main__":
    main()
