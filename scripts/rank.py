"""Publish current 2026 v4 power ratings.

Before Week 1 these are preseason ratings. After ``scripts.update_v4`` they blend
the validated preseason prior with completed-game evidence.

Run: python -m scripts.rank
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import ARTIFACTS, PROJECTION_YEAR
from src import v4 as V4
from src.dynamic import WeeklyRatingState, current_power_ratings


def main():
    model_path = ARTIFACTS / "model_v4.json"
    frame_path = ARTIFACTS / f"{PROJECTION_YEAR}_v4_team_frame.csv"
    state_path = ARTIFACTS / f"{PROJECTION_YEAR}_dynamic_state.json"
    if not model_path.exists() or not frame_path.exists():
        raise FileNotFoundError("v4 artifacts are missing; run "
                                "python -m scripts.train_v4 first")

    model = V4.ReciprocalTeamModel.load(model_path)
    frame = pd.read_csv(frame_path, index_col="team")
    if state_path.exists():
        state = WeeklyRatingState.load(state_path)
    else:
        state = WeeklyRatingState.initialize(model, frame, PROJECTION_YEAR)
    ratings = current_power_ratings(model, frame, state)
    out_csv = ARTIFACTS / f"{PROJECTION_YEAR}_power_ratings.csv"
    ratings.to_csv(out_csv, index=False)

    print(f"\n=== {PROJECTION_YEAR} v4 Power Ratings (top 25) ===")
    for r in ratings.head(25).itertuples():
        print(f"  {r.rank:>3}. {r.team:<24} power={r.power:.3f}  "
              f"vs_avg={r.vs_average:.3f}")
    print(f"\nFull table -> {out_csv}")

    first, second = ratings.iloc[0].team, ratings.iloc[1].team
    p = state.predict(model, frame, first, second, neutral_site=True)["p_home"]
    print(f"\nExample neutral site: P({first} over {second})={p:.3f}; "
          f"reverse={1-p:.3f}")


if __name__ == "__main__":
    main()
