"""Build and optionally send a first Fourth & Jev request."""
from __future__ import annotations

import argparse
import json

from fourth_jev.client import JevClient
from fourth_jev.questions import football_questions
from fourth_jev.state import build_game_state


def demo_state() -> dict:
    return build_game_state(
        game={"home_team": "HOME", "away_team": "AWAY", "neutral": False, "week": 1, "season": 2026},
        home={"team_metrics": {}},
        away={"team_metrics": {}},
        model={"v5_home_win_probability": None, "v5_predicted_margin": None},
        context={"note": "Replace demo blocks with repository-derived pregame state."},
        sources={"status": "scaffold"},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Send the request to TypeSafe instead of printing it")
    args = parser.parse_args()
    client = JevClient()
    state = demo_state()
    result = client.evaluate(state, football_questions()) if args.live else client.payload(state, football_questions())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
