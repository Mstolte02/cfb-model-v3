"""Stage single-week PFF offense/defense player reports for the availability signal.

One request per (season, week, side) returns every FBS player's snaps that week, so
2021-25 is about 150 requests. Files land outside Git in
``source-data/pff_api/player_weekly/{side}_{season}_w{WW}.csv`` with a canonical CFBD
``team`` column; existing files are skipped, so the job resumes where it stopped.

    python -m scripts.sync_pff_weekly_players --seasons 2021-2025
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.availability import WEEKLY_DIR
from src.data.pff_api import PffClient, _atomic_csv

SIDES = ("offense", "defense")
KEEP = ["player_id", "player", "position", "franchise_id", "team_name",
        "snap_counts_total", "snap_counts_defense", "player_game_count"]


def seasons_arg(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def main(seasons, weeks=range(1, 16), min_remaining=.2, pause=.5):
    client = PffClient()
    WEEKLY_DIR.mkdir(parents=True, exist_ok=True)
    made = 0
    for season in seasons:
        for week in weeks:
            for side in SIDES:
                dest = WEEKLY_DIR / f"{side}_{season}_w{week:02d}.csv"
                if dest.exists():
                    continue
                frame = client.position_report(season, side, week_group="REG",
                                               week=week, week_to=week,
                                               division="fbs")
                frame = frame[[c for c in KEEP if c in frame.columns]].copy()
                frame["team"] = frame.pop("team_name")
                _atomic_csv(frame, dest)
                made += 1
                rl = client.rate_limit
                print(f"{dest.name}: {len(frame)} players "
                      f"(rate {rl.remaining}/{rl.limit})", flush=True)
                if rl.limit and rl.remaining is not None and \
                        rl.remaining < min_remaining * rl.limit:
                    print("rate-limit reserve reached; stopping", flush=True)
                    return made
                time.sleep(pause)
    return made


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", default="2021-2025")
    args = parser.parse_args()
    print(f"{main(seasons_arg(args.seasons))} files written")
