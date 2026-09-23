"""Stage PFF reports for windows of weeks, for the in-season WAR research.

Pulls the ten reports the production WAR build reads - the five v1 facet reports and
the five v2 position-report families - for each (season, window). Files land outside
Git in ``source-data/pff_api/war_windows/{season}_w{a:02d}-{b:02d}/{name}.csv``;
existing files are skipped, so the job resumes where it stopped.

    python -m scripts.sync_pff_war_windows --seasons 2022-2025 --windows 1-3,4-16,1-6,7-16
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PFF_API_DIR
from src.data.pff_api import PffApiError, PffClient, _atomic_csv

WINDOW_DIR = Path(PFF_API_DIR) / "war_windows"
LEGACY = ("passing", "rushing", "receiving", "blocking", "defense")
POSITION = {"pass_blocking": "pass-blocking", "run_blocking": "run-blocking",
            "pass_rush": "pass-rush", "run_defense": "run-defense",
            "coverage": "coverage"}


def window_dir(season: int, a: int, b: int) -> Path:
    return WINDOW_DIR / f"{season}_w{a:02d}-{b:02d}"


def parse_ranges(text: str) -> list[tuple[int, int]]:
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out.append((int(a), int(b or a)))
    return out


def seasons_arg(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def main(seasons, windows, pause=.4):
    client = PffClient()
    made = 0
    for season in seasons:
        for a, b in windows:
            dest = window_dir(season, a, b)
            dest.mkdir(parents=True, exist_ok=True)
            weeks = ",".join(str(w) for w in range(a, b + 1))
            jobs = [(n, lambda n=n: client.facet_report(n, season, week=weeks))
                    for n in LEGACY]
            jobs += [(n, lambda r=r: client.position_report(
                         season, r, week_group="REG", week=a, week_to=b, division="fbs"))
                     for n, r in POSITION.items()]
            for name, call in jobs:
                path = dest / f"{name}.csv"
                if path.exists():
                    continue
                rl = client.rate_limit
                if rl.remaining is not None and rl.remaining < 3 and rl.reset:
                    wait = max(0, rl.reset - time.time()) + 2
                    print(f"  quota low, waiting {wait:.0f}s", flush=True)
                    time.sleep(wait)
                frame = None
                for attempt in range(4):
                    try:
                        frame = call()
                        break
                    except PffApiError as exc:
                        # PFF's upstream returns sporadic 500s on heavy v2 reports;
                        # the client retries only 429/503, so back off here.
                        print(f"  {name} attempt {attempt + 1} failed: {exc}", flush=True)
                        time.sleep(10 * (attempt + 1))
                if frame is None:
                    print(f"  SKIPPED {season} w{a}-{b} {name}", flush=True)
                    continue
                _atomic_csv(frame, path)
                made += 1
                print(f"{season} w{a}-{b} {name}: {len(frame)} rows "
                      f"(remaining {client.rate_limit.remaining})", flush=True)
                time.sleep(pause)
    print(f"done, {made} files written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default="2022-2025")
    ap.add_argument("--windows", default="1-3,4-16,1-6,7-16")
    args = ap.parse_args()
    main(seasons_arg(args.seasons), parse_ranges(args.windows))
