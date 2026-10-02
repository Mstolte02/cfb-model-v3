"""Freeze a current-week, market-blind input slate for Jev."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from fourth_jev.current import export_current, write_current

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    issue_time = datetime.now(timezone.utc)
    rows = export_current(
        ROOT / "viz/data/model_v4.json", ROOT / "viz/data/schedule.json",
        ROOT / "viz/data/teams.json", ROOT / "war_model/team_projections_2026.csv",
        issue_time,
    )
    if not rows:
        raise SystemExit("no future games in the next slate of this published model")
    write_current(rows, args.out)
    print(f"froze {len(rows)} market-blind game states for week {rows[0]['state']['game']['week']} -> {args.out}")


if __name__ == "__main__":
    main()
