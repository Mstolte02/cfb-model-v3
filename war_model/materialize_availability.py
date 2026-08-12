"""Materialize current availability overrides from the append-only event stream.

Add corrections as new events; never edit history.  Status ``clear`` removes a prior
override for that player.  The generated availability_2026.csv remains the compact
input consumed by depth_correction.py.
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVENTS = HERE / "availability_events_2026.csv"
CURRENT = HERE / "availability_2026.csv"


def current_rows(path=EVENTS) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"event_id", "observed_at", "team", "player", "status", "note"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"availability event stream is missing {sorted(required)}")
    ids = [r["event_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("availability event_id values must be unique")
    rows.sort(key=lambda r: (r["observed_at"], r["event_id"]))
    latest = {(r["team"], r["player"]): r for r in rows}
    return [{"team": r["team"], "player": r["player"], "status": r["status"],
             "note": r["note"]} for r in latest.values() if r["status"] != "clear"]


def main() -> None:
    rows = current_rows()
    with CURRENT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("team", "player", "status", "note"))
        writer.writeheader(); writer.writerows(rows)
    print(f"-> {CURRENT} ({len(rows)} active overrides)")


if __name__ == "__main__":
    main()
