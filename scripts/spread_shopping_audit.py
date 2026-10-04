"""Count fresh two-book spread differences; no side prices or bet recommendation."""
from __future__ import annotations

import hashlib
import json

from config import ARTIFACTS
from scripts.capture_market_snapshot import (
    CHECKS, QUOTES, latest_quotes, parse_time, read_jsonl,
)

OUT = ARTIFACTS / "spread_shopping_audit.json"
MAX_HOURS_TO_KICKOFF = 6


def main() -> None:
    events = read_jsonl(QUOTES)
    checks = read_jsonl(CHECKS)
    starts = {int(row["game_id"]): parse_time(row["start"])
              for row in events if row.get("start")}
    latest = {}
    for check in checks:
        checked = parse_time(check["checked_at"])
        in_window = [game_id for game_id, start in starts.items()
                     if checked <= start and
                     (start - checked).total_seconds() <= MAX_HOURS_TO_KICKOFF * 3600]
        if not in_window:
            continue
        quotes = latest_quotes(events, checked)
        for game_id in in_window:
            dk = quotes.get((game_id, "DraftKings"))
            bovada = quotes.get((game_id, "Bovada"))
            if (not dk or not bovada or dk.get("spread") is None or
                    bovada.get("spread") is None):
                continue
            latest[game_id] = {
                "game_id": game_id, "checked_at": check["checked_at"],
                "draftkings_home_spread": float(dk["spread"]),
                "bovada_home_spread": float(bovada["spread"]),
                "line_difference": abs(float(dk["spread"]) - float(bovada["spread"])),
            }
    rows = sorted(latest.values(), key=lambda row: row["game_id"])
    result = {
        "status": "research_only_not_executable_prices",
        "method": "last successful paired CFBD retrieval within six hours before kickoff, one row per game",
        "limitations": ["No spread-side odds, limits, or sportsbook-native quote timestamps.",
                        "A retrieval proves what CFBD returned, not that both books would accept a bet."],
        "input_sha256": {
            "quote_events": hashlib.sha256(QUOTES.read_bytes()).hexdigest(),
            "successful_checks": hashlib.sha256(CHECKS.read_bytes()).hexdigest(),
        },
        "games": len(rows),
        "at_least_one_point": sum(row["line_difference"] >= 1 for row in rows),
        "at_least_two_points": sum(row["line_difference"] >= 2 for row in rows),
        "observations": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print({key: result[key] for key in ("games", "at_least_one_point", "at_least_two_points")})


if __name__ == "__main__":
    main()
