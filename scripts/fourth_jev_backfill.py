"""Materialize historical states and optionally evaluate them with Jev.

Examples:
  python -m scripts.fourth_jev_backfill --season 2025 --materialize-only
  python -m scripts.fourth_jev_backfill --season 2025 --live --limit 10
  python -m scripts.fourth_jev_backfill --season 2022 --season 2023 --live

Live calls require TYPESAFE_API_KEY. Results are append-only and resume-safe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fourth_jev.client import JevClient
from fourth_jev.historical import materialize_season
from fourth_jev.ledger import append_jsonl, completed_keys, ledger_key, make_record
from fourth_jev.questions import football_questions


DEFAULT_STATE_DIR = Path("artifacts/fourth_jev/states")
DEFAULT_LEDGER = Path("artifacts/fourth_jev/football_ledger.jsonl")


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def ensure_states(season: int, state_dir: Path, with_pff: bool) -> Path:
    target = state_dir / f"states_{season}.jsonl"
    if not target.exists():
        n = materialize_season(season, target, with_pff=with_pff)
        print(f"materialized {n} games -> {target}")
    else:
        print(f"using existing states -> {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, action="append", required=True)
    parser.add_argument("--live", action="store_true", help="Send calls to Jev")
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-pff", action="store_true")
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args()

    client = JevClient()
    done = completed_keys(args.ledger)
    sent = 0

    for season in args.season:
        state_file = ensure_states(season, args.state_dir, with_pff=not args.no_pff)
        if args.materialize_only:
            continue

        for item in _iter_jsonl(state_file):
            state = item["state"]
            game_id = item["game_id"]
            as_of = item["as_of"]
            key = ledger_key(game_id, as_of, state, client.model, "football")
            if key in done:
                continue

            payload = client.payload(state, football_questions())
            if not args.live:
                print(json.dumps({"ledger_key": key, "payload": payload}, indent=2))
                return

            response = client.evaluate(state, football_questions())
            record = make_record(
                game_id=game_id,
                as_of=as_of,
                state=state,
                model=response.get("model", client.model),
                answers=response.get("answers", {}),
                stage="football",
                metadata={
                    "usage": response.get("usage", {}),
                    "season": season,
                    "outcome": item.get("outcome", {}),
                },
            )
            append_jsonl(args.ledger, record)
            done.add(record["ledger_key"])
            sent += 1
            print(f"{season} {state['game']['away_team']} @ {state['game']['home_team']} -> {record['ledger_key'][:10]}")

            if args.limit is not None and sent >= args.limit:
                print(f"stopped at --limit {args.limit}")
                return

    print(f"completed; new live Jev calls: {sent}")


if __name__ == "__main__":
    main()
