"""Evaluate a frozen current-week state file with Jev and append each answer."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from fourth_jev.client import JevClient
from fourth_jev.ledger import append_jsonl, completed_keys, ledger_key, make_record
from fourth_jev.questions import football_questions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    client = JevClient()
    done = completed_keys(args.ledger)
    sent = 0
    with args.states.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            state, as_of = row["state"], row["as_of"]
            key = ledger_key(row["game_id"], as_of, state, client.model)
            if key in done:
                continue
            date = state["game"]["kickoff_date"]
            if date <= datetime.now(timezone.utc).date().isoformat():
                raise SystemExit(f"refusing an in-progress or completed game: {row['game_id']}")
            if not args.live:
                print(json.dumps({"game_id": row["game_id"], "payload": client.payload(state, football_questions())}, indent=2))
                return
            response = client.evaluate(state, football_questions())
            record = make_record(
                game_id=row["game_id"], as_of=as_of, state=state, model=client.model,
                answers=response["answers"],
                metadata={"resolved_model": response.get("model", client.model),
                          "usage": response.get("usage", {}),
                          "requested_at_utc": datetime.now(timezone.utc).isoformat()},
            )
            append_jsonl(args.ledger, record)
            done.add(key)
            sent += 1
            print(f"recorded {row['game_id']} {state['game']['away_team']} @ {state['game']['home_team']}", flush=True)
            if args.limit and sent >= args.limit:
                break
    print(f"new Jev forecasts: {sent}")


if __name__ == "__main__":
    main()
