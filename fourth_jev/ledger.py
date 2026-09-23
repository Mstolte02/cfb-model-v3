"""Append-only prediction ledger utilities for Fourth & Jev."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

QUESTION_VERSION = "0.1.0"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def state_hash(state: dict) -> str:
    return hashlib.sha256(canonical_json(state).encode("utf-8")).hexdigest()


def ledger_key(game_id: str | int, as_of: str, state: dict, model: str, stage: str = "football") -> str:
    raw = {"game_id": str(game_id), "as_of": as_of, "state_hash": state_hash(state), "model": model, "stage": stage, "question_version": QUESTION_VERSION}
    return hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest()


def append_jsonl(path: str | Path, row: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(canonical_json(row) + "\n")


def completed_keys(path: str | Path) -> set[str]:
    target = Path(path)
    if not target.exists():
        return set()
    out: set[str] = set()
    with target.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("ledger_key"):
                out.add(row["ledger_key"])
    return out


def make_record(*, game_id: str | int, as_of: str, state: dict, model: str, answers: dict, stage: str = "football", metadata: dict | None = None) -> dict:
    key = ledger_key(game_id, as_of, state, model, stage)
    return {
        "ledger_key": key,
        "stage": stage,
        "question_version": QUESTION_VERSION,
        "game_id": str(game_id),
        "as_of": as_of,
        "jev_model": model,
        "state_hash": state_hash(state),
        "answers": answers,
        "metadata": metadata or {},
    }
