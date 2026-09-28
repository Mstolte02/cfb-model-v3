"""Canonical state builders for Fourth & Jev."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "0.1.0"


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def build_game_state(*, game: dict, home: dict, away: dict, model: dict | None = None, context: dict | None = None, sources: dict | None = None, as_of: str | None = None) -> dict:
    state = {
        "schema": "fourth-and-jev.game-state",
        "schema_version": SCHEMA_VERSION,
        "as_of": as_of or datetime.now(timezone.utc).isoformat(),
        "game": game,
        "home": home,
        "away": away,
        "existing_model": model or {},
        "context": context or {},
        "source_manifest": sources or {},
        "contract": {"pregame_only": True, "market_blind": True, "later_information_forbidden": True},
    }
    return _clean(state)


def build_market_state(football_state: dict, football_answers: dict, market: dict) -> dict:
    return _clean({
        "schema": "fourth-and-jev.market-state",
        "schema_version": SCHEMA_VERSION,
        "football_state": football_state,
        "football_answers": football_answers,
        "market": market,
        "contract": {"football_forecast_generated_without_market": True},
    })
