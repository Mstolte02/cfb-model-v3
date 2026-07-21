"""Pulls real data from CollegeFootballData.com (the "CFB Data" source in the
methodology doc). Requires a FREE api key:

    1. Register at https://collegefootballdata.com/key  (takes ~30 seconds)
    2. export CFBD_API_KEY="your_key_here"   (or put it in a .env file)

All responses are cached to data/raw/ as JSON so we only hit the API once.
If no key is present, callers fall back to synthetic data (see load.py).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests

from config import DATA_RAW

BASE = "https://api.collegefootballdata.com"


def _load_dotenv() -> None:
    """Minimal .env loader so we don't add a dependency."""
    env = Path(__file__).resolve().parents[2] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def has_key() -> bool:
    _load_dotenv()
    return bool(os.environ.get("CFBD_API_KEY"))


def _get(endpoint: str, params: dict, cache_name: str) -> list:
    """GET an endpoint with on-disk caching."""
    cache = DATA_RAW / cache_name
    if cache.exists():
        return json.loads(cache.read_text())

    _load_dotenv()
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        raise RuntimeError("CFBD_API_KEY not set; cannot reach the API.")

    resp = requests.get(
        f"{BASE}{endpoint}",
        params=params,
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    cache.write_text(json.dumps(data))
    return data


def advanced_season_stats(year: int) -> list:
    """Per-team advanced stats: success rate, explosiveness, PPA (EPA) for
    offense and defense. Powers the feature vectors (doc §1A / §4.1)."""
    return _get(
        "/stats/season/advanced",
        {"year": year, "excludeGarbageTime": "true"},
        f"advanced_{year}.json",
    )


def games(year: int, season_type: str = "regular") -> list:
    """Completed game results with scores, home/away, neutral-site flag.
    These are the labels for training the logistic regression (doc §4.3)."""
    return _get(
        "/games",
        {"year": year, "seasonType": season_type, "division": "fbs"},
        f"games_{year}.json",
    )


def game_advanced(year: int) -> list:
    """Per-team, per-game advanced stats (for EWMA / recency-weighted ratings)."""
    return _get("/stats/game/advanced", {"year": year, "excludeGarbageTime": "true"},
                f"game_advanced_{year}.json")


def returning_production(year: int) -> list:
    """Bill Connelly-style returning production (percent of PPA returning).
    Computed preseason from the prior roster, so leakage-free for season N."""
    return _get("/player/returning", {"year": year}, f"returning_{year}.json")


def talent(year: int) -> list:
    """Team talent composite ratings (recruiting-based) for a season.
    Known preseason, so usable as a leakage-free prior (doc §B)."""
    return _get("/talent", {"year": year}, f"talent_{year}.json")


def fbs_teams(year: int) -> list:
    return _get("/teams/fbs", {"year": year}, f"teams_{year}.json")
