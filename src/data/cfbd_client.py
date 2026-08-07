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
import time
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
    """GET an endpoint with on-disk caching.

    AN EMPTY RESPONSE IS NOT CACHED, and an empty cache file is not a hit. Asking CFBD
    for a season it has not published yet returns `[]`, and caching that turns a
    "not yet" into a permanent "no": talent_2026.json and rankings_2026.json both sat
    at two bytes for weeks, and the talent one is load-bearing - config.py's
    PROJECTION_TALENT_FALLBACK_YEAR quietly substitutes the 2025 composite whenever
    2026 is absent, so a frozen cache means the fallback never lifts and nothing says
    so. Re-asking costs one request per run for a year that genuinely is not out.
    """
    cache = DATA_RAW / cache_name
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached:
            return cached

    _load_dotenv()
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        raise RuntimeError("CFBD_API_KEY not set; cannot reach the API.")

    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    for attempt in range(6):
        resp = requests.get(f"{BASE}{endpoint}", params=params, headers=headers, timeout=30)
        if resp.status_code == 429:                 # rate limited: back off and retry
            wait = float(resp.headers.get("Retry-After", 2 ** attempt))
            time.sleep(min(wait, 30))
            continue
        resp.raise_for_status()
        data = resp.json()
        if data:
            cache.write_text(json.dumps(data))
        return data
    raise RuntimeError(f"CFBD rate-limited after retries: {endpoint} {params}")


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


# --- Player-level PPA (EPA) + volume, for the QB-value / WAR layer -------------

def player_ppa_season(year: int) -> list:
    """Per-player season PPA (EPA/play + total), all positions. `id` is stable
    across seasons/teams (aging + transfer tracking)."""
    return _get("/ppa/players/season", {"year": year, "excludeGarbageTime": "true"},
                f"ppa_players_season_{year}.json")


def player_ppa_games(year: int, week: int) -> list:
    """Per-player, per-game PPA. Carries `opponent`, so it powers the opponent
    adjustment. Scoped by week (the endpoint 400s without a team/week bound)."""
    return _get("/ppa/players/games",
                {"year": year, "week": week, "excludeGarbageTime": "true"},
                f"ppa_players_games_{year}_wk{week}.json")


def player_season_stats(year: int, category: str) -> list:
    """Per-player season stats for a category (e.g. 'passing' -> ATT/COMPLETIONS/
    YDS/TD/INT). Used for QB volume (dropback proxy)."""
    return _get("/stats/player/season", {"year": year, "category": category},
                f"player_stats_{category}_{year}.json")
