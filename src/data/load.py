"""Unified data access over the CFBD API (fresh pulls, cached to data/raw).

team_stats(year) -> DataFrame: team, season, <each feature in config.FEATURES>
games(year)      -> DataFrame: season, home_team, away_team, home_points,
                               away_points, neutral_site

Requires CFBD_API_KEY (see src/data/cfbd_client.py). No synthetic fallback:
if there's no key/data, callers fail loudly rather than fabricate numbers.
"""
from __future__ import annotations

import pandas as pd

from config import FEATURES, CFBD_FEATURES
from src.data import cfbd_client, trumedia


def require_key() -> None:
    if not cfbd_client.has_key():
        raise RuntimeError(
            "CFBD_API_KEY not set. Add it to cfb-model/.env "
            "(get a free key at https://collegefootballdata.com/key)."
        )


def _flatten_advanced(raw: list, year: int) -> pd.DataFrame:
    """Map CFBD's nested /stats/season/advanced payload to flat features."""
    rows = []
    for t in raw:
        off, deff = t.get("offense") or {}, t.get("defense") or {}
        rows.append({
            "team": t["team"],
            "season": year,
            "off_success_rate": off.get("successRate"),
            "off_rush_ppa": (off.get("rushingPlays") or {}).get("ppa"),
            "off_havoc": (off.get("havoc") or {}).get("total"),
            "def_ppa": deff.get("ppa"),
            "def_line_yds": deff.get("lineYards"),
            "def_havoc": (deff.get("havoc") or {}).get("total"),
        })
    df = pd.DataFrame(rows)
    # Drop only on CFBD-native features; TruMedia is merged after (may be missing).
    df = df.dropna(subset=CFBD_FEATURES)
    return df.drop_duplicates(subset=["team"]).reset_index(drop=True)


# TruMedia stat name -> model feature name.
_TM_RENAME = {"rz_td": "off_rz_td", "rz_def_td": "def_rz_td",
              "press_allowed": "off_press_allowed", "press_gen": "def_press",
              "fp_margin": "fp_margin"}


def team_stats(year: int) -> pd.DataFrame:
    require_key()
    cfbd = _flatten_advanced(cfbd_client.advanced_season_stats(year), year)
    try:
        tm = trumedia.load()
        tm = tm[tm["season"] == year][["team", *_TM_RENAME]].rename(columns=_TM_RENAME)
        cfbd = cfbd.merge(tm, on="team", how="left")
    except Exception as e:                       # TruMedia optional / may be absent
        print(f"  [warn] TruMedia merge skipped for {year}: {e}")
    return cfbd


def game_advanced(year: int) -> pd.DataFrame:
    """Per-team-per-game advanced stats, flattened to the game-available subset."""
    require_key()
    raw = cfbd_client.game_advanced(year)
    rows = []
    for g in raw:
        off, deff = g.get("offense") or {}, g.get("defense") or {}
        rows.append({
            "team": g["team"], "season": year,
            "week": g.get("week"), "season_type": g.get("seasonType", "regular"),
            "opponent": g.get("opponent"),
            "off_ppa": off.get("ppa"),
            "off_pass_ppa": (off.get("passingPlays") or {}).get("ppa"),
            "off_rush_ppa": (off.get("rushingPlays") or {}).get("ppa"),
            "off_success_rate": off.get("successRate"),
            "off_explosiveness": off.get("explosiveness"),
            "def_ppa": deff.get("ppa"),
            "def_success_rate": deff.get("successRate"),
            "def_explosiveness": deff.get("explosiveness"),
        })
    return pd.DataFrame(rows)


def returning_production(year: int) -> pd.DataFrame:
    require_key()
    raw = cfbd_client.returning_production(year)
    df = pd.DataFrame([{"team": r["team"], "season": year,
                        "rp": r.get("percentPPA")} for r in raw]).dropna()
    return df.drop_duplicates(subset=["team"]).reset_index(drop=True)


def talent(year: int) -> pd.DataFrame:
    require_key()
    raw = cfbd_client.talent(year)
    df = pd.DataFrame([{"team": t["team"], "season": year,
                        "talent": float(t["talent"])} for t in raw])
    return df.drop_duplicates(subset=["team"]).reset_index(drop=True)


def games(year: int, refresh=False) -> pd.DataFrame:
    require_key()
    raw = cfbd_client.games(year, refresh=refresh)
    rows = [{
        "season": year,
        "week": g.get("week"),
        "home_team": g["homeTeam"],
        "away_team": g["awayTeam"],
        "home_points": g.get("homePoints"),
        "away_points": g.get("awayPoints"),
        "neutral_site": g.get("neutralSite", False),
    } for g in raw if g.get("completed") and g.get("homePoints") is not None]
    columns = ["season", "week", "home_team", "away_team", "home_points",
               "away_points", "neutral_site"]
    return pd.DataFrame(rows, columns=columns).reset_index(drop=True)
