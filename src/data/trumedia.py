"""Loader for the manually-exported TruMedia season stats (data/trumedia_stats.csv).

Provides the metrics CFBD lacks: red-zone TD% (off/def), PFF pressure% (off/def),
field-position margin, 3rd/4th-down conversion, early-down pass rate (style),
blitz rate (style). Names are reconciled to CFBD's canonical team names.

load() -> long DataFrame [team, season, <stats>] with CFBD team names.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from config import ROOT

CSV = ROOT / "data" / "trumedia_stats.csv"

# TruMedia full-name prefixes -> CFBD canonical (only where prefix match fails).
ALIAS = {
    "Hawaii": "Hawai'i",
    "San Jose State": "San José State",
    "Appalachian State": "App State",
    "Connecticut": "UConn",
    "Mississippi Rebels": "Ole Miss",      # vs "Mississippi State ..." (prefix-matched)
    "USF": "South Florida",
}

# Output column -> source column. Percentages are parsed to fractions.
PCT_COLS = {
    "rz_td": "RZTD%", "rz_def_td": "RZDefTD%",
    "press_allowed": "PFFPressured%", "press_gen": "PFFPrsr%",
    "third_conv": "3rd/4thCv%", "early_pass_rate": "1st/2ndPassPlay%",
}


def _pct(x):
    if pd.isna(x):
        return None
    m = re.search(r"(-?\d+\.?\d*)\s*%", str(x))
    return float(m.group(1)) / 100 if m else None


def _cfbd_names():
    # Use the raw CFBD endpoint directly (NOT load.team_stats) to avoid a circular
    # import: load.team_stats now merges TruMedia, which would call back here.
    from src.data import cfbd_client
    from config import STAT_YEARS
    names = set()
    for y in STAT_YEARS:
        names |= {t["team"] for t in cfbd_client.advanced_season_stats(y)}
    return sorted(names, key=len, reverse=True)   # longest first to disambiguate


def _matcher(cfbd_sorted):
    def match(full):
        full = str(full)
        for a, c in ALIAS.items():
            if full.startswith(a):
                return c
        for c in cfbd_sorted:
            if full.startswith(c):
                return c
        return None
    return match


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    out = pd.DataFrame({"team_full": df["team"], "season": df["season"].astype(int)})
    for name, src in PCT_COLS.items():
        out[name] = df[src].map(_pct)
    out["fp_margin"] = pd.to_numeric(df["FPMgn"], errors="coerce")
    out["games"] = pd.to_numeric(df["G"], errors="coerce")
    out["blitz_pg"] = (pd.to_numeric(df["Drpbk5+Rush"], errors="coerce") / out["games"])

    match = _matcher(_cfbd_names())
    out["team"] = out["team_full"].map(match)
    missing = sorted(out.loc[out["team"].isna(), "team_full"].unique())
    if missing:
        raise ValueError(f"Unmapped TruMedia teams: {missing}")
    return out.drop(columns="team_full").drop_duplicates(["team", "season"])


STAT_COLS = list(PCT_COLS) + ["fp_margin", "blitz_pg"]
