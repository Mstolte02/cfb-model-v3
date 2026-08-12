"""Team talent from the player WAR build in war_model/.

The existing PFF talent signal is a position-weighted average of last year's grades
over this year's roster. WAR is the same idea carried further: it is already
denominated in wins, so depth weighting falls out of snap counts rather than being
imposed, and the relative worth of a corner against a guard comes from a random
forest fitted to team wins instead of a hand-set weight vector. It also carries the
CFBD PPA signal, which measurably adds to the PFF grades on their own.

Leakage discipline matches src/data/pff.py exactly: for season N a team's roster is
the set of players PFF graded at that team in N, and each carries his season N-1 WAR.
Nothing from season N's results enters season N's talent.

2026 has no results to look back on, so it uses the projected WAR from
projections_2026_v2.csv, which is what that model exists to produce.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# The build used to live in ~/Downloads/rb-win-model and is now war_model/ in this
# repo, so the two halves of the model version together. WAR_DIR still overrides, for
# pointing at a build made somewhere else.
_HERE = Path(__file__).resolve().parents[2]
WAR_DIR = Path(os.environ.get("WAR_DIR", _HERE / "war_model"))
PLAYER_WAR = WAR_DIR / "hybrid_player_war.csv"

# WHICH OPINION OF EACH PLAYER THE PROJECTION USES.
#
# One build. Our WAR, with EA's ORDERING substituted for anyone under EA_BLEND_SNAPS
# prior snaps, and the starting quarterbacks re-ordered by the five-source composite
# in qbs_2026.xlsx (PFSN, PFF, EPA, an execs poll, EA). Every proven player is on our
# own number; EA only ranks players we cannot rank ourselves.
#
# THERE USED TO BE A SECOND, PFF-ONLY BUILD selected by CFB_WAR_VARIANT and shipped
# beside this one behind a header toggle. It is gone, along with the toggle, the
# `_pff` output suffixes and the environment-variable guard that kept the two in step.
# One set of numbers is the answer, so there is nothing to choose between.


def _projection_file():
    """The final 2026 projection.

    depth_correction.py is the last stage: it fixes the depth chart against prior
    snaps, strips the injury-driven share the projection hands a backup, and applies
    the quarterback value map. The fallbacks below are for a partially built tree.
    """
    try:
        from config import EA_BLEND_SNAPS
    except Exception:
        EA_BLEND_SNAPS = 0
    final = WAR_DIR / "projections_2026_final.csv"
    if final.exists():
        return final
    if EA_BLEND_SNAPS:
        blended = WAR_DIR / "projections_2026_blended.csv"
        if blended.exists():
            return blended
        print(f"  [warn] EA_BLEND_SNAPS={EA_BLEND_SNAPS} but {blended.name} is absent; "
              f"using the unblended projection.")
    return WAR_DIR / "projections_2026_v2.csv"


PROJECTIONS = _projection_file()

# the WAR build already emits CFBD-style school names, but a handful of its own
# spellings differ from the CFBD FBS set the model indexes on.
TEAM_FIX = {"Massachusetts": "Massachusetts", "Hawai'i": "Hawai'i",
            "San José State": "San José State", "Miami (OH)": "Miami (OH)"}


def available() -> bool:
    return PLAYER_WAR.exists()


def _load() -> pd.DataFrame:
    w = pd.read_csv(PLAYER_WAR)
    w["team"] = w.team.replace(TEAM_FIX)
    return w


def team_war_by_year() -> dict[int, pd.Series]:
    """{season: Series(team -> summed prior-year WAR of this season's roster)}.

    A player who transferred shows up at his new team carrying his old value, and a
    player who left simply is not on the roster - the same transfer handling the PFF
    talent signal gets, for the same reason.
    """
    w = _load()
    prior = (w.groupby(["season", "player_id"], as_index=False).war.sum()
              .rename(columns={"war": "prior_war"}))
    prior["season"] += 1  # carry season N-1 value forward to season N

    roster = w[["season", "player_id", "team"]].drop_duplicates()
    j = roster.merge(prior, on=["season", "player_id"], how="left")
    j["prior_war"] = j.prior_war.fillna(0.0)

    out = {}
    for season, g in j.groupby("season"):
        s = g.groupby("team").prior_war.sum()
        if len(s) > 50:  # a season with almost no carry-forward is the first year
            out[int(season)] = s
    return out


def lagged_team_talent(index_by_year: dict[int, pd.Index] | None = None) -> dict[int, pd.Series]:
    """Leakage-safe prior-team WAR, keyed by the season being entered.

    ``team_war_by_year`` uses the season-N participant table as the N roster and
    attaches N-1 WAR.  That is transfer-aware retrospectively, but not preseason-safe:
    the completed season reveals who played and where.  Here each team's realized WAR
    is summed in N-1 and carried forward to N without consulting any N rows.

    When ``index_by_year`` is provided, values are aligned to the other preseason
    features. Missing teams remain missing so the caller can make the imputation
    policy explicit (the v4 team frame uses neutral zero plus a coverage flag).
    """
    w = _load()
    raw = {int(season) + 1: g.groupby("team").war.sum()
           for season, g in w.groupby("season")}
    out = {}
    years = raw if index_by_year is None else index_by_year
    for year in years:
        s = raw.get(int(year))
        if s is None:
            continue
        if index_by_year is not None:
            s = s.reindex(index_by_year[year])
        mu, sd = s.mean(), s.std(ddof=0)
        out[int(year)] = (s - mu) / sd if sd and np.isfinite(sd) else s * 0.0
    return out


def projected_team_war(year: int = 2026) -> pd.Series | None:
    """Projected 2026 team WAR, summed over every two-deep slot."""
    if not PROJECTIONS.exists():
        return None
    p = pd.read_csv(PROJECTIONS)
    p["team"] = p.team.replace(TEAM_FIX)
    return p.groupby("team").proj_war.sum()


def talent_by_year(index_by_year: dict[int, pd.Index],
                   projection_year: int = 2026) -> dict[int, pd.Series]:
    """WAR talent standardized within season, on the same index as the other signals.

    The model's other talent inputs are z-scores within season; WAR is on a wins
    scale, so it is standardized the same way before it can be swapped in.
    """
    raw = team_war_by_year()
    proj = projected_team_war(projection_year)
    if proj is not None:
        raw[projection_year] = proj

    out = {}
    for year, idx in index_by_year.items():
        s = raw.get(year)
        if s is None:
            continue
        s = s.reindex(idx)
        mu, sd = s.mean(), s.std(ddof=0)
        out[year] = (s - mu) / sd if sd and np.isfinite(sd) else s * 0.0
    return out


def player_contributions(year: int = 2026) -> pd.DataFrame | None:
    """Per-player projected WAR, for the team breakdown view."""
    if not PROJECTIONS.exists():
        return None
    p = pd.read_csv(PROJECTIONS)
    p["team"] = p.team.replace(TEAM_FIX)
    # `available` is whitelisted alongside is_starter because the two answer different
    # questions on the team page: is_starter says he is in the lineup, available says
    # whether he is playing at all. A man who is out has proj_war 0 by construction, and
    # without the flag the page shows a listed first-teamer at 0.000 with no explanation.
    # `redshirt` rides along with `class` because the two are one fact split in half:
    # the chart says "RS SR", and the player ratings tab filters on the class and the
    # redshirt independently, so it needs both halves. Leaving it off this list is a
    # silent failure rather than a loud one - the column exists all the way through
    # projections_2026_final.csv and simply never arrives, so every player renders as a
    # non-redshirt and the filter returns a confidently wrong answer.
    cols = ["team", "player", "broad_group", "roster_position", "depth", "class",
            "class_source",
            "redshirt", "is_starter", "available", "is_transfer", "proj_war",
            "imputed", "stars", "snaps_2025", "war_2025"]
    return p[[c for c in cols if c in p.columns]].copy()


TALENT_NOISE = WAR_DIR / "talent_noise.json"
PRESEASON_TEAM_WAR = WAR_DIR / "preseason_team_war.csv"


def projected_team_talent(index_by_year: dict[int, pd.Index],
                          projection_year: int = 2026) -> dict[int, pd.Series]:
    """All-roster ex-ante player projections, standardized within season.

    Historical rows come from ``preseason_team_projection.py``: every roster member
    is projected from earlier seasons and the top K per position room are selected by
    the projection, never by target-season snaps.  The live projection uses the
    actual published two-deep, which is stronger information available in preseason.
    """
    raw = {}
    if PRESEASON_TEAM_WAR.exists():
        d = pd.read_csv(PRESEASON_TEAM_WAR)
        raw.update({int(y): g.set_index("team").projected_war
                    for y, g in d.groupby("season")})
    live = projected_team_war(projection_year)
    if live is not None:
        raw[int(projection_year)] = live
    out = {}
    for year, idx in index_by_year.items():
        s = raw.get(int(year))
        if s is None:
            continue
        s = s.reindex(idx)
        mu, sd = s.mean(), s.std(ddof=0)
        out[int(year)] = (s - mu) / sd if sd and np.isfinite(sd) else s * 0.0
    return out


def talent_noise_sd() -> float:
    """How much extra noise the 2026 talent feature carries, in the z units the model
    consumes. 0.0 when the WAR build has not measured it, so callers fall back to a
    deterministic simulation rather than fail.

    Deliberately a SCALAR. the WAR build also emits a per-team uncertainty file, and it
    should not be used: the team-to-team spread in it fails validation (correlation
    with how far a team's talent estimate actually missed is +0.01). See
    uncertainty.talent_noise() for the measurement.
    """
    if not TALENT_NOISE.exists():
        return 0.0
    return float(json.loads(TALENT_NOISE.read_text()).get("talent_noise_sd", 0.0))
