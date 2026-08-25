"""CFBD head-coach records with explicit team-season attribution.

CFBD can return multiple head coaches for one school-season after a firing or interim
appointment. The season is assigned to the coach with the most recorded games. Ties
are deterministic by stable coach id. ``coach_share`` records the selected coach's
games divided by all returned coach-games for that team-season, and
``midseason_change`` preserves the fact that attribution was not unique.
"""
from __future__ import annotations

import pandas as pd

from src.data import cfbd_client


DEFAULT_MIN_YEAR = 2014
DEFAULT_MAX_YEAR = 2025


def _fbs_membership(min_year: int, max_year: int) -> pd.DataFrame:
    rows = []
    for year in range(min_year, max_year + 1):
        for team in cfbd_client.fbs_teams(year):
            rows.append({"season": year, "team_id": int(team["id"]),
                         "team": team["school"]})
    return pd.DataFrame(rows).drop_duplicates(["season", "team_id"])


def load_coach_seasons(min_year: int = DEFAULT_MIN_YEAR,
                       max_year: int = DEFAULT_MAX_YEAR,
                       refresh: bool = False) -> pd.DataFrame:
    """All returned FBS coach-school-season records before dominant attribution."""
    raw = cfbd_client.coaches(min_year, max_year, refresh=refresh)
    rows = []
    for coach in raw:
        coach_id = coach.get("id")
        if coach_id is None:
            raise ValueError("CFBD /coaches row is missing its stable id")
        name = " ".join(filter(None, [coach.get("firstName"),
                                      coach.get("lastName")])).strip()
        for season in coach.get("seasons") or []:
            year = season.get("year")
            team_id = season.get("teamId")
            if year is None or team_id is None or not min_year <= int(year) <= max_year:
                continue
            rows.append({
                "coach_id": int(coach_id), "coach_name": name,
                "first_name": coach.get("firstName"),
                "last_name": coach.get("lastName"),
                "hire_date": coach.get("hireDate"),
                "season": int(year), "team_id": int(team_id),
                "school_raw": season.get("school"),
                "conference": season.get("conference"),
                "games": float(season.get("games") or 0),
                "wins": float(season.get("wins") or 0),
                "losses": float(season.get("losses") or 0),
                "ties": float(season.get("ties") or 0),
                "srs": season.get("srs"),
                "sp_overall": season.get("spOverall"),
                "sp_offense": season.get("spOffense"),
                "sp_defense": season.get("spDefense"),
            })
    if not rows:
        raise RuntimeError("CFBD /coaches returned no season records")
    frame = pd.DataFrame(rows)
    membership = _fbs_membership(min_year, max_year)
    frame = frame.merge(membership, on=["season", "team_id"], how="inner",
                        validate="many_to_one")
    return frame.sort_values(["season", "team", "coach_id"]).reset_index(drop=True)


def dominant_head_coaches(min_year: int = DEFAULT_MIN_YEAR,
                          max_year: int = DEFAULT_MAX_YEAR,
                          refresh: bool = False) -> pd.DataFrame:
    """One leakage-neutral attribution row per matched FBS team-season."""
    raw = load_coach_seasons(min_year, max_year, refresh=refresh)
    keys = ["season", "team_id"]
    raw["coach_candidates"] = raw.groupby(keys).coach_id.transform("nunique")
    raw["coach_games_total"] = raw.groupby(keys).games.transform("sum")
    raw["coach_share"] = np_where_positive(
        raw.coach_games_total, raw.games / raw.coach_games_total, 0.0)
    raw["midseason_change"] = raw.coach_candidates > 1
    selected = (raw.sort_values([*keys, "games", "coach_id"],
                                ascending=[True, True, False, True])
                  .drop_duplicates(keys, keep="first"))
    return selected.reset_index(drop=True)


def preseason_head_coaches(min_year: int = DEFAULT_MIN_YEAR,
                            max_year: int = DEFAULT_MAX_YEAR,
                            refresh: bool = False) -> pd.DataFrame:
    """Reconstruct the coach in place entering each season without game counts.

    When a school has an in-season interim, the prior-season incumbent is retained
    if present. If the incumbent is absent, candidates whose recorded hire date is
    after August 1 are excluded. Ambiguous remaining multi-coach rows are deliberately
    unresolved instead of using season-N games to pick a coach.
    """
    raw = load_coach_seasons(min_year, max_year, refresh=refresh).copy()
    raw["hire_timestamp"] = pd.to_datetime(raw.hire_date, errors="coerce", utc=True)
    dominant = dominant_head_coaches(min_year, max_year, refresh=refresh)
    previous = dominant[["season", "team_id", "coach_id"]].copy()
    previous["season"] += 1
    previous = previous.rename(columns={"coach_id": "prior_coach_id"})
    membership = _fbs_membership(min_year, max_year).merge(
        previous, on=["season", "team_id"], how="left")

    rows = []
    grouped = {(int(season), int(team_id)): group
               for (season, team_id), group in raw.groupby(["season", "team_id"])}
    for row in membership.itertuples(index=False):
        group = grouped.get((int(row.season), int(row.team_id)))
        prior_id = getattr(row, "prior_coach_id", np_nan())
        chosen, rule = None, "unmatched"
        if group is not None and len(group) == 1:
            chosen, rule = group.iloc[0], "only_candidate"
        elif group is not None:
            if pd.notna(prior_id) and int(prior_id) in set(group.coach_id):
                chosen = group[group.coach_id == int(prior_id)].iloc[0]
                rule = "returning_incumbent"
            else:
                cutoff = pd.Timestamp(year=int(row.season), month=8, day=1,
                                      tz="UTC")
                eligible = group[group.hire_timestamp.isna() |
                                 group.hire_timestamp.lt(cutoff)]
                if len(eligible) == 1:
                    chosen, rule = eligible.iloc[0], "pre_august_hire"
                else:
                    rule = "ambiguous_zeroed"
        coach_id = int(chosen.coach_id) if chosen is not None else pd.NA
        coach_name = chosen.coach_name if chosen is not None else None
        resolved = chosen is not None
        known_change = (pd.notna(prior_id) and
                        (not resolved or coach_id != int(prior_id)))
        rows.append({
            "season": int(row.season), "team_id": int(row.team_id),
            "team": row.team, "coach_id": coach_id, "coach_name": coach_name,
            "prior_coach_id": (int(prior_id) if pd.notna(prior_id) else pd.NA),
            "resolved_preseason": bool(resolved), "assignment_rule": rule,
            "hc_change": int(known_change),
        })
    out = pd.DataFrame(rows).sort_values(["team_id", "season"])
    out["coach_id"] = out.coach_id.astype("Int64")
    out["prior_coach_id"] = out.prior_coach_id.astype("Int64")
    tenure = []
    prior_team, prior_coach, run = None, None, 0
    for row in out.itertuples(index=False):
        current = int(row.coach_id) if pd.notna(row.coach_id) else None
        if row.team_id == prior_team and current is not None and current == prior_coach:
            run += 1
        else:
            run = 1 if current is not None else 0
        tenure.append(run)
        prior_team, prior_coach = row.team_id, current
    out["hc_tenure_year"] = np_minimum(tenure, 5)
    out["hc_first_year"] = ((out.hc_tenure_year == 1) &
                            out.resolved_preseason).astype(int)
    return out.reset_index(drop=True)


def np_where_positive(denominator: pd.Series, value: pd.Series,
                      fallback: float) -> pd.Series:
    """Small pandas-only helper that avoids importing numpy in the data loader."""
    out = pd.Series(fallback, index=denominator.index, dtype=float)
    good = denominator > 0
    out.loc[good] = value.loc[good]
    return out


def np_nan():
    return float("nan")


def np_minimum(values, cap: int) -> pd.Series:
    return pd.Series(values, dtype=int).clip(upper=cap)
