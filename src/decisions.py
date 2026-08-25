"""Play-call decision profiles: pass rate over expected and fourth-down aggression.

The shipping style feature is TruMedia's ``1st/2ndPassPlay%`` - a raw early-down pass
rate that knows nothing about down, distance, field position, score or clock. It
therefore measures the *situation* at least as much as the *coach*: a team that
trails all year is forced to throw and reads as aggressive, and a team that leads all
year is allowed to run and reads as conservative. Neither is a decision.

This module separates the call from the situation that produced it. A league-wide
model predicts P(pass) and P(go for it) from the game state alone; a team's profile
is its mean residual against that expectation. What survives is the part of the call
the situation does not explain.

Two contracts matter and are enforced by ``lagged_profiles``:

* The expectation model for season N is fit only on plays from seasons <= N-1. A
  model fit on all seasons would let season-N league drift define season-N's own
  baseline.
* A team's profile entering season N is built only from its seasons <= N-1.

Fourth-down decisions are scarce (a few dozen per team-season in the normal-course
zone), so those rates are shrunk toward the league mean by their own count. Early-down
pass rates are measured on hundreds of plays and are left alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data import fbs
from src.data import plays as PLAYS  # noqa: F401  (re-exported for callers)


# A fourth-down choice is only a choice in the normal course of a game. Down three
# scores with a minute left, everyone goes; up four scores, everyone punts. Those
# rows measure the scoreboard, not the coach, so the aggression rate is computed on
# a restricted zone and the excluded count is reported alongside it.
NORMAL_COURSE_MAX_LEAD = 21.0
NORMAL_COURSE_MIN_SECONDS = 300.0
# Empirical-Bayes strength for fourth-down rates, in plays. Roughly the median
# normal-course fourth-down count per team-season, so a team with a typical sample
# is pulled halfway to the league mean.
FOURTH_PRIOR_PLAYS = 25.0

PROFILE_COLUMNS = [
    "proe", "proe_neutral", "proe_leading", "proe_trailing", "proe_late_close",
    "proe_game_sd", "fourth_go_oe", "fourth_kick_oe", "early_down_pass_rate",
]
# Counts travel with the profile so a caller can weight or audit coverage.
COUNT_COLUMNS = ["n_early", "n_fourth", "n_games"]


def _urgency(seconds_left: pd.Series) -> pd.Series:
    """0 at kickoff, 1 at the final whistle. Squared so it bites late."""
    return (1.0 - seconds_left / 3600.0).clip(0.0, 1.0) ** 2


def _situation(frame: pd.DataFrame) -> pd.DataFrame:
    """Design matrix for the expected-call models: game state only, never team."""
    distance = frame.distance.clip(lower=1.0, upper=30.0)
    ytg = frame.yards_to_goal.clip(lower=1.0, upper=99.0)
    urgency = _urgency(frame.seconds_left)
    score = frame.score_diff.clip(-28.0, 28.0)
    design = pd.DataFrame({
        "log_distance": np.log(distance),
        "ytg": ytg / 100.0,
        "goal_to_go": (distance >= ytg).astype(float),
        "own_half": (ytg > 50.0).astype(float),
        "score": score / 10.0,
        "score_urgency": (score / 10.0) * urgency,
        "trailing": (score < 0).astype(float),
        "trailing_urgency": (score < 0).astype(float) * urgency,
        "urgency": urgency,
        "second_half": (frame.period >= 3).astype(float),
        "timeouts": frame.offense_timeouts.fillna(3.0) / 3.0,
    }, index=frame.index)
    return design


def _early_design(frame: pd.DataFrame) -> pd.DataFrame:
    """Early-down design adds the down itself and its distance interaction."""
    design = _situation(frame)
    down2 = (frame.down == 2).astype(float)
    down3 = (frame.down == 3).astype(float)
    design["down2"] = down2
    design["down3"] = down3
    design["down2_distance"] = down2 * design.log_distance
    design["down3_distance"] = down3 * design.log_distance
    return design


def _fit(design: pd.DataFrame, target: pd.Series) -> LogisticRegression:
    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(design.to_numpy(float), target.to_numpy(int))
    return model


def early_downs(frame: pd.DataFrame) -> pd.DataFrame:
    """First through third down run/pass snaps - where tendency is expressed."""
    return frame[(frame.down <= 3) & frame.kind.isin(("pass", "rush"))]


def fourth_downs(frame: pd.DataFrame) -> pd.DataFrame:
    """Fourth downs where a real choice existed."""
    kinds = ("pass", "rush", "punt", "fg")
    zone = frame[(frame.down == 4) & frame.kind.isin(kinds)]
    return zone[(zone.score_diff.abs() <= NORMAL_COURSE_MAX_LEAD) &
                (zone.seconds_left >= NORMAL_COURSE_MIN_SECONDS)]


def fit_expectations(train: pd.DataFrame) -> dict:
    """League expectation models from a pool of completed seasons."""
    early = early_downs(train)
    fourth = fourth_downs(train)
    go = fourth.kind.isin(("pass", "rush"))
    kick = fourth[~go]
    return {
        "pass": _fit(_early_design(early), (early.kind == "pass").astype(int)),
        "go": _fit(_situation(fourth), go.astype(int)),
        # Given a kick, is it a field goal? Separates "aggressive" from "has a
        # kicker and good field position", which the go model alone conflates.
        "fg": _fit(_situation(kick), (kick.kind == "fg").astype(int)),
        "seasons": sorted(int(s) for s in train.season.unique()),
        "n_early": int(len(early)), "n_fourth": int(len(fourth)),
    }


def _residual(model: LogisticRegression, design: pd.DataFrame,
              actual: pd.Series) -> pd.Series:
    expected = model.predict_proba(design.to_numpy(float))[:, 1]
    return pd.Series(actual.to_numpy(float) - expected, index=design.index)


def _shrink(rate: pd.Series, count: pd.Series, prior_plays: float) -> pd.Series:
    """Pull a rate toward zero by its own sample size."""
    weight = count / (count + prior_plays)
    return rate * weight


def team_season_profiles(frame: pd.DataFrame, expectations: dict) -> pd.DataFrame:
    """Decision profile per (season, offense) using the supplied expectations."""
    early = early_downs(frame).copy()
    early["resid"] = _residual(expectations["pass"], _early_design(early),
                               (early.kind == "pass").astype(int))
    early["is_pass"] = (early.kind == "pass").astype(float)
    early["neutral"] = ((early.score_diff.abs() <= 8.0) &
                        (early.period <= 2)).astype(float)
    early["leading"] = (early.score_diff > 8.0).astype(float)
    early["trailing"] = (early.score_diff < -8.0).astype(float)
    early["late_close"] = ((early.score_diff.abs() <= 8.0) &
                           (early.period >= 3)).astype(float)

    fourth = fourth_downs(frame).copy()
    go = fourth.kind.isin(("pass", "rush"))
    fourth["go_resid"] = _residual(expectations["go"], _situation(fourth),
                                   go.astype(int))
    kick = fourth[~go].copy()
    if len(kick):
        kick["fg_resid"] = _residual(expectations["fg"], _situation(kick),
                                     (kick.kind == "fg").astype(int))

    rows = []
    keys = sorted(set(zip(early.season, early.offense)))
    fourth_groups = dict(tuple(fourth.groupby(["season", "offense"])))
    kick_groups = (dict(tuple(kick.groupby(["season", "offense"])))
                   if len(kick) else {})
    for key in keys:
        season, team = key
        block = early[(early.season == season) & (early.offense == team)]
        if len(block) < 100:
            continue
        by_game = block.groupby("game_id").resid.mean()
        fourth_block = fourth_groups.get(key)
        kick_block = kick_groups.get(key)
        n_fourth = float(len(fourth_block)) if fourth_block is not None else 0.0
        n_kick = float(len(kick_block)) if kick_block is not None else 0.0
        rows.append({
            "season": int(season), "team": team,
            "proe": float(block.resid.mean()),
            "proe_neutral": _conditional(block, "neutral"),
            "proe_leading": _conditional(block, "leading"),
            "proe_trailing": _conditional(block, "trailing"),
            "proe_late_close": _conditional(block, "late_close"),
            "proe_game_sd": float(by_game.std(ddof=0)) if len(by_game) > 1 else 0.0,
            "early_down_pass_rate": float(block.is_pass.mean()),
            "fourth_go_oe_raw": (float(fourth_block.go_resid.mean())
                                 if n_fourth else 0.0),
            "fourth_kick_oe_raw": (float(kick_block.fg_resid.mean())
                                   if n_kick else 0.0),
            "n_early": int(len(block)), "n_fourth": int(n_fourth),
            "n_kick": int(n_kick), "n_games": int(block.game_id.nunique()),
        })
    profiles = pd.DataFrame(rows)
    profiles["fourth_go_oe"] = _shrink(profiles.fourth_go_oe_raw,
                                       profiles.n_fourth, FOURTH_PRIOR_PLAYS)
    profiles["fourth_kick_oe"] = _shrink(profiles.fourth_kick_oe_raw,
                                         profiles.n_kick, FOURTH_PRIOR_PLAYS)
    return profiles


def _conditional(block: pd.DataFrame, flag: str) -> float:
    chosen = block[block[flag] > 0]
    # Too few snaps in a state is not evidence of neutrality, but zero is the
    # right neutral default: the residual is already centred on the league.
    return float(chosen.resid.mean()) if len(chosen) >= 30 else 0.0


def _recency_weighted(history: pd.DataFrame, columns, half_life=1.5) -> pd.Series:
    """Collapse a team's or coach's prior seasons, weighting recent ones more."""
    age = history.season.max() - history.season if len(history) else 0
    weight = 0.5 ** (age / half_life)
    weight = weight * np.sqrt(history.n_early.clip(lower=1.0))
    total = float(weight.sum()) or 1.0
    return pd.Series({c: float((history[c] * weight).sum() / total)
                      for c in columns})


def lagged_profiles(target_years, plays_frame: pd.DataFrame | None = None,
                    coach_assignment: pd.DataFrame | None = None
                    ) -> dict[int, pd.DataFrame]:
    """Season-N decision profiles built only from plays in seasons <= N-1.

    When ``coach_assignment`` is supplied the result also carries the incoming head
    coach's own profile, carried across schools. That is the channel the team-level
    lag cannot see: a new coach's tendencies arrive with them, and the team's own
    history describes a staff that has left.
    """
    frame = PLAYS.load() if plays_frame is None else plays_frame
    frame = fbs.filter_frame(frame, "season", "offense", "defense")
    result = {}
    for season in sorted(int(y) for y in target_years):
        history = frame[frame.season < season]
        if history.empty:
            result[season] = pd.DataFrame(columns=["team", *PROFILE_COLUMNS])
            continue
        expectations = fit_expectations(history)
        profiles = team_season_profiles(history, expectations)
        rows = []
        for team, block in profiles.groupby("team"):
            values = _recency_weighted(block, PROFILE_COLUMNS)
            rows.append({"team": team, **values,
                         "decision_seasons": int(len(block)),
                         "n_early": int(block.n_early.sum()),
                         "n_fourth": int(block.n_fourth.sum())})
        current = pd.DataFrame(rows).set_index("team")
        if coach_assignment is not None:
            current = _attach_coach_profiles(current, profiles, coach_assignment,
                                             season)
        current.attrs["expectation_seasons"] = expectations["seasons"]
        current.attrs["max_play_season"] = int(history.season.max())
        result[season] = current
    return result


def _attach_coach_profiles(current: pd.DataFrame, profiles: pd.DataFrame,
                           assignment: pd.DataFrame, season: int) -> pd.DataFrame:
    """Carry the incoming coach's own decision profile across a school change."""
    prior = assignment[assignment.season < season]
    if prior.empty:
        for column in PROFILE_COLUMNS:
            current[f"coach_{column}"] = 0.0
        current["coach_has_decision_history"] = 0.0
        return current
    # Every prior (coach, team, season) the assignment table resolved, joined to the
    # decision profile that coach's team posted that season.
    linked = prior.merge(profiles, on=["season", "team"], how="inner")
    by_coach = {}
    for coach_id, block in linked.groupby("coach_id"):
        by_coach[int(coach_id)] = _recency_weighted(block, PROFILE_COLUMNS)
    now = assignment[assignment.season == season].set_index("team")
    coach_ids = now.coach_id.reindex(current.index)
    for column in PROFILE_COLUMNS:
        current[f"coach_{column}"] = [
            float(by_coach[int(cid)][column])
            if pd.notna(cid) and int(cid) in by_coach else 0.0
            for cid in coach_ids]
    current["coach_has_decision_history"] = [
        1.0 if pd.notna(cid) and int(cid) in by_coach else 0.0
        for cid in coach_ids]
    return current


# Team-level feature names. The v4 architecture differences these between the two
# teams itself, so every column here is a team trait, not a matchup quantity.
TENDENCY = ["dec_proe", "dec_fourth_go"]
FULL = [*TENDENCY, "dec_proe_neutral", "dec_proe_leading", "dec_proe_trailing",
        "dec_proe_late_close", "dec_kick", "dec_game_sd"]
COACH_CARRIED = ["dec_coach_proe", "dec_coach_fourth_go", "dec_coach_has_history"]
INTERACTIONS = ["dec_proe_x_talent", "dec_proe_x_returning", "dec_proe_x_war",
                "dec_fourth_x_talent", "dec_fourth_x_returning"]
ALL_FEATURES = [*FULL, *COACH_CARRIED, *INTERACTIONS]

_SOURCE = {
    "dec_proe": "proe", "dec_fourth_go": "fourth_go_oe",
    "dec_proe_neutral": "proe_neutral", "dec_proe_leading": "proe_leading",
    "dec_proe_trailing": "proe_trailing", "dec_proe_late_close": "proe_late_close",
    "dec_kick": "fourth_kick_oe", "dec_game_sd": "proe_game_sd",
    "dec_coach_proe": "coach_proe", "dec_coach_fourth_go": "coach_fourth_go_oe",
    "dec_coach_has_history": "coach_has_decision_history",
}


def attach_decision_features(frame: pd.DataFrame,
                             profile: pd.DataFrame) -> pd.DataFrame:
    """Join team decision traits and complete the season-N interactions.

    A team with no prior play history gets zero, which is the league mean after the
    residual construction - an unknown tendency, not an extreme one.
    """
    out = frame.copy()
    for name, source in _SOURCE.items():
        if source in profile.columns:
            out[name] = profile[source].reindex(out.index).astype(float)
        else:
            out[name] = 0.0
    out[list(_SOURCE)] = out[list(_SOURCE)].fillna(0.0)
    war = out.war_projected if "war_projected" in out.columns else out.get(
        "war_lag", pd.Series(0.0, index=out.index))
    out["dec_proe_x_talent"] = out.dec_proe * out.talent
    out["dec_proe_x_returning"] = out.dec_proe * out.returning
    out["dec_proe_x_war"] = out.dec_proe * war
    out["dec_fourth_x_talent"] = out.dec_fourth_go * out.talent
    out["dec_fourth_x_returning"] = out.dec_fourth_go * out.returning
    return out


def stability(profiles: pd.DataFrame, columns=PROFILE_COLUMNS) -> pd.DataFrame:
    """Year-over-year correlation of each metric within a team.

    A decision metric is only a coach trait if it repeats. This is the cheapest
    check that separates a tendency from a season of noise, and it runs before any
    predictive claim is worth making.
    """
    rows = []
    ordered = profiles.sort_values(["team", "season"])
    for column in columns:
        pairs = []
        for _, block in ordered.groupby("team"):
            values = block[[column, "season"]].dropna()
            for (_, a), (_, b) in zip(values.iterrows(), values.iloc[1:].iterrows()):
                if b.season == a.season + 1:
                    pairs.append((a[column], b[column]))
        if len(pairs) > 2:
            left, right = zip(*pairs)
            r = float(np.corrcoef(left, right)[0, 1])
        else:
            r = float("nan")
        rows.append({"metric": column, "lag1_r": r, "pairs": len(pairs)})
    return pd.DataFrame(rows)
