"""Team scoring-level and pace inputs for the points model.

`src/spread.py` predicts a side's points from three numbers: the scorer's
opponent-adjusted offense rating, the opponent's defense rating, and a home flag.
That is enough to rank margins - a margin is a difference, and the difference of two
standardised composites carries the signal. It is not enough to predict a **total**,
for two structural reasons:

* **No pace.** Points are roughly efficiency times possessions, and the model has no
  possession term at all. A 60-play team and an 85-play team with identical efficiency
  ratings receive the same points prediction.
* **No level.** `O` and `D` are standardised opponent-adjusted *rate* composites. They
  say a team is 1.4 SD above average, not that it scores 34 a game. The ridge has to
  recover a points level from a z-score, which it can only do on league average.

Both are visible in the published totals: `corr(model_total, actual_total) = .099`
over 2022-25 and `.002` in 2025, with a prediction SD of 5.2 against the market's 7.4.

This module supplies the two missing inputs, lagged so season N sees only N-1 and
earlier. Nothing here touches the margin model, which is not broken.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import DATA_RAW, GAME_YEARS
from src.data import plays as PLAYS


LEVEL_COLUMNS = ["points_for", "points_against"]
PACE_COLUMNS = ["off_plays", "def_plays"]
PROFILE_COLUMNS = [*LEVEL_COLUMNS, *PACE_COLUMNS]
MIN_GAMES = 6
# League fallbacks for a team with no prior season, in the same units as the columns.
DEFAULTS = {"points_for": 28.0, "points_against": 28.0,
            "off_plays": 72.8, "def_plays": 72.8}


def scoring_levels(years=GAME_YEARS) -> pd.DataFrame:
    """Per team-season points scored and allowed per game, from completed games."""
    rows = []
    for year in years:
        path = DATA_RAW / f"games_{year}.json"
        if not path.exists():
            continue
        for game in json.loads(path.read_text()):
            if game.get("homePoints") is None or game.get("awayPoints") is None:
                continue
            if game.get("seasonType") == "postseason":
                continue
            home, away = game.get("homeTeam"), game.get("awayTeam")
            hp, ap = float(game["homePoints"]), float(game["awayPoints"])
            rows.append({"season": year, "team": home, "pf": hp, "pa": ap})
            rows.append({"season": year, "team": away, "pf": ap, "pa": hp})
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["season", "team"]).agg(
        points_for=("pf", "mean"), points_against=("pa", "mean"),
        games=("pf", "size")).reset_index()
    return grouped[grouped.games >= MIN_GAMES]


def pace_levels(plays_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per team-season offensive plays run and defensive plays faced, per game."""
    frame = PLAYS.load() if plays_frame is None else plays_frame
    offense = (frame.groupby(["season", "game_id", "offense"]).size()
               .rename("plays").reset_index()
               .groupby(["season", "offense"])
               .agg(off_plays=("plays", "mean"), off_games=("game_id", "size"))
               .reset_index().rename(columns={"offense": "team"}))
    defense = (frame.groupby(["season", "game_id", "defense"]).size()
               .rename("plays").reset_index()
               .groupby(["season", "defense"])
               .agg(def_plays=("plays", "mean"), def_games=("game_id", "size"))
               .reset_index().rename(columns={"defense": "team"}))
    merged = offense.merge(defense, on=["season", "team"], how="inner")
    return merged[(merged.off_games >= MIN_GAMES) & (merged.def_games >= MIN_GAMES)]


def team_season_profiles(years=GAME_YEARS,
                         plays_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Scoring level and pace together, one row per team-season."""
    return scoring_levels(years).merge(pace_levels(plays_frame),
                                       on=["season", "team"], how="inner")


def lagged_profiles(target_years, profiles: pd.DataFrame | None = None
                    ) -> dict[int, pd.DataFrame]:
    """Season-N level and pace from season N-1 only.

    One prior season rather than a weighted history: scoring level and pace both
    respond quickly to a scheme or staff change, and the margin model already carries
    the multi-season signal through its opponent-adjusted ratings.
    """
    profiles = team_season_profiles() if profiles is None else profiles
    result = {}
    for season in sorted(int(y) for y in target_years):
        prior = profiles[profiles.season == season - 1]
        frame = prior.set_index("team")[PROFILE_COLUMNS] if len(prior) else \
            pd.DataFrame(columns=PROFILE_COLUMNS)
        frame.attrs["source_season"] = season - 1
        result[season] = frame
    return result


def attach(frame: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    """Join lagged level and pace onto a model frame, defaulting to league average."""
    out = frame.copy()
    for column in PROFILE_COLUMNS:
        if column in profile.columns:
            values = profile[column].reindex(out.index).astype(float)
        else:
            values = pd.Series(np.nan, index=out.index)
        out[column] = values.fillna(DEFAULTS[column])
    return out


# The selected specification. `both_interact` won the expanding-fold comparison in
# `scripts.totals_backtest`; the interaction terms exist because points are closer to
# a product of scoring level and possessions than to a sum of them.
POINTS_FEATURES = ["O_scorer", "D_opponent", "home", "pf_scorer", "pa_opponent",
                   "offp_scorer", "defp_opponent", "pf_x_pace", "pa_x_pace"]


def side_row(scorer, opponent, is_home: float) -> dict:
    """One scoring row: the scorer's traits against the opponent's.

    The JS port in `viz/app.js` recomputes this arithmetic, so the two must agree
    term for term. Changing anything here means changing that.
    """
    return {
        "O_scorer": float(scorer.O), "D_opponent": float(opponent.D),
        "home": float(is_home),
        "pf_scorer": float(scorer.points_for),
        "pa_opponent": float(opponent.points_against),
        "offp_scorer": float(scorer.off_plays),
        "defp_opponent": float(opponent.def_plays),
        "pf_x_pace": float(scorer.points_for * scorer.off_plays /
                           DEFAULTS["off_plays"]),
        "pa_x_pace": float(opponent.points_against * opponent.def_plays /
                           DEFAULTS["def_plays"]),
    }


def build_points_rows(frame: pd.DataFrame, games_df: pd.DataFrame,
                      labelled: bool = True):
    """Two rows per game - each team's scoring - plus aligned game metadata."""
    teams = set(frame.index)
    rows, target, meta = [], [], []
    for _, game in games_df.iterrows():
        home, away = game["home_team"], game["away_team"]
        if home not in teams or away not in teams:
            continue
        if labelled and (pd.isna(game["home_points"]) or
                         pd.isna(game["away_points"])):
            continue
        neutral = bool(game.get("neutral_site", False))
        fh, fa = frame.loc[home], frame.loc[away]
        rows.append(side_row(fh, fa, 0.0 if neutral else 1.0))
        rows.append(side_row(fa, fh, 0.0))
        if labelled:
            target.extend([float(game["home_points"]), float(game["away_points"])])
        meta.append({
            "week": game.get("week"), "home_team": home, "away_team": away,
            "neutral_site": neutral,
            "actual_total": (float(game["home_points"]) + float(game["away_points"]))
            if labelled else np.nan})
    return pd.DataFrame(rows), np.asarray(target, float), pd.DataFrame(meta)
