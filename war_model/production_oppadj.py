"""Would opponent-adjusted production replace PFF grades in WAR?

`horse_race.py` already compares the PFF facet set against the CFBD production set and
finds production behind: next-season CV r of .374 against .475. But that comparison is
not the one worth having, because the CFBD facets are **raw**. A quarterback's PPA per
play is credited to him whether he threw it against the best secondary in the league
or the worst, and a schedule is a large part of a season's production.

This builds the fair version. For each facet, every game in the schedule contributes
one observation of the form

    season-mean production of team i  ~  attack[i] - defence[j]

solved as a ridge over all games in the season, so a team that faced hard defences has
its attack term lifted and a team that feasted has its lowered. Each player's grade is
then shifted by his own team's correction. Volume, standardisation and everything
downstream are untouched, so the only thing that changes between the two runs is
whether the production knows who it was earned against.

The comparison is then identical to `horse_race.py` stage 1 and stage 3, on the same
rows, with the same estimator and the same season-blocked folds.

Run: ../venv/bin/python production_oppadj.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict

from cfbd_facets import CFBD_FACETS

HERE = os.path.dirname(os.path.abspath(__file__))
FACETS_PARQUET = f"{HERE}/cfbd_facet_values.parquet"
SCHEDULE = f"{HERE}/schedule.csv"
OUT_JSON = f"{HERE}/production_oppadj.json"
OUT_PARQUET = f"{HERE}/cfbd_facet_values_oppadj.parquet"

# Offensive facets are earned against the opponent's defence and vice versa. The sign
# of the correction flips with the side, which is the whole point of the exercise.
DEF_FACETS = {"havoc_dl", "havoc_lb", "havoc_db", "tackle_lb", "tackle_db", "cov_db"}
RIDGE_ALPHA = 25.0


def team_season_means(values: pd.DataFrame) -> pd.DataFrame:
    """Snap-weighted mean grade per (season, team, facet)."""
    v = values[values.snaps > 0].copy()
    v["w"] = v.grade * v.snaps
    grouped = v.groupby(["season", "cfbd_team", "facet"]).agg(
        num=("w", "sum"), den=("snaps", "sum")).reset_index()
    grouped["mean_grade"] = grouped.num / grouped.den
    return grouped[["season", "cfbd_team", "facet", "mean_grade"]]


def schedule_pairs(schedule: pd.DataFrame) -> pd.DataFrame:
    """Two directed rows per game: (team, opponent)."""
    home = schedule.rename(columns={"home_team": "team", "away_team": "opponent"})
    away = schedule.rename(columns={"away_team": "team", "home_team": "opponent"})
    both = pd.concat([home[["season", "team", "opponent"]],
                      away[["season", "team", "opponent"]]], ignore_index=True)
    return both.dropna()


def solve_attack_defence(rows: pd.DataFrame, teams: list[str]) -> dict[str, float]:
    """Ridge on y ~ attack[team] - defence[opponent]; return the attack correction.

    The correction returned is attack minus the raw team mean, i.e. how much the
    schedule was worth. Adding it to a player's grade removes his schedule.
    """
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    design = np.zeros((len(rows), 2 * n))
    for r, (team, opponent) in enumerate(zip(rows.team, rows.opponent)):
        design[r, index[team]] = 1.0
        design[r, n + index[opponent]] = -1.0
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    model.fit(design, rows.y.to_numpy(float))
    attack = model.coef_[:n] + model.intercept_
    return {t: float(attack[i]) for t, i in index.items()}


def adjust(values: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Return the facet table with `grade` replaced by its schedule-free version."""
    means = team_season_means(values)
    pairs = schedule_pairs(schedule)
    out = values.copy()
    corrections = []
    for (season, facet), block in means.groupby(["season", "facet"]):
        lookup = dict(zip(block.cfbd_team, block.mean_grade))
        games = pairs[pairs.season == season]
        games = games[games.team.isin(lookup) & games.opponent.isin(lookup)]
        if len(games) < 50:
            continue
        rows = games.assign(y=games.team.map(lookup))
        teams = sorted(set(rows.team) | set(rows.opponent))
        attack = solve_attack_defence(rows, teams)
        # Centre the correction so the league mean grade is unchanged: this removes
        # schedule, it does not inflate or deflate the whole season. Only teams that
        # actually appear in the season's schedule get a correction - a programme
        # mid-transition from FCS has facet rows but no games here, and keeps its raw
        # grade rather than borrowing someone else's schedule.
        delta = {t: attack[t] - lookup[t] for t in attack if t in lookup}
        centre = float(np.nanmean(list(delta.values())))
        sign = -1.0 if facet in DEF_FACETS else 1.0
        for team, value in delta.items():
            corrections.append({"season": season, "facet": facet,
                                "cfbd_team": team,
                                "correction": sign * (value - centre)})
    corr = pd.DataFrame(corrections)
    out = out.merge(corr, on=["season", "facet", "cfbd_team"], how="left")
    out["correction"] = out.correction.fillna(0.0)
    out["grade_raw"] = out.grade
    out["grade"] = out.grade + out.correction
    return out


def restandardize(values: pd.DataFrame) -> pd.DataFrame:
    """Redo the z and value columns from the adjusted grade, within season and facet."""
    out = values.copy()
    def _z(g):
        sd = float(g.grade.std(ddof=0)) or 1.0
        return (g.grade - g.grade.mean()) / sd
    out["z"] = out.groupby(["season", "facet"], group_keys=False).apply(_z)
    out["value"] = out.z * out.snaps
    return out


def team_matrix(values: pd.DataFrame) -> pd.DataFrame:
    """(season, team) x facet matrix of summed player value, standardized."""
    agg = values.groupby(["season", "cfbd_team", "facet"]).value.sum().reset_index()
    wide = agg.pivot_table(index=["season", "cfbd_team"], columns="facet",
                           values="value", fill_value=0.0)
    wide = wide.groupby(level=0).transform(
        lambda c: (c - c.mean()) / (c.std(ddof=0) or 1.0))
    wide.columns = [f"cfbd_{c}" for c in wide.columns]
    return wide.reset_index().rename(columns={"cfbd_team": "team"})


def targets() -> pd.DataFrame:
    """Adjusted win percentage per team-season, matching horse_race's target."""
    matrix = pd.read_csv(f"{HERE}/horse_race_matrix.csv")
    return matrix[["season", "team", "adj_win_pct"]]


def score(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> float:
    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                  random_state=0, n_jobs=-1)
    pred = cross_val_predict(model, X, y, cv=GroupKFold(n_splits=5), groups=groups)
    return float(np.corrcoef(pred, y)[0, 1])


def evaluate(matrix: pd.DataFrame, label: str) -> dict:
    tgt = targets()
    same = matrix.merge(tgt, on=["season", "team"], how="inner").dropna()
    feats = [c for c in same.columns if c.startswith("cfbd_")]
    same_r = score(same[feats], same.adj_win_pct.to_numpy(float),
                   same.season.to_numpy())

    forward = matrix.copy()
    forward["season"] = forward.season + 1
    fwd = forward.merge(tgt, on=["season", "team"], how="inner").dropna()
    fwd_r = score(fwd[feats], fwd.adj_win_pct.to_numpy(float),
                  fwd.season.to_numpy())
    return {"label": label, "n_same": int(len(same)), "same_season_r": same_r,
            "n_forward": int(len(fwd)), "next_season_r": fwd_r,
            "n_features": len(feats)}


def main():
    values = pd.read_parquet(FACETS_PARQUET)
    schedule = pd.read_csv(SCHEDULE)
    print(f"player-facet rows: {len(values):,}   schedule games: {len(schedule):,}")

    raw = evaluate(team_matrix(values), "cfbd raw")
    adjusted_values = restandardize(adjust(values, schedule))
    adjusted_values.to_parquet(OUT_PARQUET, index=False)
    adj = evaluate(team_matrix(adjusted_values), "cfbd opponent-adjusted")

    moved = adjusted_values.correction.abs()
    result = {
        "question": "does opponent-adjusting production close the gap to PFF",
        "method": {"model": "ridge attack/defence on the season schedule, per facet",
                   "ridge_alpha": RIDGE_ALPHA,
                   "estimator": "RandomForest, GroupKFold on season, as horse_race"},
        "correction_size": {
            "mean_abs": float(moved.mean()), "p90": float(moved.quantile(.90)),
            "grade_sd": float(adjusted_values.grade_raw.std())},
        "cfbd_raw": raw, "cfbd_opponent_adjusted": adj,
        "pff_reference": {"same_season_r": 0.830, "next_season_r": 0.475,
                          "source": "horse_race.py on the same rows"},
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"\n{'set':28s} {'same-season r':>14} {'next-season r':>14}")
    print(f"{'PFF (reference)':28s} {0.830:>14.3f} {0.475:>14.3f}")
    print(f"{raw['label']:28s} {raw['same_season_r']:>14.3f} "
          f"{raw['next_season_r']:>14.3f}")
    print(f"{adj['label']:28s} {adj['same_season_r']:>14.3f} "
          f"{adj['next_season_r']:>14.3f}")
    print(f"\ncorrection: mean |shift| {moved.mean():.4f}, p90 {moved.quantile(.90):.4f}"
          f"  (grade SD {adjusted_values.grade_raw.std():.4f})")
    print(f"-> {OUT_JSON}\n-> {OUT_PARQUET}")
    return result


if __name__ == "__main__":
    main()
