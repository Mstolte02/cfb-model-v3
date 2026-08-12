"""Leakage-safe historical player projections aggregated to team talent.

The original 2026 player model predicts a two-deep, but its historical training
population first retained the K players with the most *target-season* snaps.  That is
reasonable for describing a completed season and invalid for a preseason backtest.

This experiment starts with every player on the CFBD season roster, assigns zero WAR
to roster members with no PFF row, builds features only from earlier seasons, predicts
everybody, and then takes the K highest projected players in each team/position room.
No target-season snap count or target-season depth result chooses the population.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

import artifacts
from build_recruiting import load_recruits
from facets import YEARS as WAR_YEARS
from project_2026_v2 import (FEATURES, build_history, build_population, fit,
                             load_rosters, make_training, slot_counts)


OUT_CSV = HERE / "preseason_team_war.csv"
OUT_JSON = HERE / "preseason_team_war_metrics.json"
PROJECTION_YEARS = list(range(2020, 2026))


def _z(series: pd.Series) -> pd.Series:
    sd = float(series.std(ddof=0))
    return (series - series.mean()) / sd if sd > 1e-12 else series * 0.0


def aggregate_top_k(rows: pd.DataFrame, pred: np.ndarray, slots: dict) -> pd.Series:
    d = rows[["team", "group"]].copy()
    d["proj_war"] = np.asarray(pred, float)
    d["rank"] = d.groupby(["team", "group"]).proj_war.rank(
        ascending=False, method="first")
    d = d[d["rank"] <= d.group.map(slots).fillna(2)]
    return d.groupby("team").proj_war.sum()


def main():
    player_war = pd.read_csv(HERE / artifacts.PLAYER_WAR)
    ratings = pd.read_csv(HERE / artifacts.TEAM_RATINGS)
    records = pd.read_csv(HERE / "records.csv")
    recruits = load_recruits()
    roster_2026 = pd.read_csv(HERE / "roster_2026.csv")
    slots, starter_slots = slot_counts(roster_2026)

    history = build_history(player_war)
    fbs = set(records.team.unique())
    roster_years = sorted(set(WAR_YEARS) | set(PROJECTION_YEARS))
    rosters = load_rosters(roster_years, fbs)
    pop = build_population(history, rosters)
    valid_targets = [y for y in WAR_YEARS if (y - 1) in set(WAR_YEARS)]
    needed = sorted(set(valid_targets) | set(PROJECTION_YEARS))
    training = make_training(pop, history, ratings, recruits, rosters,
                             starter_slots, needed)
    groups = sorted(history.group.dropna().unique())
    training["group_code"] = training.group.map({g: i for i, g in enumerate(groups)})
    training["share_lag1"] = training.share_lag1.fillna(0.0)

    rows, metrics = [], []
    actual_team = history.groupby(["season", "team"]).war.sum()
    for year in PROJECTION_YEARS:
        train_years = [y for y in valid_targets if y < year]
        tr = training[training.target_season.isin(train_years)]
        te = training[training.target_season == year].copy()
        if tr.empty or te.empty:
            continue
        model = fit().fit(tr[FEATURES], tr.war)
        player_pred = model.predict(te[FEATURES])
        team_pred = aggregate_top_k(te, player_pred, slots)
        for team, value in team_pred.items():
            rows.append({"season": year, "team": team, "projected_war": value})

        if year in actual_team.index.get_level_values("season"):
            actual = actual_team.xs(year).reindex(team_pred.index).dropna()
            predicted = team_pred.reindex(actual.index)
            az, pz = _z(actual), _z(predicted)
            metrics.append({
                "season": year, "train_target_seasons": train_years,
                "n_roster_players": int(len(te)), "n_teams": int(len(actual)),
                "team_war_r": float(np.corrcoef(predicted, actual)[0, 1]),
                "team_war_z_mae": float(mean_absolute_error(az, pz)),
                "roster_players_with_target_snaps": int((te.snaps > 0).sum()),
                "roster_players_without_target_snaps": int((te.snaps <= 0).sum()),
            })
            print(f"{year}: teams={len(actual)} r={metrics[-1]['team_war_r']:.3f} "
                  f"zMAE={metrics[-1]['team_war_z_mae']:.3f}")

    out = pd.DataFrame(rows)
    out["projected_war_z"] = out.groupby("season").projected_war.transform(_z)
    out.to_csv(OUT_CSV, index=False)
    payload = {
        "contract": "all season-roster players predicted from N-1 and earlier; top K selected by prediction",
        "target_snap_population_selection": False,
        "slot_counts": slots, "features": FEATURES, "folds": metrics,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"-> {OUT_CSV}\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
