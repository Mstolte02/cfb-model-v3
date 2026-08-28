"""Leakage-safe historical player projections aggregated to team POSITION GROUPS.

``preseason_team_projection.py`` produces one projected WAR number per team.  The
matchup experiment needs the same number split by position room, because a team's
projected WAR against an opponent's projected WAR is one comparison and eleven
rooms against eleven rooms is a different one.

Everything about the contract is unchanged and deliberately shares the same
functions: every player on the season-N CFBD roster is predicted from N-1 and
earlier, the top K in each team/position room are chosen by the prediction, and no
target-season snap count selects the population.  The only difference is where the
sum stops.  Group sums therefore reconcile exactly with ``preseason_team_war.csv``.
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

import artifacts
from build_recruiting import load_recruits
from facets import YEARS as WAR_YEARS
from preseason_team_projection import PROJECTION_YEARS
from project_2026_v2 import (FEATURES, build_history, build_population, fit,
                             load_rosters, make_training, slot_counts)


OUT_CSV = HERE / "preseason_group_war.csv"
OUT_JSON = HERE / "preseason_group_war_metrics.json"


def aggregate_top_k_by_group(rows: pd.DataFrame, pred: np.ndarray,
                             slots: dict) -> pd.DataFrame:
    """The same top-K selection as the team aggregate, stopped one level earlier."""
    d = rows[["team", "group"]].copy()
    d["proj_war"] = np.asarray(pred, float)
    d["rank"] = d.groupby(["team", "group"]).proj_war.rank(
        ascending=False, method="first")
    d = d[d["rank"] <= d.group.map(slots).fillna(2)]
    return d.groupby(["team", "group"], as_index=False).proj_war.sum()


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

    frames, folds = [], []
    for year in PROJECTION_YEARS:
        train_years = [y for y in valid_targets if y < year]
        tr = training[training.target_season.isin(train_years)]
        te = training[training.target_season == year].copy()
        if tr.empty or te.empty:
            continue
        model = fit().fit(tr[FEATURES], tr.war)
        pred = model.predict(te[FEATURES])
        g = aggregate_top_k_by_group(te, pred, slots)
        g.insert(0, "season", year)
        frames.append(g)
        folds.append({"season": int(year), "train_target_seasons": train_years,
                      "n_teams": int(g.team.nunique()),
                      "n_groups": int(g.group.nunique()),
                      "team_total": float(g.proj_war.sum())})
        print(f"{year}: teams={g.team.nunique()} groups={g.group.nunique()}")

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps({
        "contract": "all season-roster players predicted from N-1 and earlier; "
                    "top K per team/position room selected by the prediction",
        "reconciles_with": "preseason_team_war.csv (group sums equal team totals)",
        "slot_counts": slots, "features": FEATURES, "folds": folds}, indent=2))
    print(f"-> {OUT_CSV}\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
