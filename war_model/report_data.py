"""Stage 6: collapse every output into one JSON bundle for the report."""
import json, os
import numpy as np
import pandas as pd

import artifacts

from facets import YEARS, POS_GROUP
from rb_analysis import RB_FACETS, PRETTY

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    qual = pd.read_csv(f"{HERE}/rb_qualified.csv")
    career = pd.read_csv(f"{HERE}/rb_career.csv")
    war = pd.read_csv(f"{HERE}/{artifacts.PLAYER_WAR}")
    pos = pd.read_csv(f"{HERE}/position_table.csv", index_col=0)
    weights = pd.read_csv(f"{HERE}/{artifacts.FACET_WEIGHTS}", index_col=0)
    team = pd.read_csv(f"{HERE}/team_implied_wins.csv")
    ratings = pd.read_csv(f"{HERE}/{artifacts.TEAM_RATINGS}")

    out = {}

    # leaderboards by season
    cols = ["player", "team", "carries", "yards", "ypa", "touchdowns", "waa", "war"]
    out["seasons"] = {}
    for s in YEARS:
        d = qual[qual.season == s].nlargest(25, "war")[cols + [f"war_{f}" for f in RB_FACETS]]
        out["seasons"][str(s)] = json.loads(d.round(4).to_json(orient="records"))

    out["career"] = json.loads(
        career.nlargest(25, "war")[["player", "teams", "seasons", "carries", "yards", "war"]]
        .round(3).to_json(orient="records"))

    # positional value
    out["positions"] = json.loads(pos.reset_index().round(3).to_json(orient="records"))

    # facet weights
    w = weights.reset_index().rename(columns={"index": "facet"})
    out["weights"] = json.loads(w.round(4).to_json(orient="records"))

    # where RB wins come from
    wcols = [f"war_{f}" for f in RB_FACETS]
    share = qual[wcols].sum()
    out["rb_facet_share"] = [
        {"facet": PRETTY[c[4:]], "pct": round(float(share[c] / share.sum() * 100), 1)}
        for c in wcols]

    # WAR vs yards divergence
    out["risers"] = json.loads(qual.nlargest(8, "rank_shift")[
        ["season", "player", "team", "yards", "carries", "rank_yards", "rank_war", "war"]]
        .round(3).to_json(orient="records"))
    out["fallers"] = json.loads(qual.nsmallest(8, "rank_shift")[
        ["season", "player", "team", "yards", "carries", "rank_yards", "rank_war", "war"]]
        .round(3).to_json(orient="records"))

    # scatter: WAR vs rushing yards, most recent season
    sc = qual[qual.season == 2025][["player", "yards", "war", "carries"]].dropna()
    out["scatter"] = json.loads(sc.round(3).to_json(orient="records"))

    # correlations with conventional metrics
    out["correlations"] = [
        {"metric": m, "r": round(float(qual[[m, "war"]].dropna()[m].corr(qual.war)), 3)}
        for m in ["yards", "carries", "avoided_tackles", "grades_run", "touchdowns",
                  "yco_attempt", "ypa", "breakaway_percent"]]

    # headline numbers
    rb_all = war[war.position.isin(["HB", "FB"])]
    out["headline"] = {
        "player_seasons": int(len(war)),
        "rb_seasons": int(len(rb_all)),
        "qualified": int(len(qual)),
        "teams": int(ratings.team.nunique()),
        "games": 3944,
        "rb_mean_war": round(float(pos.loc["RB", "mean_WAR"]), 3),
        "qb_mean_war": round(float(pos.loc["QB", "mean_WAR"]), 3),
        "rb_yoy": round(float(pos.loc["RB", "yoy_r"]), 3),
        "implied_r": round(float(np.corrcoef(team.implied_wins, team.fbs_wins)[0, 1]), 3),
        "implied_mae": round(float(np.abs(team.implied_wins - team.fbs_wins).mean()), 2),
        "massey_r": round(float(np.corrcoef(ratings.massey, ratings.adj_win_pct)[0, 1]), 3),
        "top_rb_war": round(float(qual.war.max()), 3),
    }

    # spearman WAR vs yards by season
    out["spearman"] = [
        {"season": int(s),
         "rho": round(float(qual[qual.season == s].rank_war.corr(
             qual[qual.season == s].rank_yards, method="spearman")), 3)}
        for s in YEARS]

    json.dump(out, open(f"{HERE}/report.json", "w"), indent=1)
    print(f"report.json written  ({os.path.getsize(f'{HERE}/report.json')/1024:.0f} KB)")
    print(json.dumps(out["headline"], indent=1))


if __name__ == "__main__":
    main()
