"""Stage 9: team win projections and the report data bundle."""
import json, os
import numpy as np
import pandas as pd

import artifacts
from sklearn.linear_model import LinearRegression

HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = ["QB", "RB", "WR", "TE", "OT", "IOL", "DT", "EDGE", "LB", "CB", "SAF"]


def calibrate_wins(war, recs):
    """Map summed team WAR to actual wins, fitted on 2021-25 rather than assumed.

    Summing the roster's WAR and adding a replacement baseline understates spread,
    because a projection is a conditional mean. Fitting the map on realized seasons
    puts team totals back on a real win scale.
    """
    t = war.groupby(["season", "team"], as_index=False).war.sum()
    t = t.merge(recs, on=["season", "team"])
    t["win_rate"] = t.fbs_wins / t.fbs_games
    X = t[["war"]].to_numpy()
    y = t.win_rate.to_numpy()
    lm = LinearRegression().fit(X, y)
    r = np.corrcoef(lm.predict(X), y)[0, 1]
    return lm, r, t


def main():
    proj = pd.read_csv(f"{HERE}/projections_2026_v2.csv")
    metrics = json.load(open(f"{HERE}/projection_metrics.json"))
    war = pd.read_csv(f"{HERE}/{artifacts.PLAYER_WAR}")
    recs = pd.read_csv(f"{HERE}/records.csv")
    pos = pd.read_csv(f"{HERE}/position_table.csv", index_col=0)
    weights = pd.read_csv(f"{HERE}/{artifacts.FACET_WEIGHTS}", index_col=0)
    ratings = pd.read_csv(f"{HERE}/{artifacts.TEAM_RATINGS}")

    lm, r, hist = calibrate_wins(war, recs)
    print(f"team WAR -> win rate: slope={lm.coef_[0]:.4f} intercept={lm.intercept_:.4f}  r={r:.3f}")

    team = proj.groupby(["team", "conference"], as_index=False).agg(
        proj_war=("proj_war", "sum"), slots=("player", "count"),
        imputed=("imputed", "sum"), starters=("is_starter", "sum"))
    team["proj_win_rate"] = lm.predict(team[["proj_war"]])
    team["proj_wins_12"] = (team.proj_win_rate * 12).round(2)
    team = team.sort_values("proj_war", ascending=False).reset_index(drop=True)
    team["rank"] = team.index + 1

    # per-group team strength
    for g in GROUPS:
        s = proj[proj.broad_group == g].groupby("team").proj_war.sum()
        team[g] = team.team.map(s).fillna(0.0).round(3)

    print("\ntop 15 projected 2026 teams:")
    print(team.head(15)[["rank", "team", "conference", "proj_war", "proj_wins_12"]]
          .to_string(index=False))

    out = {}
    out["teams"] = json.loads(team.round(3).to_json(orient="records"))

    # player leaderboards per position group
    cols = ["player", "team", "broad_group", "class", "is_starter", "is_transfer",
            "proj_war", "imputed", "stars", "snaps_2025", "war_2025"]
    out["players"] = {}
    for g in GROUPS:
        d = proj[proj.broad_group == g].nlargest(20, "proj_war")[cols]
        out["players"][g] = json.loads(d.round(3).to_json(orient="records"))
    out["overall"] = json.loads(proj.nlargest(30, "proj_war")[cols].round(3).to_json(orient="records"))

    # best players with no snap history at all - the pure imputations
    imp = proj[proj.imputed].nlargest(15, "proj_war")[cols]
    out["top_imputed"] = json.loads(imp.round(3).to_json(orient="records"))

    # positional value: what the model says each group is worth
    pv = proj.groupby("broad_group").agg(
        total=("proj_war", "sum"), mean=("proj_war", "mean"), n=("player", "count"))
    pv["per_starter"] = proj[proj.is_starter == 1].groupby("broad_group").proj_war.mean()
    out["position_value"] = json.loads(
        pv.reset_index().round(3).to_json(orient="records"))

    out["facet_weights"] = json.loads(
        weights.reset_index().rename(columns={"index": "facet"}).round(4).to_json(orient="records"))
    out["position_table"] = json.loads(pos.reset_index().round(3).to_json(orient="records"))

    out["coverage"] = json.loads(proj.groupby("broad_group").apply(
        lambda g: pd.Series({"slots": len(g), "imputed": int(g.imputed.sum()),
                             "pct_imputed": round(g.imputed.mean()*100, 1)}),
        include_groups=False).reset_index().to_json(orient="records"))

    out["meta"] = {
        "slots": int(len(proj)), "teams": int(proj.team.nunique()),
        "imputed": int(proj.imputed.sum()),
        "imputed_pct": round(float(proj.imputed.mean()*100), 1),
        "matched_pct": round(float((~proj.imputed).mean()*100), 1),
        **metrics,
        "war_calib_r": round(float(r), 3),
        "massey_r": round(float(np.corrcoef(ratings.massey, ratings.adj_win_pct)[0, 1]), 3),
        "games": 3944, "player_seasons": int(len(war)),
        "rb_mean_war": float(pos.loc["RB", "mean_WAR"]),
        "qb_mean_war": float(pos.loc["QB", "mean_WAR"]),
    }

    json.dump(out, open(f"{HERE}/report_2026.json", "w"), indent=1)
    team.to_csv(f"{HERE}/team_projections_2026.csv", index=False)
    print(f"\nreport_2026.json + team_projections_2026.csv written")


if __name__ == "__main__":
    main()
