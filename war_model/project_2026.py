"""Stage 8: project 2026 WAR for every player on the two-deep.

Trained on the season-to-season transitions inside the 2021-25 WAR build. One model
covers everybody: players with a long snap history, players with a handful of snaps,
and players with none at all. The gradient booster takes NaN natively, so a true
freshman simply arrives with his history features missing and his recruiting rating,
class, team, and projected depth slot carrying the prediction.
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from build_roster_2026 import norm_name, PFF_TO_GROUP
from build_recruiting import load_recruits

HERE = os.path.dirname(os.path.abspath(__file__))
CLASS_NUM = {"FR": 1, "SO": 2, "JR": 3, "SR": 4, "GR": 5}

# redshirt status is deliberately absent: the two-deep reports it for 2026, but it
# cannot be reconstructed for the historical seasons, so it has no training signal
FEATURES = ["war_lag1", "war_lag2", "war_lag3", "snaps_lag1", "snaps_lag2",
            "rate_lag1", "prior_seasons", "class_num", "is_starter", "is_transfer",
            "stars", "rating", "team_rating", "team_massey", "group_code"]


def build_history(war):
    """Player-season WAR/snaps plus the within-team-position snap rank (role)."""
    w = war.copy()
    w["group"] = w.position.map(PFF_TO_GROUP)
    w = w[w.group.notna()]
    w = w.groupby(["season", "player_id", "player", "team", "group"], as_index=False).agg(
        war=("war", "sum"), snaps=("snaps", "sum"))
    w["rank_in_group"] = w.groupby(["season", "team", "group"]).snaps.rank(
        ascending=False, method="first")
    w["is_starter"] = (w.rank_in_group <= 1).astype(int)
    return w


def make_training(w, ratings, rec):
    """One row per (player, season t+1) with features known from t and earlier."""
    rows = []
    for t1 in range(2022, 2026):
        cur = w[w.season == t1].copy()
        for lag in (1, 2, 3):
            p = w[w.season == t1 - lag][["player_id", "war", "snaps"]].rename(
                columns={"war": f"war_lag{lag}", "snaps": f"snaps_lag{lag}"})
            cur = cur.merge(p, on="player_id", how="left")
        cur["prior_seasons"] = sum(cur[f"snaps_lag{l}"].notna() & (cur[f"snaps_lag{l}"] > 0)
                                   for l in (1, 2, 3)).astype(float)
        cur["rate_lag1"] = cur.war_lag1 / cur.snaps_lag1.replace(0, np.nan)
        # class is unknown historically; approximate by seasons of prior FBS snaps
        cur["class_num"] = (cur.prior_seasons + 1).clip(upper=5)
        # transfer = played for a different team last season
        prev_team = w[w.season == t1 - 1][["player_id", "team"]].rename(
            columns={"team": "prev_team"})
        cur = cur.merge(prev_team, on="player_id", how="left")
        cur["is_transfer"] = ((cur.prev_team.notna()) & (cur.prev_team != cur.team)).astype(int)
        cur["redshirt"] = np.nan
        cur["target_season"] = t1
        rows.append(cur)
    tr = pd.concat(rows, ignore_index=True)

    tr["key"] = tr.player.map(norm_name)
    tr = tr.merge(rec[["key", "stars", "rating"]], on="key", how="left")
    tr["team_rating"] = tr.team.map(rec.groupby("committedTo").rating.mean())
    # prior-season team strength
    r = ratings[["season", "team", "massey"]].copy()
    r["season"] += 1
    tr = tr.merge(r.rename(columns={"massey": "team_massey"}), on=["season", "team"], how="left")
    return tr


def main():
    war = pd.read_csv(f"{HERE}/player_war.csv")
    ratings = pd.read_csv(f"{HERE}/team_ratings.csv")
    rec = load_recruits()
    w = build_history(war)

    tr = make_training(w, ratings, rec)
    groups = sorted(w.group.dropna().unique())
    gcode = {g: i for i, g in enumerate(groups)}
    tr["group_code"] = tr.group.map(gcode)

    print(f"training rows (player-seasons with a projection target): {len(tr)}")
    print(f"  of which no prior snaps at all: {int((tr.prior_seasons == 0).sum())}")

    # ---- holdout: train through 2024, predict 2025 -------------------------
    trn = tr[tr.target_season < 2025]
    tst = tr[tr.target_season == 2025]
    mdl = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    mdl.fit(trn[FEATURES], trn.war)
    pred = mdl.predict(tst[FEATURES])

    print(f"\nholdout = 2025 season (n={len(tst)}), trained on 2022-24 targets")
    print(f"  model        r = {np.corrcoef(pred, tst.war)[0,1]:.3f}   "
          f"MAE = {mean_absolute_error(tst.war, pred):.4f}")
    naive = tst.war_lag1.fillna(0)
    print(f"  carry-forward r = {np.corrcoef(naive, tst.war)[0,1]:.3f}   "
          f"MAE = {mean_absolute_error(tst.war, naive):.4f}")
    grp_mean = trn.groupby("group_code").war.mean()
    base = tst.group_code.map(grp_mean)
    print(f"  position mean r = {np.corrcoef(base, tst.war)[0,1]:.3f}   "
          f"MAE = {mean_absolute_error(tst.war, base):.4f}")

    # how well does it do on the players who need imputing?
    noh = tst[tst.prior_seasons == 0]
    if len(noh) > 30:
        p2 = mdl.predict(noh[FEATURES])
        print(f"\n  players with NO prior snaps (n={len(noh)}): "
              f"r = {np.corrcoef(p2, noh.war)[0,1]:.3f}  MAE = {mean_absolute_error(noh.war, p2):.4f}")
        print(f"    predicting the group mean instead: MAE = "
              f"{mean_absolute_error(noh.war, noh.group_code.map(grp_mean)):.4f}")

    # ---- refit on everything, then project 2026 ---------------------------
    final = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    final.fit(tr[FEATURES], tr.war)

    ros = pd.read_csv(f"{HERE}/roster_2026_features.csv")
    ros["war_lag1"] = ros.war_2025.where(ros.snaps_2025 > 0)
    ros["war_lag2"] = ros.war_2024.where(ros.snaps_2024 > 0)
    ros["war_lag3"] = ros.war_2023.where(ros.snaps_2023 > 0)
    ros["snaps_lag1"] = ros.snaps_2025.where(ros.snaps_2025 > 0)
    ros["snaps_lag2"] = ros.snaps_2024.where(ros.snaps_2024 > 0)
    ros["rate_lag1"] = ros.war_lag1 / ros.snaps_lag1
    ros["class_num"] = ros["class"].map(CLASS_NUM)
    ros["is_starter"] = ros.is_starter.astype(int)
    ros["is_transfer"] = ros.is_transfer.astype(int)
    ros["redshirt"] = ros.redshirt.astype(int)
    ros["group_code"] = ros.broad_group.map(gcode)
    m25 = ratings[ratings.season == 2025].set_index("team").massey
    ros["team_massey"] = ros.team.map(m25)

    ros["proj_war"] = final.predict(ros[FEATURES])
    ros["imputed"] = ~ros.has_history

    print(f"\n2026 projections: {len(ros)} slots, "
          f"{int(ros.imputed.sum())} fully imputed ({ros.imputed.mean()*100:.1f}%)")
    print(f"  projected WAR total: {ros.proj_war.sum():.0f}")

    ros.to_csv(f"{HERE}/projections_2026.csv", index=False)
    json.dump({"groups": groups, "features": FEATURES}, open(f"{HERE}/model_meta.json", "w"))
    print("projections_2026.csv written")


if __name__ == "__main__":
    main()
