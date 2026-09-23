"""Does the sample-size prior improve the shipped preseason WAR projection?

audit/WAR_SAMPLE_SIZE_UPDATING.md showed a per-player filter that counts each season
by its snaps beats fixed year weights at predicting next season. The shipped
projection (war_model/project_2026_v2.py) is a boosted tree that already sees
war_lag1-3 and snaps_lag1-2, so it may have learned the same thing. This adds the
filter's per-snap mean and its standard deviation as two features and compares the
tree with and without them.

Holdouts 2023, 2024 and 2025, each trained on earlier target seasons only. For
holdout H the filter's parameters are fitted on seasons before H, and the filter at
target season T reads only seasons before T.

    python -m scripts.war_projection_prior_test
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "war_model"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import mean_absolute_error  # noqa: E402

import artifacts  # noqa: E402
import project_2026_v2 as P  # noqa: E402
from config import ARTIFACTS  # noqa: E402
from scripts import war_credibility_backtest as K  # noqa: E402

OUT = ARTIFACTS / "war_projection_prior_test.json"
HOLDOUTS = [2023, 2024, 2025]


def filter_features(d: pd.DataFrame, seasons: list[int], fit_before: int) -> pd.DataFrame:
    """(player_id, group, target_season) -> kal_m, kal_sd for every target season,
    with parameters fitted on seasons < fit_before."""
    out = []
    tfit = seasons.index(fit_before) if fit_before in seasons else len(seasons)
    for g in sorted(d.group.unique()):
        ids, R, S = K.panel(d, g, seasons + [max(seasons) + 1])
        train = (S[:, :tfit] >= K.MIN_TARGET_SNAPS) & np.isfinite(R[:, :tfit])
        mu = float(np.average(R[:, :tfit][train], weights=S[:, :tfit][train]))
        params = K.fit_kalman(R, S, tfit, mu)
        T = len(seasons)
        m, P_ = K.kalman(R, S, params, mu, T)
        for t in range(1, T + 1):
            target = seasons[t] if t < T else max(seasons) + 1
            out.append(pd.DataFrame({"player_id": ids, "group": g,
                                     "target_season": target,
                                     "kal_m": m[:, t], "kal_sd": np.sqrt(P_[:, t])}))
    return pd.concat(out, ignore_index=True)


def main():
    from build_recruiting import load_recruits
    war = pd.read_csv(P.HERE + "/" + artifacts.PLAYER_WAR)
    ratings = pd.read_csv(P.HERE + "/" + artifacts.TEAM_RATINGS)
    recs = pd.read_csv(P.HERE + "/records.csv")
    rec = load_recruits()
    ros26 = pd.read_csv(P.HERE + "/roster_2026.csv")
    K_, S_ = P.slot_counts(ros26)
    w = P.build_history(war)
    rosters = P.load_rosters(P.WAR_YEARS, set(recs.team.unique()))
    pop = P.build_population(w, rosters, K_)
    targets = [y for y in P.WAR_YEARS if (y - 1) in set(P.WAR_YEARS)]
    tr = P.make_training(pop, w, ratings, rec, rosters, S_, targets)
    gcode = {g: i for i, g in enumerate(sorted(w.group.dropna().unique()))}
    tr["group_code"] = tr.group.map(gcode)
    tr["share_lag1"] = tr.share_lag1.fillna(0.0)
    tr["player_id"] = tr.player_id.astype("string")

    d = K.load()                      # production WAR, per-snap, grouped
    seasons = sorted(d.season.unique())
    feats = P.FEATURES + ["kal_m", "kal_sd"]
    report = {"holdouts": {}}
    for H in HOLDOUTS:
        kf = filter_features(d, seasons, H)
        kf["player_id"] = kf.player_id.astype("string")
        t = tr.merge(kf, on=["player_id", "group", "target_season"], how="left")
        trn, tst = t[t.target_season < H], t[t.target_season == H]
        res = {}
        for name, cols in (("shipped", P.FEATURES), ("with_prior", feats)):
            preds = []
            for seed in range(3):
                preds.append(P.fit(seed).fit(trn[cols], trn.war).predict(tst[cols]))
            p = np.mean(preds, axis=0)
            res[name] = {"mae": float(mean_absolute_error(tst.war, p)),
                         "r": float(np.corrcoef(p, tst.war)[0, 1]),
                         "rmse": float(np.sqrt(np.mean((p - tst.war) ** 2)))}
            tst = tst.assign(**{f"p_{name}": p})
        hist = tst[tst.prior_seasons > 0]
        for name in ("shipped", "with_prior"):
            res[name]["mae_returning"] = float(mean_absolute_error(hist.war, hist[f"p_{name}"]))
        # team-season sums: what the team model actually reads
        team = tst.groupby("team")[["war", "p_shipped", "p_with_prior"]].sum()
        for name in ("shipped", "with_prior"):
            res[name]["team_r"] = float(np.corrcoef(team[f"p_{name}"], team.war)[0, 1])
            res[name]["team_mae"] = float(mean_absolute_error(team.war, team[f"p_{name}"]))
        res["n"] = int(len(tst))
        res["kal_coverage"] = float(tst.kal_m.notna().mean())
        report["holdouts"][str(H)] = res
        print(H, json.dumps(res, indent=1), flush=True)
    s = report["holdouts"]
    report["mean_mae_change_pct"] = float(np.mean(
        [100 * (s[h]["with_prior"]["mae"] / s[h]["shipped"]["mae"] - 1) for h in s]))
    report["mean_team_mae_change_pct"] = float(np.mean(
        [100 * (s[h]["with_prior"]["team_mae"] / s[h]["shipped"]["team_mae"] - 1) for h in s]))
    OUT.write_text(json.dumps(report, indent=2))
    print("mean MAE change %", round(report["mean_mae_change_pct"], 2),
          " team MAE change %", round(report["mean_team_mae_change_pct"], 2))


if __name__ == "__main__":
    main()
