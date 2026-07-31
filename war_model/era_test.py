"""Does the extra history actually help the projection, or is the sport too different?

Adding 2014-19 to the WAR build gives the projection model nine target seasons instead
of four. The naive expectation is that more data is better. The first run said
otherwise - the 2025 holdout got worse - which is worth taking seriously rather than
explaining away: the transfer portal opened in Oct 2018 and one-time-transfer became
free in 2021, so a player-movement model fitted on 2015 is fitted on a sport where
players essentially did not move.

This trains the same model on different windows of history and scores all of them on
the same held-out 2025, so the only thing changing is which seasons it learned from.

Run: ./rbenv/bin/python era_test.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

import artifacts
from facets import YEARS as WAR_YEARS
from build_recruiting import load_recruits
from project_2026_v2 import (HERE, FEATURES, build_history, build_population,
                             load_rosters, make_training, slot_counts, fit)

SEEDS = range(3)


def main():
    war = pd.read_csv(f"{HERE}/{artifacts.PLAYER_WAR}")
    ratings = pd.read_csv(f"{HERE}/{artifacts.TEAM_RATINGS}")
    recs = pd.read_csv(f"{HERE}/records.csv")
    rec = load_recruits()
    ros26 = pd.read_csv(f"{HERE}/roster_2026.csv")
    K, S = slot_counts(ros26)

    w = build_history(war)
    rosters = load_rosters(WAR_YEARS, set(recs.team.unique()))
    pop = build_population(w, rosters, K)
    targets = [y for y in WAR_YEARS if (y - 1) in set(WAR_YEARS)]
    tr = make_training(pop, w, ratings, rec, rosters, S, targets)
    groups = sorted(w.group.dropna().unique())
    tr["group_code"] = tr.group.map({g: i for i, g in enumerate(groups)})
    tr["share_lag1"] = tr.share_lag1.fillna(0.0)

    tst = tr[tr.target_season == 2025]
    pool = tr[tr.target_season < 2025]
    print(f"held-out 2025: {len(tst)} slots")
    print(f"available target seasons: {sorted(pool.target_season.unique())}\n")

    # how much of the older data even carries a recruiting rating, since that is what
    # the model leans on for players with no snap history
    print("recruiting coverage by target season:")
    for s, g in pool.groupby("target_season"):
        print(f"  {int(s)}: stars present {g.stars.notna().mean()*100:5.1f}%   rows {len(g)}")

    windows = {
        "2022-24 only (portal era)":      [2022, 2023, 2024],
        "2019 + 2022-24":                 [2019, 2022, 2023, 2024],
        "2018-19 + 2022-24":              [2018, 2019, 2022, 2023, 2024],
        "2017-19 + 2022-24":              [2017, 2018, 2019, 2022, 2023, 2024],
        "all available":                  sorted(pool.target_season.unique()),
        "pre-portal only (2015-19)":      [2015, 2016, 2017, 2018, 2019],
    }

    print(f"\n{'training window':<30}{'rows':>7}{'r':>8}{'MAE':>9}"
          f"{'r (no history)':>16}{'top50':>8}")
    print("-" * 78)
    for nm, yrs in windows.items():
        trn = pool[pool.target_season.isin(yrs)]
        if not len(trn):
            continue
        rs, maes, nh, tops = [], [], [], []
        for sd in SEEDS:
            m = fit(sd).fit(trn[FEATURES], trn.war)
            p = m.predict(tst[FEATURES])
            rs.append(np.corrcoef(p, tst.war)[0, 1])
            maes.append(mean_absolute_error(tst.war, p))
            z = (tst.prior_seasons == 0).to_numpy()
            nh.append(np.corrcoef(p[z], tst.war[z])[0, 1])
            tops.append(tst.war.to_numpy()[np.argsort(-p)[:50]].mean())
        print(f"{nm:<30}{len(trn):>7}{np.mean(rs):>8.3f}{np.mean(maes):>9.4f}"
              f"{np.mean(nh):>16.3f}{np.mean(tops):>8.3f}")

    print(f"\n{'carry last season forward':<30}{'':>7}"
          f"{np.corrcoef(tst.war_lag1.fillna(0), tst.war)[0,1]:>8.3f}"
          f"{mean_absolute_error(tst.war, tst.war_lag1.fillna(0)):>9.4f}")
    print(f"{'oracle top-50':<30}{'':>7}{'':>8}{'':>9}{'':>16}"
          f"{tst.war.nlargest(50).mean():>8.3f}")


if __name__ == "__main__":
    main()
