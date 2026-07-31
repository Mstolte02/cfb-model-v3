"""Which of the three v1 defects actually mattered?

v1 and v2 report holdout numbers on different populations - v1 scores only players who
took a snap, v2 scores every two-deep slot - so their headline r and MAE cannot be
compared to each other. This holds the test set fixed at the v2 2025 two-deep and
varies only how the training rows are built, which is the comparison that means
something.

Each variant trains on target seasons 2022-24 and is scored on 2025.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from project_2026_v2 import (
    HERE, FEATURES, build_history, build_population, load_rosters,
    make_training, slot_counts, fit)
from build_roster_2026 import norm_name
from build_recruiting import load_recruits

SEEDS = range(5)


def score(trn, tst, feats):
    r, mae, top = [], [], []
    for s in SEEDS:
        m = fit(s).fit(trn[feats], trn.war)
        p = m.predict(tst[feats])
        r.append(np.corrcoef(p, tst.war)[0, 1])
        mae.append(mean_absolute_error(tst.war, p))
        # do the top-50 picks actually deliver? mean realised WAR of the top 50
        top.append(tst.war.to_numpy()[np.argsort(-p)[:50]].mean())
    return np.mean(r), np.mean(mae), np.mean(top)


def main():
    war = pd.read_csv(f"{HERE}/player_war.csv")
    ratings = pd.read_csv(f"{HERE}/team_ratings.csv")
    recs = pd.read_csv(f"{HERE}/records.csv")
    rec = load_recruits()
    ros26 = pd.read_csv(f"{HERE}/roster_2026.csv")
    K, S = slot_counts(ros26)

    w = build_history(war)
    rosters = load_rosters([2021, 2022, 2023, 2024, 2025], set(recs.team.unique()))
    pop = build_population(w, rosters, K)
    tr = make_training(pop, w, ratings, rec, rosters, S, [2022, 2023, 2024, 2025])
    groups = sorted(w.group.dropna().unique())
    tr["group_code"] = tr.group.map({g: i for i, g in enumerate(groups)})
    tr["share_lag1"] = tr.share_lag1.fillna(0.0)

    # the leaky v1 flag, reconstructed on the same rows for a like-for-like contrast
    tr["is_starter_v1"] = (tr.snap_rank <= 1).astype(int)
    tr["class_num_v1"] = (tr.prior_seasons + 1).clip(upper=5)

    tst = tr[tr.target_season == 2025]
    base = tr[tr.target_season < 2025]
    print(f"fixed test set: 2025 two-deep slots, n={len(tst)}")
    print(f"training pool: {len(base)} rows "
          f"({int((~base.played).sum())} of them never took a snap)\n")

    variants = {}

    # v1 as it stands: leaky starter flag, approximated class, no non-players
    v1 = base[base.played].copy()
    v1["is_starter"] = v1.is_starter_v1
    v1["class_num"] = v1.class_num_v1
    f_nosh = [f for f in FEATURES if f != "share_lag1"]
    variants["v1 (all three defects)"] = (v1, f_nosh)

    # fix 1 only: starter density matched
    a = base[base.played].copy()
    a["class_num"] = a.class_num_v1
    variants["+ fix starter density"] = (a, f_nosh)

    # fix 1+2: also train on roster members who never played
    b = base.copy()
    b["class_num"] = b.class_num_v1
    variants["+ add the non-players"] = (b, f_nosh)

    # fix 1+2+3: real class year from the CFBD roster
    variants["+ real class year"] = (base, f_nosh)

    # and the ex-ante role feature that replaces what the leaked flag was doing
    variants["+ prior snap share (v2)"] = (base, FEATURES)

    print(f"  {'variant':<26} {'r':>7} {'MAE':>8} {'top50 actual WAR':>18}")
    for nm, (trn, feats) in variants.items():
        r, mae, top = score(trn, tst, feats)
        print(f"  {nm:<26} {r:>7.3f} {mae:>8.4f} {top:>18.4f}")

    # what a perfect ranker would get, and what carry-forward gets
    print(f"\n  {'oracle top-50':<26} {'':>7} {'':>8} "
          f"{tst.war.nlargest(50).mean():>18.4f}")
    naive = tst.war_lag1.fillna(0).to_numpy()
    print(f"  {'carry-forward':<26} {np.corrcoef(naive, tst.war)[0,1]:>7.3f} "
          f"{mean_absolute_error(tst.war, naive):>8.4f} "
          f"{tst.war.to_numpy()[np.argsort(-naive)[:50]].mean():>18.4f}")

    # the specific failure mode: never-played players ranked above proven ones
    print("\nzero-history slots in the 2025 test set, predicted vs actual:")
    for nm, (trn, feats) in variants.items():
        m = fit(0).fit(trn[feats], trn.war)
        p = m.predict(tst[feats])
        z = tst.prior_seasons == 0
        print(f"  {nm:<26} predicted {p[z.to_numpy()].mean():.4f}  "
              f"actual {tst.war[z].mean():.4f}  "
              f"ratio {p[z.to_numpy()].mean()/tst.war[z].mean():.2f}x")


if __name__ == "__main__":
    main()
