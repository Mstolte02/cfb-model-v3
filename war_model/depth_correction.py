"""Fix who plays, and how much, before the projection is summed into team WAR.

TWO FAULTS, both about playing time, both measured rather than asserted.

1. THE DEPTH CHART IS WRONG IN PLACES. Ourlads lists A.J. Holmes Jr. as Texas Tech's
   backup nose tackle. He played 455 snaps in 2025 for 0.51 WAR; the two men listed
   ahead of him played 305 and 395 for 0.19 and 0.25. He is not a backup. Across the
   two-deep there are 277 cases on 107 teams where a listed backup out-snapped AND
   out-produced a listed starter in the same unit - about a fifth of all team-position
   units - and they land hardest on the line and secondary: CB 49, EDGE 41, OL 39,
   DT 38, WR 37.

   The chart is one opinion about who will play. Prior-season snaps are a measurement
   of who did. When they disagree by a wide margin the measurement wins, and the two
   swap places within the unit so the number of starters is unchanged.

2. BACKUPS ARE CREDITED FOR INJURIES WE ARE NOT PROJECTING. The projection is trained
   on realized seasons, and realized seasons contain starters getting hurt. So the
   backup slot inherits the average of "sat all year" and "started six games", and a
   true freshman quarterback with no snaps comes out at 13% of his room.

   Splitting historical unit-seasons by how much of the load the starters carried, and
   taking the healthiest quarter as the no-injury case, gives what a backup takes when
   nobody goes down:

       QB   21.7% realized -> 4.7% healthy      OL  16.2% -> 8.4%
       RB   51.7% -> 35.5%                      CB  19.5% -> 7.5%
       DT   41.1% -> 26.5%                      WR  26.4% -> 14.2%

   Quarterback is the extreme - a backup quarterback is worth almost nothing unless
   you are modelling injuries, which we are not. Running back barely moves, because a
   committee is a real thing rather than an injury artefact. Interior defensive line
   stays high for the same reason: that room genuinely rotates.

   The room TOTAL is held fixed and the share moved from backups to starters, because
   the snaps do not disappear when nobody is hurt - the starter takes them.

Run: ../venv/bin/python depth_correction.py [--in FILE] [--out FILE]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
from build_roster_2026 import PFF_TO_GROUP  # noqa: E402

# how many of each group are on the field at once - from the two-deep's own slot counts
STARTERS = {"QB": 1, "RB": 1, "WR": 3, "TE": 1, "OL": 5,
            "DT": 2, "EDGE": 2, "LB": 2, "CB": 3, "SAF": 2}

# a swap needs a wide margin, not a nose. Below these the chart is left alone: a
# backup with 40 more snaps and a hair more WAR is noise, and the chart may know
# about a spring move we do not.
MIN_SNAPS = 200
MIN_SNAP_MARGIN = 50
MIN_WAR_MARGIN = 0.05


def healthy_backup_share(seasons_csv=f"{HERE}/hybrid_player_war.csv"):
    """Backup share of a unit's snaps in the healthiest quarter of unit-seasons.

    'Healthiest' = the starters carried the most of the load, which is the observable
    shadow of nobody important getting hurt.
    """
    w = pd.read_csv(seasons_csv)
    w = w[w.snaps > 0].copy()
    w["group"] = w.position.map(PFF_TO_GROUP)
    w = w[w.group.notna()]
    key = ["season", "team", "group"]
    w["rk"] = w.groupby(key).snaps.rank(ascending=False, method="first")
    w["share"] = w.snaps / w.groupby(key).snaps.transform("sum")
    w["bk"] = w.rk > w.group.map(STARTERS)
    st = w[~w.bk].groupby(key).share.sum().rename("st_share")
    w = w.merge(st, on=key)
    out = {}
    for g, d in w.groupby("group"):
        hi = d.st_share.quantile(0.75)
        h = d[d.st_share >= hi]
        out[g] = float(h[h.bk].share.sum() / h.share.sum())
    return pd.Series(out)


def fix_depth(d):
    """Swap a listed backup with a listed starter he clearly outplayed."""
    d = d.copy()
    d["is_starter"] = d.is_starter.astype(bool)
    d["sn"] = d.snaps_2025.fillna(0.0)
    d["w25"] = d.war_2025.fillna(0.0)
    swaps = []
    for (t, g), unit in d.groupby(["team", "broad_group"]):
        st = unit[unit.is_starter]
        bk = unit[~unit.is_starter]
        if st.empty or bk.empty:
            continue
        # repeatedly promote the best backup over the weakest starter while the gap holds
        st_idx = list(st.index)
        bk_idx = list(bk.sort_values("w25", ascending=False).index)
        for b in bk_idx:
            if not st_idx:
                break
            worst = min(st_idx, key=lambda i: d.at[i, "w25"])
            if (d.at[b, "sn"] >= MIN_SNAPS
                    and d.at[b, "sn"] - d.at[worst, "sn"] >= MIN_SNAP_MARGIN
                    and d.at[b, "w25"] - d.at[worst, "w25"] >= MIN_WAR_MARGIN):
                d.at[b, "is_starter"], d.at[worst, "is_starter"] = True, False
                swaps.append((t, g, d.at[b, "player"], d.at[worst, "player"]))
                st_idx.remove(worst)
                st_idx.append(b)
            else:
                break
    return d, swaps


def reweight(d, target):
    """Move share from backups to starters until the backup share matches the
    no-injury target, holding each room's total fixed."""
    d = d.copy()
    d["proj_war_raw"] = d.proj_war
    for (t, g), unit in d.groupby(["team", "broad_group"]):
        tgt = target.get(g)
        if tgt is None:
            continue
        st, bk = unit[unit.is_starter], unit[~unit.is_starter]
        tot = unit.proj_war.sum()
        if tot <= 0 or st.empty or bk.empty:
            continue
        cur_bk = bk.proj_war.sum()
        if cur_bk <= 0:
            continue
        want_bk = tot * tgt
        # only ever pull backups DOWN - the point is to stop paying for injuries
        if want_bk >= cur_bk:
            continue
        d.loc[bk.index, "proj_war"] = bk.proj_war * (want_bk / cur_bk)
        st_tot = st.proj_war.sum()
        if st_tot > 0:
            d.loc[st.index, "proj_war"] = st.proj_war * ((tot - want_bk) / st_tot)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", action="store_true",
                    help="correct is_starter on roster_2026.csv INSTEAD of reweighting "
                         "a projection. This has to run first: is_starter is a FEATURE "
                         "of project_2026_v2, so fixing it after the projection leaves "
                         "the mislabelled player carrying the value the label gave him.")
    ap.add_argument("--in", dest="src", default=f"{HERE}/projections_2026_blended.csv")
    ap.add_argument("--out", dest="out", default=f"{HERE}/projections_2026_final.csv")
    args = ap.parse_args()

    if args.roster:
        r = pd.read_csv(f"{HERE}/roster_2026.csv")
        r, swaps = fix_depth(r)
        print(f"depth-chart swaps on the roster: {len(swaps)}")
        for t, g, up, down in swaps[:10]:
            print(f"  {t:<16}{g:<5} {up}  over  {down}")
        if len(swaps) > 10:
            print(f"  ... and {len(swaps)-10} more")
        r.drop(columns=["sn", "w25"]).to_csv(f"{HERE}/roster_2026.csv", index=False)
        print(f"-> roster_2026.csv (re-run project_2026_v2.py now)")
        return

    src = args.src if os.path.exists(args.src) else f"{HERE}/projections_2026_v2.csv"
    d = pd.read_csv(src)
    print(f"in: {os.path.basename(src)}   {len(d)} slots")

    target = healthy_backup_share()
    print("\nno-injury backup share by group:")
    for g, v in target.sort_values().items():
        print(f"  {g:<6}{v*100:5.1f}%")

    d, swaps = fix_depth(d)
    print(f"\ndepth-chart swaps: {len(swaps)}")
    for t, g, up, down in swaps[:8]:
        print(f"  {t:<16}{g:<5} {up}  over  {down}")
    if len(swaps) > 8:
        print(f"  ... and {len(swaps)-8} more")

    before = d.groupby("broad_group").apply(
        lambda x: x[~x.is_starter].proj_war.sum() / x.proj_war.sum(),
        include_groups=False)
    d = reweight(d, target)
    after = d.groupby("broad_group").apply(
        lambda x: x[~x.is_starter].proj_war.sum() / x.proj_war.sum(),
        include_groups=False)
    cmp = pd.DataFrame({"backup WAR share before": before * 100,
                        "after": after * 100, "target": target * 100}).round(1)
    print("\n" + cmp.to_string())

    print(f"\nleague total WAR: {d.proj_war_raw.sum():.1f} -> {d.proj_war.sum():.1f} "
          f"(room totals are held fixed)")
    d.to_csv(args.out, index=False)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
