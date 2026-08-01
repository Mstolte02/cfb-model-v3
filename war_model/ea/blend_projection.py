"""Use EA's opinion for lightly-played players and ours for everyone else.

BELOW A SNAP THRESHOLD the projection is weakest - on the 2025 holdout it correlates
.634 with what a no-prior-snap player actually did against .805 for a 600+ snap one,
about twice the error relative to mean WAR - and EA is most different there, agreeing
with us .526 on no-history players and .903 on established starters. So this replaces
our ranking of the players under the threshold with EA's.

WHAT IS TAKEN FROM EA IS THE ORDERING, NOT THE SCALE. Inside the sub-300 bucket EA's
raw WAR has 0.8x our mean and 1.3x our spread, so splicing it in directly would both
shift the level and widen the distribution, and neither of those is something EA has
earned - its wins scale is inherited from our fit rather than fitted to anything. The
values are quantile-mapped onto the PFF distribution of the same bucket, so the
marginal distribution of WAR is unchanged and only WHO IS WHERE moves.

The mapping is done WITHIN POSITION GROUP. Mapping globally would let EA's view of how
a quarterback compares to a guard leak in, and that comparison is the one thing the
fitted facet weights already answer.

THIS IS UNVALIDATED. Every test in gap_analysis.py says EA is DIFFERENT in the tail;
none says it is BETTER, and the only accuracy test available - team ratings against
2025 results - had EA behind at .42 against .51. It is a hypothesis being run, not a
measured improvement.

Run: ../../venv/bin/python blend_projection.py [--threshold 300]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WAR_DIR = os.path.dirname(HERE)
sys.path.insert(0, WAR_DIR)
from build_roster_2026 import norm_name  # noqa: E402

SRC = f"{WAR_DIR}/projections_2026_v2.csv"
OUT = f"{WAR_DIR}/projections_2026_blended.csv"


def quantile_map(src_vals, target_vals):
    """Give each src value the target value at its own rank.

    Rank-based rather than a mean/sd match because the WAR distribution in this bucket
    is heavily right-skewed - most of these players are worth nearly nothing and a few
    are worth a lot - and a linear rescale would not preserve that shape.
    """
    if len(src_vals) == 0:
        return np.array([])
    order = np.argsort(np.argsort(src_vals.to_numpy()))
    return np.sort(target_vals.to_numpy())[order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=300,
                    help="prior-season snaps below which EA's ordering is used")
    args = ap.parse_args()

    pff = pd.read_csv(SRC)
    pff["key"] = pff.player.map(norm_name)
    ea = pd.read_csv(f"{HERE}/ea_player_war_2026.csv")
    e = (ea[["key", "team", "war", "overall"]]
         .rename(columns={"war": "ea_war", "overall": "ea_ovr"})
         .drop_duplicates(["key", "team"]))
    d = pff.merge(e, on=["key", "team"], how="left")
    d["prior_snaps"] = d.snaps_2025.fillna(0.0)

    low = (d.prior_snaps < args.threshold) & d.ea_war.notna()
    d["war_source"] = np.where(low, "EA", "PFF")
    d["proj_war_pff"] = d.proj_war

    # quantile-map inside each position group, within the low bucket only
    new = d.proj_war.copy()
    moved = 0
    for g, idx in d[low].groupby("broad_group").groups.items():
        sub = d.loc[idx]
        new.loc[idx] = quantile_map(sub.ea_war, sub.proj_war)
        moved += len(idx)
    d["proj_war"] = new

    n_low = int((d.prior_snaps < args.threshold).sum())
    print(f"threshold: {args.threshold} prior snaps")
    print(f"  slots below it:            {n_low}")
    print(f"  of those, EA-ranked:       {moved}  ({moved/n_low*100:.1f}%)")
    print(f"  no EA rating, kept as PFF: {n_low - moved}")
    print(f"  slots at or above:         {len(d) - n_low}  (untouched)")

    # the marginal distribution must be unchanged - that is the point of the mapping
    a, b = d.proj_war_pff, d.proj_war
    print(f"\nleague total WAR: {a.sum():.1f} -> {b.sum():.1f}   "
          f"(mapping is rank-preserving within group, so this should barely move)")
    print(f"Spearman old vs new, all slots: {a.corr(b, method='spearman'):.3f}")
    sub = d[low]
    print(f"  within the EA-ranked bucket:  "
          f"{sub.proj_war_pff.corr(sub.proj_war, method='spearman'):.3f}")

    tp = d.groupby("team").proj_war.sum()
    to = d.groupby("team").proj_war_pff.sum()
    shift = (tp - to).sort_values()
    print(f"\nteam WAR shifts most (EA's ranking of young players moves these rosters):")
    for t, v in list(shift.tail(5).items())[::-1]:
        print(f"  {t:<20}{v:+.2f}")
    for t, v in shift.head(5).items():
        print(f"  {t:<20}{v:+.2f}")

    d.drop(columns=["key"]).to_csv(OUT, index=False)
    print(f"\n-> {OUT}")
    print("   src/data/war.py picks this up when config.EA_BLEND_SNAPS is set; "
          "delete it or set that to 0 to go back.")


if __name__ == "__main__":
    main()
