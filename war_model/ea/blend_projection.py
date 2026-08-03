"""Choose whose opinion of each player the 2026 projection uses.

Three rules, applied in order, each replacing our ORDERING of a set of players with
someone else's:

  1. OL, WR and DT take EA's ranking at every snap level, not just for the unproven.
  2. Quarterbacks take the `Average` column of qbs_2026.xlsx - a composite z-score
     across PFSN, PFF, EPA, an execs poll and EA - for the starter it names.
  3. Everyone else keeps the older rule: below --threshold prior snaps, EA's ranking;
     at or above it, ours.

WHAT IS TAKEN FROM THE OTHER SOURCE IS THE ORDERING, NOT THE SCALE, and that is what
makes rules 1 and 2 safe to state so bluntly. EA's raw WAR is on its own scale (inside
the sub-300 bucket, 0.8x our mean and 1.3x our spread) and the quarterback column is a
z-score, which is not a wins scale at all. Both are quantile-mapped onto OUR proj_war
values for the same set of players, so the mapping is a permutation: the multiset of
WAR is identical before and after, the league total does not move, and only WHO IS
WHERE changes. That is the alignment guarantee, and main() asserts it rather than
trusting it.

The mapping is done WITHIN POSITION GROUP. Mapping globally would let EA's view of how
a quarterback compares to a guard leak in, and that comparison is the one thing the
fitted facet weights already answer.

RULE 3 IS UNVALIDATED, and rules 1 and 2 are asserted rather than measured. Every test
in gap_analysis.py says EA is DIFFERENT in the tail; none says it is BETTER, and the
only accuracy test available - team ratings against 2025 results - had EA behind at .42
against .51. Rules 1 and 2 are here because they were tested elsewhere and chosen; this
file implements them and checks the scale, it does not judge them.

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

# Position groups where EA's ranking is used for EVERY player, not only the unproven.
FULL_EA_GROUPS = ("OL", "WR", "DT")

# The quarterback sheet, copied into the repo so the build does not read from
# ~/Downloads. One row per team, naming that team's starter and scoring him on five
# independent opinions; `Average` is the mean of the five z-scores.
QB_XLSX = f"{HERE}/qbs_2026.xlsx"
QB_COL = "Average"

# The sheet writes some schools out in full where CFBD abbreviates. Names are matched
# on player AND team, so these have to resolve or seven starters silently keep PFF.
QB_TEAM_ALIAS = {
    "Mississippi": "Ole Miss",
    "North Carolina State": "NC State",
    "Hawaii": "Hawai'i",
    "Central Florida": "UCF",
    "Appalachian State": "App State",
    "Miami (Ohio)": "Miami (OH)",
    "Louisiana-Monroe": "UL Monroe",
}


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


def load_qb_sheet():
    """{(player key, CFBD team): composite z}, one row per team's named starter.

    Returns an empty dict rather than raising if the sheet is absent, so a clone that
    has never been given it still builds - the quarterbacks simply stay on PFF.
    """
    if not os.path.exists(QB_XLSX):
        print(f"  [warn] {os.path.basename(QB_XLSX)} not found; quarterbacks stay PFF.")
        return {}
    q = pd.read_excel(QB_XLSX)
    q = q[q[QB_COL].notna()]
    q["key"] = q.Player.map(norm_name)
    q["cfbd_team"] = q.Team.replace(QB_TEAM_ALIAS)
    return dict(zip(zip(q.key, q.cfbd_team), q[QB_COL].astype(float)))


def apply_map(d, mask, src_col, label, new, source):
    """Quantile-map `src_col` onto our own proj_war for the masked rows, by group.

    Returns the number of slots moved. Rows are grouped by broad_group so a source's
    view of how a guard compares to a receiver never enters; within a group the result
    is a permutation of the values already there.
    """
    moved = 0
    for _, idx in d[mask].groupby("broad_group").groups.items():
        sub = d.loc[idx]
        new.loc[idx] = quantile_map(sub[src_col], sub.proj_war)
        source.loc[idx] = label
        moved += len(idx)
    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=300,
                    help="prior-season snaps below which EA's ordering is used, for "
                         "the groups not covered by a full-group rule")
    args = ap.parse_args()

    pff = pd.read_csv(SRC)
    pff["key"] = pff.player.map(norm_name)
    ea = pd.read_csv(f"{HERE}/ea_player_war_2026.csv")
    e = (ea[["key", "team", "war", "overall"]]
         .rename(columns={"war": "ea_war", "overall": "ea_ovr"})
         .drop_duplicates(["key", "team"]))
    d = pff.merge(e, on=["key", "team"], how="left")
    d["prior_snaps"] = d.snaps_2025.fillna(0.0)
    d["proj_war_pff"] = d.proj_war

    qb_z = load_qb_sheet()
    d["qb_z"] = [qb_z.get((k, t)) for k, t in zip(d.key, d.team)]

    new = d.proj_war.copy()
    source = pd.Series("PFF", index=d.index)

    # --- rule 1: EA outright for the three groups, at every snap level -------------
    full_ea = d.broad_group.isin(FULL_EA_GROUPS) & d.ea_war.notna()
    n_full = apply_map(d, full_ea, "ea_war", "EA-full", new, source)

    # --- rule 2: the quarterback sheet, for the starters it names ------------------
    qb = (d.broad_group == "QB") & d.qb_z.notna()
    n_qb = apply_map(d, qb, "qb_z", "QB-avg", new, source)

    # --- rule 3: the old snap-threshold blend, on whatever is left ----------------
    # A row claimed above is excluded, not re-mapped: rule 3 acting on a subset of an
    # already-permuted group would reshuffle values rule 1 or 2 had just placed.
    claimed = full_ea | qb
    low = ((d.prior_snaps < args.threshold) & d.ea_war.notna() & ~claimed
           & ~d.broad_group.isin(FULL_EA_GROUPS) & (d.broad_group != "QB"))
    n_low_ea = apply_map(d, low, "ea_war", "EA", new, source)

    d["proj_war"] = new
    d["war_source"] = source

    # ---- what happened -----------------------------------------------------------
    n_qb_slots = int((d.broad_group == "QB").sum())
    n_qb_named = len(qb_z)
    print(f"rule 1  EA outright for {', '.join(FULL_EA_GROUPS)}:")
    for g in FULL_EA_GROUPS:
        tot = int((d.broad_group == g).sum())
        got = int((full_ea & (d.broad_group == g)).sum())
        print(f"          {g:<4}{got:>5} of {tot:<5} slots EA-ranked "
              f"({tot - got} have no EA rating, kept PFF)")
    print(f"rule 2  quarterback sheet: {n_qb} of {n_qb_named} named starters matched "
          f"into {n_qb_slots} QB slots")
    if n_qb < n_qb_named:
        named = set(qb_z)
        hit = set(zip(d.loc[qb, "key"], d.loc[qb, "team"]))
        for k, t in sorted(named - hit):
            print(f"          [miss] {k} ({t}) is not in our two-deep; that team's "
                  f"QB stays PFF")
    print(f"rule 3  EA under {args.threshold} snaps elsewhere: {n_low_ea} slots")
    print(f"        untouched (PFF): {int((source == 'PFF').sum())} of {len(d)}")

    # ---- the scale check ---------------------------------------------------------
    # Every rule is a within-group permutation, so each group's WAR total must be
    # EXACTLY what it was. This is the whole basis for saying EA's numbers and the
    # quarterback z-scores are on our wins scale, so it is asserted, not eyeballed.
    print("\nscale check - each rule permutes values within a group, so every group "
          "total must be unchanged:")
    print(f"  {'group':<7}{'slots':>6}{'total WAR before':>19}{'after':>10}"
          f"{'mean':>8}{'sd':>7}{'moved':>7}")
    worst = 0.0
    for g, sub in d.groupby("broad_group"):
        before, after = sub.proj_war_pff.sum(), sub.proj_war.sum()
        worst = max(worst, abs(before - after))
        n_moved = int((sub.war_source != "PFF").sum())
        print(f"  {g:<7}{len(sub):>6}{before:>19.3f}{after:>10.3f}"
              f"{sub.proj_war.mean():>8.3f}{sub.proj_war.std():>7.3f}{n_moved:>7}")
    assert worst < 1e-6, f"group WAR total moved by {worst:.6f}; the map is not a permutation"
    print(f"  largest group-total drift: {worst:.2e}  (a permutation, as required)")

    a, b = d.proj_war_pff, d.proj_war
    print(f"\nleague total WAR: {a.sum():.2f} -> {b.sum():.2f}")
    print(f"Spearman old vs new, all slots: {a.corr(b, method='spearman'):.3f}")
    for label in ("EA-full", "QB-avg", "EA"):
        sub = d[d.war_source == label]
        if len(sub) > 1:
            print(f"  within {label:<8}({len(sub):>4} slots): "
                  f"{sub.proj_war_pff.corr(sub.proj_war, method='spearman'):>6.3f}")

    tp = d.groupby("team").proj_war.sum()
    to = d.groupby("team").proj_war_pff.sum()
    shift = (tp - to).sort_values()
    print(f"\nteam WAR shifts most (the league total is fixed, so these are transfers "
          f"between rosters, not new wins):")
    for t, v in list(shift.tail(8).items())[::-1]:
        print(f"  {t:<20}{v:+.2f}")
    print("  ...")
    for t, v in shift.head(8).items():
        print(f"  {t:<20}{v:+.2f}")

    d.drop(columns=["key"]).to_csv(OUT, index=False)
    print(f"\n-> {OUT}")
    print("   src/data/war.py picks this up when config.EA_BLEND_SNAPS is set; "
          "delete it or set that to 0 to go back.")


if __name__ == "__main__":
    main()
