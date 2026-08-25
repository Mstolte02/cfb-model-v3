"""Choose whose opinion of each player the 2026 projection uses.

Two rules, applied in order, each replacing our ORDERING of a set of players with
someone else's:

  1. Quarterbacks take the `Average` column of qbs_2026.xlsx - a composite z-score
     across PFSN, PFF, EPA, an execs poll and EA - for the starter it names.
  2. Everyone else: below --threshold prior snaps, EA's ranking; at or above it, ours.

Everything not caught by those two is this model's own number, out of the PFF and CFBD
facets. That is the whole rule now: PFF everywhere, except the quarterbacks and except
the players with no track record to judge them on.

THERE USED TO BE A RULE 0 - OL, WR and DT took EA's ranking at EVERY snap level, not
just for the unproven - and it has been removed. It was the one rule here that
overrode our own numbers for players we have plenty of evidence about, and it never
had a result behind it: every test in gap_analysis.py says EA is DIFFERENT in the
tail, none says it is BETTER, and the only accuracy test available - team ratings
against 2025 results - had EA behind at .42 against .51. Dropping it moves the proven
OL, WR and DT back onto PFF and leaves EA doing the one job the evidence supports,
which is ranking players we cannot rank ourselves. main() counts and prints how many
slots that is on the build in hand, rather than this docstring naming a figure that
goes stale the next time the roster changes.

WHAT IS TAKEN FROM THE OTHER SOURCE IS THE ORDERING, NOT THE SCALE. EA's raw WAR is on
its own scale (inside the sub-300 bucket, 0.8x our mean and 1.3x our spread) and the
quarterback column is a z-score, which is not a wins scale at all. Both are
quantile-mapped onto OUR proj_war values for the same set of players, so the mapping is
a permutation: the multiset of WAR is identical before and after, the league total does
not move, and only WHO IS WHERE changes. That is the alignment guarantee, and main()
asserts it rather than trusting it.

The mapping is done WITHIN POSITION GROUP. Mapping globally would let EA's view of how
a quarterback compares to a guard leak in, and that comparison is the one thing the
fitted facet weights already answer.

RULE 2 IS UNVALIDATED and rule 1 is a judgement rather than a measurement. This file
implements them and checks the scale; it does not judge them.

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

# The quarterback sheet, copied into the repo so the build does not read from
# ~/Downloads. One row per team, naming that team's starter and scoring him on five
# independent opinions; `Average` is the mean of the five z-scores.
#
# This was briefly switched to `PFF Z` alone and switched back. The single-source
# column is also the sparser one - ten of 124 named starters have no PFF grade, and
# on `Average` they still get an opinion from the other four sources.
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
    """{(player key, CFBD team): PFF z}, one row per team's named starter.

    Returns an empty dict rather than raising if the sheet is absent, so a clone that
    has never been given it still builds - the quarterbacks simply stay on PFF.

    Single-source columns carry the literal string "Unknown" where that source has no
    opinion, which the old `Average` column hid by averaging whatever was present. A
    quarterback with no PFF grade - a true freshman, a JUCO arrival - is dropped here
    and keeps the projection's own number, which is the right answer: there is no PFF
    evidence to override it with.
    """
    if not os.path.exists(QB_XLSX):
        print(f"  [warn] {os.path.basename(QB_XLSX)} not found; quarterbacks stay PFF.")
        return {}
    q = pd.read_excel(QB_XLSX)
    graded = pd.to_numeric(q[QB_COL], errors="coerce")
    missing = int(graded.isna().sum())
    if missing:
        print(f"  [info] {missing} of {len(q)} named starters have no {QB_COL}; "
              f"they keep the projection's own value.")
    q = q[graded.notna()].copy()
    q[QB_COL] = graded[graded.notna()]
    q["key"] = q.Player.map(norm_name)
    q["cfbd_team"] = q.Team.replace(QB_TEAM_ALIAS)
    return dict(zip(zip(q.key, q.cfbd_team), q[QB_COL].astype(float)))


def apply_qb_map(d, verbose=True):
    """Rule 2, as a step that can be run on its own frame at its own point.

    IT RUNS LAST, AFTER depth_correction's reweight, and that ordering is the whole
    reason it lives in a function instead of inside main(). The reweight scales each
    team's starting quarterback by (room_total - room_total*backup_target)/starter_total,
    which is a TEAM-SPECIFIC multiplier - 1.00 to 1.63, median 1.027, driven entirely
    by how much WAR that room's backups happened to carry. Running it after this map
    re-sorted the quarterbacks the sheet had just ordered: Spearman against the sheet
    fell from 0.9999 to 0.9885 and the largest single move was 20 places, with Dante
    Moore jumping two men on a x1.144 and CJ Bailey climbing eight on a x1.187.

    So the map goes last and is the final word on who is where. The cost is that the
    reweight's backup share no longer lands exactly on target for the named rooms -
    permuting starters across teams changes each room's total - and main() reports how
    far it drifts. That is the right way round: the backup share is a correction for
    injuries we are not projecting, and being a few tenths of a point off it matters
    much less than the starting quarterbacks being in the wrong order.

    Returns (frame, n_matched, n_named).
    """
    d = d.copy()
    if "key" not in d.columns:
        d["key"] = d.player.map(norm_name)
    qb_z = load_qb_sheet()
    d["qb_z"] = [qb_z.get((k, t)) for k, t in zip(d.key, d.team)]

    new = d.proj_war.copy()
    source = d.war_source.copy() if "war_source" in d.columns else pd.Series(
        "PFF", index=d.index)
    qb = (d.broad_group == "QB") & d.qb_z.notna()
    apply_map(d, qb, "qb_z", "QB-avg", new, source)
    d["proj_war"] = new
    d["war_source"] = source

    if verbose:
        n = int(qb.sum())
        print(f"\nquarterback sheet applied AFTER the reweight: {n} of {len(qb_z)} "
              f"named starters matched")
        if n < len(qb_z):
            hit = set(zip(d.loc[qb, "key"], d.loc[qb, "team"]))
            for k, t in sorted(set(qb_z) - hit):
                print(f"  [miss] {k} ({t}) is not in our two-deep; that team's QB "
                      f"stays as projected")
    return d, int(qb.sum()), len(qb_z)


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
    d["proj_war_pff"] = d.proj_war

    new = d.proj_war.copy()
    source = pd.Series("PFF", index=d.index)

    # --- rule 1 IS NOT HERE ------------------------------------------------------
    # The quarterback sheet is applied by apply_qb_map(), which depth_correction calls
    # after its reweight, because the reweight's team-specific multiplier was undoing
    # the ordering this map imposes. See apply_qb_map's docstring.
    #
    # The OTHER half of what the sheet says - who starts - still happens well before
    # this file, in depth_correction.pin_starters(), which writes is_starter onto
    # roster_2026.csv BEFORE project_2026_v2 runs. That timing is not negotiable:
    # is_starter is an input FEATURE of the projection, so a roster carrying the wrong
    # starter gets the wrong man projected as one. Syracuse is the worked example -
    # Odom was flagged starter on 2025 snaps and came out the best quarterback in FBS
    # at 1.83, above Julian Sayin. Nothing about where the VALUE map runs touches that;
    # the pin is already on the frame when this file opens it.
    qb_slots = d.broad_group == "QB"

    # --- rule 2: the snap-threshold blend, on everyone who is not a quarterback ----
    # Quarterbacks are excluded outright - rule 1 will claim them at the end of the
    # pipeline, and letting EA reorder them here would be overwritten anyway.
    low = (d.prior_snaps < args.threshold) & d.ea_war.notna() & ~qb_slots
    n_low_ea = apply_map(d, low, "ea_war", "EA", new, source)

    d["proj_war"] = new
    d["war_source"] = source

    # ---- what happened -----------------------------------------------------------
    n_qb_slots = int((d.broad_group == "QB").sum())
    n_qb_named = len(load_qb_sheet())
    print(f"rule 1  quarterback sheet: deferred to depth_correction, which applies it "
          f"after the reweight ({n_qb_named} named starters, {n_qb_slots} QB slots)")
    print(f"rule 2  EA under {args.threshold} prior snaps: {n_low_ea} slots")
    print(f"        untouched (PFF): {int((source == 'PFF').sum())} of {len(d)}")

    # The proven slots that the old OL/WR/DT full-group rule used to take off PFF and
    # hand to EA. Reported so the effect of dropping that rule stays visible instead of
    # becoming invisible the moment it is gone.
    was_full_ea = (d.broad_group.isin(("OL", "OT", "IOL", "WR", "DT"))
                   & d.ea_war.notna() & (d.prior_snaps >= args.threshold))
    print(f"        of those PFF slots, {int(was_full_ea.sum())} are proven OL/WR/DT "
          f"that the removed full-group rule would have given to EA")

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
    for label in ("QB-avg", "EA"):
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
