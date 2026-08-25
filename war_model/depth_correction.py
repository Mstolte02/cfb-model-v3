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

   The room TOTAL is held fixed.  The old implementation forced every room all the
   way to this league-average split.  That made a binary July depth chart overpower
   the player model: a strong listed backup could fall from first to fifty-first at
   his position while the starter inherited his value.  The current implementation
   treats the chart as one noisy role prior, blends it with prior usage and intrinsic
   player value, and publishes an opportunity range instead of pretending the listed
   1/2 is certain.

Run: ../venv/bin/python depth_correction.py [--in FILE] [--out FILE]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
from build_roster_2026 import PFF_TO_GROUP, norm_name  # noqa: E402

# how many of each group are on the field at once - from the two-deep's own slot counts.
# The five linemen are two tackles and three interior.
STARTERS = {"QB": 1, "RB": 1, "WR": 3, "TE": 1, "OT": 2, "IOL": 3,
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


AVAILABILITY = f"{HERE}/availability_2026.csv"


def apply_availability(d, path=AVAILABILITY):
    """Curated overrides for things no model can infer: injuries, and named starters.

    A two-deep scraped in July does not know that a player tore something in August,
    and no amount of reasoning over last season's snaps will discover it. This is the
    one place where a human statement about who is playing enters the pipeline.

      out      - unavailable for the season. Removed from the room entirely: his WAR
                 goes to zero and the snaps he would have taken are redistributed to
                 whoever is left, which is what actually happens.
      starter  - starts at his listed position, whatever the depth chart says.

    Returns (frame, notes). Absent file is not an error; most weeks there is nothing
    to override.
    """
    notes = []
    if not os.path.exists(path):
        return d, notes
    if os.path.abspath(path) == os.path.abspath(AVAILABILITY):
        events = f"{HERE}/availability_events_2026.csv"
        if os.path.exists(events):
            from materialize_availability import current_rows  # noqa: E402
            expected = pd.DataFrame(current_rows(events)).sort_values(
                ["team", "player"]).reset_index(drop=True)
            actual = pd.read_csv(path).sort_values(["team", "player"]).reset_index(drop=True)
            if list(actual.columns) != list(expected.columns) or not actual.equals(expected):
                raise ValueError("availability_2026.csv has drifted from the append-only "
                                 "event stream; run materialize_availability.py")
    ov = pd.read_csv(path)
    d = d.copy()
    if "pinned_starter" not in d.columns:
        d["pinned_starter"] = False
    d["pinned_starter"] = d.pinned_starter.fillna(False).astype(bool)
    # project_2026_v2 writes is_starter as 0/1, and assigning False into an int64
    # column raises rather than coercing. This used to be cast by whoever happened to
    # run before us; casting it here instead means the flag columns are the same dtype
    # no matter which stage hands us the frame.
    d["is_starter"] = d.is_starter.fillna(False).astype(bool)
    d["available"] = True
    for _, r in ov.iterrows():
        m = (d.team == r.team) & (d.player == r.player)
        if not m.any():
            notes.append(f"[warn] {r.player} ({r.team}) is not in the two-deep; ignored")
            continue
        if r.status == "out":
            d.loc[m, ["available", "is_starter", "pinned_starter"]] = False
            if "proj_war" in d.columns:      # absent on the roster frame
                d.loc[m, "proj_war"] = 0.0
            notes.append(f"OUT      {r.team:<16}{r.player:<22}{r.note}")
        elif r.status == "starter":
            d.loc[m, ["is_starter", "pinned_starter"]] = True
            notes.append(f"STARTER  {r.team:<16}{r.player:<22}{r.note}")
        else:
            notes.append(f"[warn] unknown status {r.status!r} for {r.player}; ignored")
    return d, notes


def pin_starters(d):
    """Apply every external statement about WHO STARTS to a roster frame.

    Runs on roster_2026.csv, BEFORE project_2026_v2, and that timing is the whole
    point: is_starter is an input FEATURE of the projection, so a roster carrying the
    wrong starter gets the wrong man projected as one, and no amount of relabelling
    afterwards fixes the value he was given. Syracuse is the worked example - Odom was
    flagged the starter, so he was projected as one at 0.97 and Angeli as a backup at
    0.10, and correcting the flag downstream merely handed Odom's starter-sized value
    to Angeli through the reweight.

    Two sources, same idea: the quarterback sheet names each team's starter, and
    availability_2026.csv records injuries and manual starter calls.
    """
    sys.path.insert(0, f"{HERE}/ea")
    from blend_projection import load_qb_sheet  # noqa: E402
    d = d.copy()
    d["is_starter"] = d.is_starter.astype(bool)
    d["pinned_starter"] = False

    qb_z = load_qb_sheet()
    named = pd.Series([(k, t) in qb_z for k, t in zip(d.key, d.team)], index=d.index)
    named &= d.broad_group == "QB"
    rooms = (d.broad_group == "QB") & d.team.isin(d.loc[named, "team"].unique())
    d.loc[rooms, "is_starter"] = named.loc[rooms]
    d.loc[named, "pinned_starter"] = True

    d, notes = apply_availability(d)
    return d, int(named.sum()), notes


def enforce_slots(d, value_col="proj_war"):
    """Each listed position starts exactly as many players as the two-deep says.

    fix_depth swaps within a POSITION GROUP, so on the offensive line it can promote a
    backup centre over a starting right guard - and then the room fields two centres
    and nobody at right guard. Seven OL rooms were in that state, Notre Dame among
    them. A backup centre outplaying a guard is not evidence that he plays guard.

    The expected count comes from the two-deep itself: however many depth-1 players a
    team lists at a position is how many start there. That keeps the genuinely
    multi-starter labels intact - a team listing two players at DT still starts two -
    without hard-coding a formation anywhere.

    THIS ONLY REPAIRS THE COUNT. A position already starting the right number of
    players is left completely alone, even if a backup there projects higher - that
    judgement belongs to fix_depth, which makes it against REALIZED snaps and production
    behind deliberate thresholds. Re-picking every starter here by projected WAR instead
    silently overrode those thresholds and rewrote 461 slots league-wide; restricted to
    genuine breakage it touches a couple of dozen.

    When a slot does have to be filled, the order is: pinned first, then whoever is
    already flagged (so a repair disturbs as little as possible), then the shallower
    depth, then projected value.
    """
    d = d.copy()
    if "available" not in d.columns:
        d["available"] = True
    if "pinned_starter" not in d.columns:
        d["pinned_starter"] = False
    d["is_starter"] = d.is_starter.astype(bool)
    fixed = []
    # materialized up front: the loop writes back into `d`, and iterating a live
    # groupby while mutating the frame it was built from is asking for trouble
    for (t, pos), room in list(d.groupby(["team", "roster_position"])):
        n_slots = int((room.depth == 1).sum())
        if n_slots == 0:
            continue
        before = set(room.index[room.is_starter & room.available])
        if len(before) == n_slots:
            continue                     # the count is right; not our business
        pool = room[room.available]
        order = pool.sort_values(["pinned_starter", "is_starter", "depth", value_col],
                                 ascending=[False, False, True, False])
        chosen = set(order.index[:n_slots])
        if chosen == set(room.index[room.is_starter]):
            continue
        d.loc[room.index, "is_starter"] = False
        d.loc[list(chosen), "is_starter"] = True
        out = [d.at[i, "player"] for i in before - chosen]
        into = [d.at[i, "player"] for i in chosen - before]
        fixed.append((t, pos, into, out))
    return d, fixed


def fix_depth(d):
    """Swap a listed backup with a listed starter he clearly outplayed.

    Two classes of player are off limits, and both are cases where someone knows
    something this function does not. A PINNED starter - named by the quarterback sheet
    or the availability file - cannot be demoted: those are statements about who starts
    THIS season, and last season's snap counts do not get to overrule them. Syracuse is
    why: Amari Odom out-snapped Steve Angeli in 2025, so this promoted him, and the
    reweight then handed him 95% of the room to finish as the best quarterback in FBS.
    An UNAVAILABLE player cannot be promoted, for the obvious reason.
    """
    d = d.copy()
    d["is_starter"] = d.is_starter.astype(bool)
    for c, default in (("pinned_starter", False), ("available", True)):
        if c not in d.columns:
            d[c] = default
        d[c] = d[c].fillna(default).astype(bool)
    d["sn"] = d.snaps_2025.fillna(0.0)
    d["w25"] = d.war_2025.fillna(0.0)
    swaps = []
    for (t, g), unit in list(d.groupby(["team", "broad_group"])):
        st = unit[unit.is_starter & ~unit.pinned_starter]
        bk = unit[~unit.is_starter & unit.available]
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


def _backup_share(d, group, rooms):
    """Backup share of a group's WAR, over a FIXED set of team rooms.

    The room set is passed in rather than derived, so the before and after figures
    describe the same teams. Deriving it from war_source would silently compare all
    134 QB rooms before the map against the 119 named ones after it.
    """
    g = d[(d.broad_group == group) & d.team.isin(rooms)]
    tot = g.proj_war.sum()
    return float(g[~g.is_starter].proj_war.sum() / tot) if tot else 0.0


def _chart_agreement(d):
    """Agreement between the two current depth-chart sources, by player.

    Missing from either source is deliberately not agreement.  The value is only a
    confidence input; it never decides who starts by itself.
    """
    maps = []
    for name in ("twodeep_2026.csv", "ourlads_2026.csv"):
        path = f"{HERE}/{name}"
        if not os.path.exists(path):
            continue
        c = pd.read_csv(path)
        c["key"] = c.player.map(norm_name)
        maps.append(c.drop_duplicates(["team", "key"]).set_index(["team", "key"]).depth)
    if len(maps) < 2:
        return pd.Series(np.nan, index=d.index)
    a, b = maps[:2]
    keys = pd.MultiIndex.from_arrays([d.team, d.player.map(norm_name)])
    av = a.reindex(keys).to_numpy()
    bv = b.reindex(keys).to_numpy()
    present = pd.notna(av) & pd.notna(bv)
    return pd.Series(np.where(present, av == bv, np.nan), index=d.index, dtype=float)


def reweight(d, target):
    """Softly allocate room value to expected 2026 roles.

    ``intrinsic_war`` is the player's schedule-neutral quality estimate and is never
    changed by the depth chart. ``proj_war`` is expected contribution after role.  A
    room's total is invariant, but the chart receives only 25-55% weight depending on
    source agreement and competition.  That avoids double-counting role already
    present in prior snaps and stops a disputed July ``2`` from erasing player quality.
    """
    d = d.copy()
    d["intrinsic_war"] = d.proj_war.clip(lower=0.0)
    d["proj_war"] = d.intrinsic_war
    d["chart_agreement"] = _chart_agreement(d)
    d["role_confidence"] = 0.35
    d["expected_snap_share"] = 0.0
    d["snap_share_low"] = 0.0
    d["snap_share_high"] = 0.0

    for (_, g), unit in d.groupby(["team", "broad_group"]):
        tgt = target.get(g)
        st, bk = unit[unit.is_starter & unit.available], unit[~unit.is_starter & unit.available]
        tot = float(unit.intrinsic_war.sum())
        if tgt is None or tot <= 0 or st.empty or bk.empty:
            n_on = float(STARTERS.get(g, 1))
            share = unit.intrinsic_war / tot if tot > 0 else pd.Series(0.0, index=unit.index)
            rate = (share * n_on).clip(0.0, 1.0)
            d.loc[unit.index, "expected_snap_share"] = rate
            d.loc[unit.index, "snap_share_low"] = (rate - .20).clip(0.0, 1.0)
            d.loc[unit.index, "snap_share_high"] = (rate + .20).clip(0.0, 1.0)
            continue

        agree = float(unit.chart_agreement.mean()) if unit.chart_agreement.notna().any() else 0.0
        weakest_starter = max(float(st.intrinsic_war.min()), 1e-9)
        competition = min(float(bk.intrinsic_war.max()) / weakest_starter, 1.5) / 1.5
        confidence = float(np.clip(.35 + .45 * agree - .25 * competition, .20, .85))
        chart_weight = .25 + .35 * confidence

        # The chart template says how the healthy historical room split is shared
        # between tiers.  Prior snaps choose players *within* a tier; intrinsic value
        # is the fallback for newcomers and zero-snap rooms.
        prior = unit.snaps_2025.fillna(0.0).clip(lower=0.0)
        base = prior if prior.sum() > 0 else unit.intrinsic_war
        template = pd.Series(0.0, index=unit.index)
        for tier, tier_share in ((st, 1.0 - float(tgt)), (bk, float(tgt))):
            weights = base.loc[tier.index]
            if weights.sum() <= 0:
                weights = unit.loc[tier.index, "intrinsic_war"]
            if weights.sum() <= 0:
                weights = pd.Series(1.0, index=tier.index)
            template.loc[tier.index] = tier_share * weights / weights.sum()

        intrinsic_share = unit.intrinsic_war / tot
        contribution_share = (1.0 - chart_weight) * intrinsic_share + chart_weight * template
        d.loc[unit.index, "proj_war"] = tot * contribution_share

        usage = prior / prior.sum() if prior.sum() > 0 else intrinsic_share
        opportunity = (1.0 - chart_weight) * usage + chart_weight * template
        rate = (opportunity * float(STARTERS.get(g, 1))).clip(0.0, 1.0)
        width = .06 + .22 * (1.0 - confidence)
        d.loc[unit.index, "role_confidence"] = confidence
        d.loc[unit.index, "expected_snap_share"] = rate
        d.loc[unit.index, "snap_share_low"] = (rate - width).clip(0.0, 1.0)
        d.loc[unit.index, "snap_share_high"] = (rate + width).clip(0.0, 1.0)

        assert abs(float(d.loc[unit.index, "proj_war"].sum()) - tot) < 1e-8

    # Backward-compatible name for analysis notebooks.  It now unambiguously means
    # quality before role rather than "whatever existed before the hard reweight".
    d["proj_war_raw"] = d.intrinsic_war
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
    ap.add_argument("--no-ea", action="store_true",
                    help="PFF-ONLY variant. Skips the quarterback value map, which is "
                         "the one step in this file carrying an outside opinion. Pair "
                         "it with --in projections_2026_v2.csv (the unblended "
                         "projection) to get a run with no EA anywhere in it. The "
                         "depth chart is deliberately NOT changed: both variants field "
                         "the same eleven men, so a difference between them is a "
                         "difference in ratings and nothing else.")
    args = ap.parse_args()

    if args.roster:
        r = pd.read_csv(f"{HERE}/roster_2026.csv")
        r, n_pin, notes = pin_starters(r)
        print(f"quarterbacks pinned from the sheet: {n_pin}")
        print(f"availability overrides: {len(notes)}")
        for n in notes:
            print(f"  {n}")

        r, swaps = fix_depth(r)
        print(f"\ndepth-chart swaps on the roster: {len(swaps)}")
        for t, g, up, down in swaps[:10]:
            print(f"  {t:<16}{g:<5} {up}  over  {down}")
        if len(swaps) > 10:
            print(f"  ... and {len(swaps)-10} more")

        # war_2025 rather than proj_war: the roster frame has no projection yet, which
        # is the entire reason this pass runs before project_2026_v2.
        r, fixed = enforce_slots(r, value_col="war_2025")
        print(f"\nposition slots repaired: {len(fixed)}")
        for t, pos, into, out in fixed[:10]:
            print(f"  {t:<16}{pos:<5} in: {', '.join(into) or '-':<24}"
                  f"out: {', '.join(out) or '-'}")
        if len(fixed) > 10:
            print(f"  ... and {len(fixed)-10} more")

        r.drop(columns=["sn", "w25"]).to_csv(f"{HERE}/roster_2026.csv", index=False)
        print(f"\n-> roster_2026.csv (re-run project_2026_v2.py now)")
        return

    src = args.src if os.path.exists(args.src) else f"{HERE}/projections_2026_v2.csv"
    d = pd.read_csv(src)
    print(f"in: {os.path.basename(src)}   {len(d)} slots")

    target = healthy_backup_share()
    print("\nno-injury backup share by group:")
    for g, v in target.sort_values().items():
        print(f"  {g:<6}{v*100:5.1f}%")

    # Order matters, and getting it wrong is silent. Availability first, so nobody who
    # is out can be promoted and nothing downstream picks him. Then fix_depth, which now
    # sees the pins and the injuries and will not overrule either - run the other way
    # round it re-swapped a pinned quarterback straight back out. Then slot repair, to
    # fill whatever the first two left open. Reweight runs last of all, because it moves
    # a room's WAR toward whoever is flagged as starting, so every correction to that
    # flag has to be settled before it.
    d, notes = apply_availability(d)
    print(f"\navailability overrides: {len(notes)}")
    for n in notes:
        print(f"  {n}")

    d, swaps = fix_depth(d)
    print(f"\ndepth-chart swaps: {len(swaps)}")
    for t, g, up, down in swaps[:8]:
        print(f"  {t:<16}{g:<5} {up}  over  {down}")
    if len(swaps) > 8:
        print(f"  ... and {len(swaps)-8} more")

    d, fixed = enforce_slots(d)
    print(f"\nposition slots repaired: {len(fixed)}")
    for t, pos, into, out in fixed[:10]:
        print(f"  {t:<16}{pos:<5} in: {', '.join(into) or '-':<24}"
              f"out: {', '.join(out) or '-'}")
    if len(fixed) > 10:
        print(f"  ... and {len(fixed)-10} more")

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

    print(f"\nleague total WAR: {d.intrinsic_war.sum():.1f} -> {d.proj_war.sum():.1f} "
          f"(room totals are held fixed)")

    if args.no_ea:
        d["war_source"] = "PFF"
        d.to_csv(args.out, index=False)
        print(f"\n--no-ea: quarterback value map SKIPPED; every number in this file is "
              f"the model's own.")
        print(f"-> {args.out}")
        return

    # ---- the quarterback sheet, last of all ---------------------------------
    # Applied here rather than in blend_projection because the reweight above scales
    # each team's starter by a team-specific factor (1.00 to 1.63), which re-sorted
    # the ordering the sheet had just imposed. Whatever runs last wins, and the sheet
    # should win. See blend_projection.apply_qb_map.
    sys.path.insert(0, f"{HERE}/ea")
    from blend_projection import apply_qb_map, load_qb_sheet, QB_COL  # noqa: E402
    sheet = load_qb_sheet()
    named_rooms = {t for _, t in sheet}
    qb_before = d[d.broad_group == "QB"].proj_war.sum()
    bk_before = _backup_share(d, "QB", named_rooms)
    d_pre = d.copy()
    d, n_qb, n_named = apply_qb_map(d)
    qb_after = d[d.broad_group == "QB"].proj_war.sum()

    # a permutation within the QB group, so the group total cannot move
    assert abs(qb_before - qb_after) < 1e-6, (
        f"QB group total moved by {abs(qb_before - qb_after):.6f}; the map is not a "
        f"permutation")

    named = d[d.war_source == "QB-avg"]
    rk = named.qb_z.rank(ascending=False)
    rho = named.qb_z.corr(named.proj_war, method="spearman")
    moved = int((named.proj_war.rank(ascending=False) - rk).abs().max())
    print(f"  QB group total {qb_before:.3f} -> {qb_after:.3f} (permutation, as required)")
    print(f"  Spearman vs the sheet's {QB_COL}: {rho:.4f}   "
          f"largest rank move: {moved}")
    # The AGGREGATE backup share over the named rooms cannot move - the map permutes
    # starter values among exactly those rooms, so their sum is invariant and the
    # backups are untouched. What moves is each room INDIVIDUALLY, and that is the
    # real cost of running the map last, so it is the number reported.
    per = [_backup_share(d, "QB", {t}) for t in sorted(named_rooms)]
    per_before = [_backup_share(d_pre, "QB", {t}) for t in sorted(named_rooms)]
    tgt = target["QB"]
    mad_b = float(np.mean([abs(x - tgt) for x in per_before]))
    mad_a = float(np.mean([abs(x - tgt) for x in per]))
    print(f"  backup share of the {len(named_rooms)} named QB rooms, aggregate: "
          f"{bk_before*100:.2f}% -> {_backup_share(d, 'QB', named_rooms)*100:.2f}% "
          f"(invariant under a permutation, as expected)")
    print(f"  per-room mean |share - {tgt*100:.2f}% target|: {mad_b*100:.2f}pp -> "
          f"{mad_a*100:.2f}pp  (this is what running the map last costs)")

    # The QB opinion is a player-quality input too.  Apply the same within-group
    # permutation to the intrinsic column independently, so QB rankings do not fall
    # back to the pre-sheet ordering while team contribution keeps the role allocation.
    q = d.copy()
    q["proj_war"] = q.intrinsic_war
    q, _, _ = apply_qb_map(q, verbose=False)
    d["intrinsic_war"] = q.proj_war
    d["proj_war_raw"] = d.intrinsic_war

    d.to_csv(args.out, index=False)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
