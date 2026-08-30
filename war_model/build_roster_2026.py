"""Stage 6: the 2026 roster, joined to our own WAR history.

The two-deep workbook is used ONLY as the roster source - which players are on which
team at which position and depth in 2026. Every valuation number comes from the PFF
Massey WAR build in stages 1-3; the workbook's own PFSN scores and weights are dropped.
"""
import json, os, re, unicodedata
import numpy as np
import pandas as pd

import artifacts
import scrape_twodeep as twodeep     # hybrid-label tables and OFFENSE, defined once

HERE = os.path.dirname(os.path.abspath(__file__))
from paths import TWODEEP_2026, require

XL = str(require(TWODEEP_2026, "the 2026 two-deep workbook", "CFB_TWODEEP_2026"))

# columns we take from the workbook: roster identity only
ROSTER_COLS = ["team", "conference", "unit", "roster_position", "broad_group",
               "depth", "player_display", "eligibility", "transfer"]

TEAM_ALIAS = {
    "Appalachian State": "App State", "Central Florida": "UCF", "Connecticut": "UConn",
    "Hawaii": "Hawai'i", "Louisiana-Monroe": "UL Monroe", "Miami (Ohio)": "Miami (OH)",
    "Mississippi": "Ole Miss", "North Carolina State": "NC State",
    "San Jose State": "San José State",
    # first year in FBS for 2026 - no prior FBS history in the PFF exports
    "North Dakota State": "North Dakota State", "Sacramento State": "Sacramento State",
}

# PFF position -> two-deep broad group
PFF_TO_GROUP = {
    "QB": "QB", "HB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
    "T": "OT", "G": "IOL", "C": "IOL",
    "DI": "DT", "ED": "EDGE", "LB": "LB", "CB": "CB", "S": "SAF",
}

# The valuation groups collapsed to the coarser ones used for IDENTITY matching only.
# See the name+position merge in main() for why the two differ.
COARSE_GROUP = {"OT": "OL", "IOL": "OL"}

SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", " ").replace("'", "").replace("-", " ")
    s = SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", s)).strip()


def parse_eligibility(e):
    """'RS SR/TR' -> class SR, redshirt True, transfer True."""
    e = str(e).upper()
    transfer = "/TR" in e or e.endswith("TR")
    rs = e.startswith("RS")
    cls = None
    for c in ("FR", "SO", "JR", "SR", "GR"):
        if re.search(rf"\b{c}\b", e.replace("/TR", " ")):
            cls = c
            break
    return pd.Series({"class": cls, "redshirt": rs, "is_transfer": transfer})


# Roster sources, best first. Each scraper emits the same columns, so this is a
# preference order rather than three code paths.
#
# TWO-DEEP replaced Ourlads as the primary. It charts all 138 FBS programs where
# Ourlads had 136 - North Dakota State and Sacramento State play their first FBS
# season in 2026 and were simply missing - and its position labels resolve the line
# into tackles and interior for every charted slot, which is what lets broad_group
# carry OT and IOL separately. Ourlads stays as the fallback and is still granular
# enough to do the same; the workbook, a 27-June export that had already drifted, is
# the last resort.
SOURCES = [("twodeep_2026.csv", "TWO-DEEP charts"),
           ("ourlads_2026.csv", "Ourlads charts")]


def load_two_deep():
    """The 2026 two-deep, preferring a scraped chart over the workbook."""
    for fname, label in SOURCES:
        scraped = f"{HERE}/{fname}"
        if not os.path.exists(scraped):
            continue
        d = pd.read_csv(scraped)
        d = d[d.depth <= 2]          # the workbook was a two-deep; match it
        conf = {t["school"]: t.get("conference") for t in
                json.load(open(f"{HERE}/cfbd_cache/teams_2026.json"))} \
            if os.path.exists(f"{HERE}/cfbd_cache/teams_2026.json") else {}
        d["conference"] = d.team.map(conf).fillna("—")
        d["eligibility"] = d["class"].fillna("")
        print(f"roster source: scraped {label} "
              f"({d.team.nunique()} teams, {len(d)} slots)")
        return d
    raw = pd.read_excel(XL, "Weighted Two Deep")
    d = raw[ROSTER_COLS].copy()
    d["team"] = d.team.replace(TEAM_ALIAS)
    d = d.rename(columns={"player_display": "player"})
    d = d[d.player.notna() & (d.player.astype(str).str.strip() != "")]
    d[["class", "redshirt", "is_transfer"]] = d.eligibility.apply(parse_eligibility)
    print(f"roster source: workbook ({d.team.nunique()} teams, {len(d)} slots)")
    return d


def refine_hybrid_groups(ros, war):
    """Re-group the scheme-nickname slots from PFF's own position for the same man.

    The scraper resolves a nickname per (team, label) off the bios, which is the right
    unit for a scheme name - one programme's MONEY is one job. IT IS THE WRONG UNIT FOR
    A NICKEL. Measured against PFF, a nickel room does not hold one job: 77 of the 126
    matched NB slots are corners and 49 are safeties, and no reading of the room fixes
    that because the split is between MEN, not between programmes. Resolving NB per
    room scored 60.5% where per-player PFF is exact by construction.

    So where a hybrid-labelled man is already matched to PFF history, his own most
    recent PFF position wins. That is also the taxonomy the facet weights are fitted in
    and the one the projection's training rows carry, so this closes a train/serve gap
    rather than opening one: Chris Cole's history rows said LB while the slot serving
    the model said CB.

    Room-mates who match nothing - true freshmen, FCS arrivals - take the PFF majority
    of the men who did match, and keep the scraper's answer if nobody did.

    Only slots the scraper flagged as hybrid are touched, and only within the range the
    label can mean. Nathan VanTimmeren is Central Michigan's LOLB in 2026 and PFF's tight
    end in 2024 and 2025, because he changed position; taking PFF's word there put an
    outside linebacker in the tight end room. History says what a man WAS, and the chart
    is the better authority on what he is about to be - so PFF only gets to settle the
    question the nickname left open, not to reopen a settled one.
    """
    if "group_source" not in ros.columns:      # Ourlads fallback carries no flag
        print("\nhybrid regrouping skipped: roster source has no group_source column")
        return ros

    hy = ros.group_source.fillna("label") != "label"
    if not hy.any():
        return ros

    latest = war.drop_duplicates("player_id")[["player_id", "group"]]
    pff = (ros[["pid_team"]].merge(latest, left_on="pid_team", right_on="player_id",
                                   how="left")["group"].to_numpy())
    ros = ros.assign(_pff_group=pff)

    # the same range the scraper allows the bios to vote within, read from one place
    allowed = ros.roster_position.map(
        lambda p: (twodeep.HYBRID_ALLOWED.get(twodeep.HYBRID_FAMILY.get(p), set())
                   | twodeep.HYBRID_ALLOWED_EXTRA.get(p, set())))
    ros.loc[[g not in a for g, a in zip(ros._pff_group, allowed)],
            "_pff_group"] = np.nan

    before = ros.broad_group.copy()
    # per player where he is known...
    ros.loc[hy & ros._pff_group.notna(), "broad_group"] = \
        ros.loc[hy & ros._pff_group.notna(), "_pff_group"]
    # ...and the room's PFF majority for the men who are not
    room = (ros[hy & ros._pff_group.notna()]
            .groupby(["team", "roster_position"])._pff_group
            .agg(lambda s: s.value_counts().idxmax()))
    fill = hy & ros._pff_group.isna()
    ros.loc[fill, "broad_group"] = (
        pd.MultiIndex.from_arrays([ros.loc[fill, "team"],
                                   ros.loc[fill, "roster_position"]])
        .map(room).to_numpy())
    ros["broad_group"] = ros.broad_group.fillna(before)
    # regrouping can move a slot across the ball, so the unit column moves with it
    ros["unit"] = np.where(ros.broad_group.isin(twodeep.OFFENSE), "OFF", "DEF")

    moved = ros.broad_group != before
    print(f"\nhybrid slots regrouped from PFF: {int(moved.sum())} of {int(hy.sum())}")
    if moved.any():
        m = (ros[moved].assign(was=before[moved])
             .groupby(["roster_position", "was", "broad_group"]).size()
             .sort_values(ascending=False))
        print(m.head(15).to_string())
    return ros.drop(columns="_pff_group")


def main():
    ros = load_two_deep()
    ros["is_starter"] = ros.depth == 1
    ros["key"] = ros.player.map(norm_name)
    print(f"2026 two-deep slots: {len(ros)}   teams: {ros.team.nunique()}")
    print(f"  class: {dict(ros['class'].value_counts(dropna=False))}")
    print(f"  transfers: {int(ros.is_transfer.sum())}   redshirts: {int(ros.redshirt.sum())}")

    # ---- our WAR history -------------------------------------------------
    war = pd.read_csv(f"{HERE}/{artifacts.PLAYER_WAR}")
    war["group"] = war.position.map(PFF_TO_GROUP)
    war["key"] = war.player.map(norm_name)

    # most recent season first, so a player's latest team wins a tie
    war = war.sort_values("season", ascending=False)

    # match 1: name + 2025 team (strongest); match 2: name + group across any season.
    #
    # BOTH REQUIRE THE KEY TO BE UNIQUE. They were `drop_duplicates`, which does not
    # resolve an ambiguity - it hides one, by keeping whichever row sorted first and
    # handing that player's entire WAR history to a stranger with the same name. Two
    # players sharing a key means we do not know which is listed, and the honest
    # answer is no match; the two-deep slot then goes through the projection's
    # no-history path, which is what it is for.
    def unique_by(frame, keys):
        n = frame.groupby(keys).player_id.transform("nunique")
        ok = frame[n == 1].drop_duplicates(keys)
        return ok, int((n > 1).sum())

    m25, amb_team = unique_by(war[war.season == 2025], ["key", "team"])
    ros = ros.merge(
        m25[["key", "team", "player_id"]].rename(columns={"player_id": "pid_team"}),
        on=["key", "team"], how="left")

    ros = refine_hybrid_groups(ros, war)

    # THE NAME+POSITION MATCH USES THE COARSE GROUP, which collapses OT and IOL back
    # to one line. Its job is identity, not valuation: it exists to stop a receiver
    # inheriting a quarterback's history, and a guard who moves out to tackle is the
    # same man. Matching on the split group refused him instead, because his PFF rows
    # say G and his 2026 slot says OT, and it cost the line 4.8 points of coverage -
    # every one of those a real player pushed onto the no-history path that exists for
    # players we have never seen. What the split is FOR is pricing the two jobs
    # differently once the history is attached, and that happens downstream of here.
    war["match_group"] = war.group.replace(COARSE_GROUP)
    ros["match_group"] = ros.broad_group.replace(COARSE_GROUP)
    by_key_group, amb_group = unique_by(war, ["key", "match_group"])
    by_key_group = by_key_group[["key", "match_group", "player_id"]]
    ros = ros.merge(by_key_group.rename(columns={"player_id": "pid_group"}),
                    on=["key", "match_group"], how="left")
    print(f"  ambiguous history keys refused: {amb_team} on name+team, "
          f"{amb_group} on name+position")

    # No name-only fallback: it matched a South Carolina EDGE and a Delaware OL to the
    # same id. Every match must agree on either team or position group.
    ros["player_id"] = ros.pid_team.fillna(ros.pid_group)
    ros["match_type"] = np.select(
        [ros.pid_team.notna(), ros.pid_group.notna()],
        ["name+team", "name+position"], default="unmatched")

    # ---- is_transfer: A NEW TEAM THIS YEAR, not "has ever transferred" ----------
    #
    # THIS IS A MODEL FEATURE AND THE TWO SIDES HAD DIFFERENT DEFINITIONS. The
    # training side in project_2026_v2 computes it as
    #     prev_team.notna() & (prev_team != team)
    # - the player played somewhere else LAST season - while this file used to pass
    # through the depth chart's "/TR" marker, which means the player transferred at
    # some point in his career and stays true for as long as he is on the roster. That
    # put the serving flag at 55.6% of slots against a training rate less than half
    # that, so the feature meant one thing when the model learned it and another when
    # it was asked to use it. It is the same shape of defect as the is_starter leak.
    #
    # Recomputed here the same way the training side does it, off the same PFF history,
    # so the definitions are identical by construction rather than by coincidence. A
    # player with no 2025 FBS history - a true freshman, or an FCS/JUCO arrival - has no
    # prev_team and is not a transfer, which is exactly what training does with him.
    prev, amb_prev = unique_by(war[war.season == 2025], ["key"])
    ros = ros.merge(prev[["key", "team"]].rename(columns={"team": "prev_team"}),
                    on="key", how="left")
    chart_tr = ros.is_transfer.fillna(False).astype(bool)
    ros["is_transfer"] = (ros.prev_team.notna() & (ros.prev_team != ros.team))
    ros["transferred_before"] = chart_tr        # kept for reference, not a feature
    print(f"\nis_transfer recomputed as 'changed team since 2025', matching training:")
    print(f"  new team this year : {int(ros.is_transfer.sum())} "
          f"({ros.is_transfer.mean()*100:.1f}%)")
    print(f"  chart's /TR marker : {int(chart_tr.sum())} ({chart_tr.mean()*100:.1f}%)"
          f"   <- what this used to send the model")
    print(f"  ambiguous 2025 name keys refused: {amb_prev}")

    # the two-deep lists a few players at two slots; keep one row per team-player so
    # nobody is counted twice in a team total
    before = len(ros)
    ros = ros.sort_values("depth").drop_duplicates(["team", "key", "broad_group"], keep="first")
    print(f"\ndeduped {before - len(ros)} repeated team-player-position slots")

    # a PFF id should not be claimed by two different teams
    clash = ros[ros.player_id.notna()].groupby("player_id").team.nunique()
    clash = clash[clash > 1]
    if len(clash):
        print(f"  dropping {len(clash)} ids claimed by multiple teams")
        ros.loc[ros.player_id.isin(clash.index), ["player_id", "match_type"]] = [np.nan, "unmatched"]

    print(f"\nmatched to PFF history: {ros.player_id.notna().sum()} "
          f"({ros.player_id.notna().mean()*100:.1f}%)")
    print(ros.match_type.value_counts().to_string())

    print("\ncoverage by position group (%):")
    cov = ros.groupby("broad_group").apply(
        lambda g: pd.Series({"slots": len(g),
                             "matched": g.player_id.notna().sum(),
                             "pct": round(g.player_id.notna().mean() * 100, 1)}),
        include_groups=False)
    print(cov.sort_values("pct").to_string())

    # ---- attach the player's career WAR record ---------------------------
    hist = war[war.player_id.isin(ros.player_id.dropna())]
    wide = hist.pivot_table(index="player_id", columns="season",
                            values="war", aggfunc="sum")
    wide.columns = [f"war_{c}" for c in wide.columns]
    snaps = hist.pivot_table(index="player_id", columns="season",
                             values="snaps", aggfunc="sum")
    snaps.columns = [f"snaps_{c}" for c in snaps.columns]
    ros = ros.merge(wide.join(snaps).reset_index(), on="player_id", how="left")

    for c in [c for c in ros.columns if c.startswith(("war_", "snaps_"))]:
        ros[c] = ros[c].fillna(0.0)
    ros["prior_seasons"] = sum((ros[f"snaps_{y}"] > 0).astype(int) for y in range(2021, 2026))
    ros["snaps_2025"] = ros.get("snaps_2025", 0.0)
    ros["has_history"] = ros.prior_seasons > 0

    print(f"\nwith any prior playing time: {int(ros.has_history.sum())} "
          f"({ros.has_history.mean()*100:.1f}%)")
    print(f"  needing full imputation:   {int((~ros.has_history).sum())}")

    ros = ros.drop(columns=["pid_team", "pid_group"])
    ros.to_csv(f"{HERE}/roster_2026.csv", index=False)
    print("\nroster_2026.csv written")


if __name__ == "__main__":
    main()
