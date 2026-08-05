"""Stage 6: the 2026 roster, joined to our own WAR history.

The two-deep workbook is used ONLY as the roster source - which players are on which
team at which position and depth in 2026. Every valuation number comes from the PFF
Massey WAR build in stages 1-3; the workbook's own PFSN scores and weights are dropped.
"""
import json, os, re, unicodedata
import numpy as np
import pandas as pd

import artifacts

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
    "T": "OL", "G": "OL", "C": "OL",
    "DI": "DT", "ED": "EDGE", "LB": "LB", "CB": "CB", "S": "SAF",
}

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


def load_two_deep():
    """The 2026 two-deep, preferring the scraped Ourlads charts over the workbook.

    The workbook was a 27-June export and had already drifted - LSU's starting back
    among others. scrape_ourlads.py pulls the live charts and emits the same columns,
    so this just picks whichever source is present and newer.
    """
    scraped = f"{HERE}/ourlads_2026.csv"
    if os.path.exists(scraped):
        d = pd.read_csv(scraped)
        d = d[d.depth <= 2]          # the workbook was a two-deep; match it
        conf = {t["school"]: t.get("conference") for t in
                json.load(open(f"{HERE}/cfbd_cache/teams_2026.json"))} \
            if os.path.exists(f"{HERE}/cfbd_cache/teams_2026.json") else {}
        d["conference"] = d.team.map(conf).fillna("—")
        d["eligibility"] = d["class"].fillna("")
        print(f"roster source: scraped Ourlads charts "
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

    by_key_group, amb_group = unique_by(war, ["key", "group"])
    by_key_group = by_key_group[["key", "group", "player_id"]]
    ros = ros.merge(by_key_group.rename(columns={"group": "broad_group",
                                                 "player_id": "pid_group"}),
                    on=["key", "broad_group"], how="left")
    print(f"  ambiguous history keys refused: {amb_team} on name+team, "
          f"{amb_group} on name+position")

    # No name-only fallback: it matched a South Carolina EDGE and a Delaware OL to the
    # same id. Every match must agree on either team or position group.
    ros["player_id"] = ros.pid_team.fillna(ros.pid_group)
    ros["match_type"] = np.select(
        [ros.pid_team.notna(), ros.pid_group.notna()],
        ["name+team", "name+position"], default="unmatched")

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
