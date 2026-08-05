"""PFF player grades (2021-2025) -> roster-aware, transfer-aware team talent.

For season N, a team's roster = players with PFF grades at that team in N. Each
player carries their PRIOR-year (N-1) grade (leakage-safe: last year's performance,
known before season N). Group by position, depth-weight by snaps (games proxy),
aggregate with the win%-optimized position weights. Transfers naturally appear at
their new team carrying their old grade; departed players simply aren't on the
roster. This is the signal the historical team-level PFSN scores could NOT capture.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

from config import PFF_DIR, TWODEEP_2026, require  # single definition; see config.py

require(PFF_DIR, "the PFF exports", "PFF_DIR")
# Exports now reach back to 2014. 2020 is skipped for the same reason the WAR
# build skips it: conference-only COVID schedules make that season
# incomparable to the others.
YEARS = [y for y in range(2014, 2026) if y != 2020]

# PFSN-derived weights (reference).
POS_WEIGHTS = {"QB": .279, "CB": .223, "EDGE": .148, "DT": .116, "SAF": .089,
               "LB": .055, "WR": .034, "OL": .032, "TE": .014, "RB": .010}
# Re-optimized on PFF grades vs win% (NNLS, scripts/compare_signals.py) -- much
# more balanced than PFSN's QB/CB-heavy weights. Used with PFF grades.
PFF_OPT_WEIGHTS = {"QB": .19, "DT": .12, "LB": .12, "OL": .12, "WR": .10,
                   "EDGE": .08, "SAF": .08, "CB": .07, "TE": .07, "RB": .03}

POS_MAP = {"QB": "QB", "HB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
           "T": "OL", "G": "OL", "C": "OL", "DI": "DT", "ED": "EDGE",
           "LB": "LB", "CB": "CB", "S": "SAF"}            # K/P/LS dropped

TEAM_MAP = {
    "AIR FORCE": "Air Force", "AKRON": "Akron", "ALABAMA": "Alabama",
    "APP STATE": "App State", "ARIZONA": "Arizona", "ARIZONA ST": "Arizona State",
    "ARK STATE": "Arkansas State", "ARKANSAS": "Arkansas", "ARMY": "Army",
    "AUBURN": "Auburn", "BALL ST": "Ball State", "BAYLOR": "Baylor",
    "BOISE ST": "Boise State", "BOSTON COL": "Boston College",
    "BOWL GREEN": "Bowling Green", "BUFFALO": "Buffalo", "BYU": "BYU",
    "C MICHIGAN": "Central Michigan", "CAL": "California", "CHARLOTTE": "Charlotte",
    "CINCINNATI": "Cincinnati", "CLEMSON": "Clemson", "COAST CAR": "Coastal Carolina",
    "COLO STATE": "Colorado State", "COLORADO": "Colorado", "DELAWARE": "Delaware",
    "DOMINION": "Old Dominion", "DUKE": "Duke", "E CAROLINA": "East Carolina",
    "E MICHIGAN": "Eastern Michigan", "FAU": "Florida Atlantic",
    "FIU": "Florida International", "FLORIDA": "Florida", "FLORIDA ST": "Florida State",
    "FRESNO ST": "Fresno State", "GA SOUTHRN": "Georgia Southern",
    "GA STATE": "Georgia State", "GA TECH": "Georgia Tech", "GEORGIA": "Georgia",
    "HAWAII": "Hawai'i", "HOUSTON": "Houston", "ILLINOIS": "Illinois",
    "INDIANA": "Indiana", "IOWA": "Iowa", "IOWA STATE": "Iowa State",
    "JAMES MAD": "James Madison", "JVILLE ST": "Jacksonville State", "KANSAS": "Kansas",
    "KANSAS ST": "Kansas State", "KENNESAW": "Kennesaw State", "KENT STATE": "Kent State",
    "KENTUCKY": "Kentucky", "LA LAFAYET": "Louisiana", "LA MONROE": "UL Monroe",
    "LA TECH": "Louisiana Tech", "LIBERTY": "Liberty", "LOUISVILLE": "Louisville",
    "LSU": "LSU", "MARSHALL": "Marshall", "MARYLAND": "Maryland", "MEMPHIS": "Memphis",
    "MIAMI FL": "Miami", "MIAMI OH": "Miami (OH)", "MICH STATE": "Michigan State",
    "MICHIGAN": "Michigan", "MIDDLE TN": "Middle Tennessee", "MINNESOTA": "Minnesota",
    "MISS STATE": "Mississippi State", "MISSOURI": "Missouri", "MO STATE": "Missouri State",
    "N CAROLINA": "North Carolina", "N ILLINOIS": "Northern Illinois",
    "N TEXAS": "North Texas", "NAVY": "Navy", "NC STATE": "NC State",
    "NEBRASKA": "Nebraska", "NEVADA": "Nevada", "NEW MEX ST": "New Mexico State",
    "NEW MEXICO": "New Mexico", "NOTRE DAME": "Notre Dame", "NWESTERN": "Northwestern",
    "OHIO": "Ohio", "OHIO STATE": "Ohio State", "OKLA STATE": "Oklahoma State",
    "OKLAHOMA": "Oklahoma", "OLE MISS": "Ole Miss", "OREGON": "Oregon",
    "OREGON ST": "Oregon State", "PENN STATE": "Penn State", "PITTSBURGH": "Pittsburgh",
    "PURDUE": "Purdue", "RICE": "Rice", "RUTGERS": "Rutgers", "S ALABAMA": "South Alabama",
    "S CAROLINA": "South Carolina", "S DIEGO ST": "San Diego State",
    "S JOSE ST": "San José State", "SM HOUSTON": "Sam Houston", "SMU": "SMU",
    "SO MISS": "Southern Miss", "STANFORD": "Stanford", "SYRACUSE": "Syracuse",
    "TCU": "TCU", "TEMPLE": "Temple", "TENNESSEE": "Tennessee", "TEXAS": "Texas",
    "TEXAS A&M": "Texas A&M", "TEXAS ST": "Texas State", "TEXAS TECH": "Texas Tech",
    "TOLEDO": "Toledo", "TROY": "Troy", "TULANE": "Tulane", "TULSA": "Tulsa",
    "UAB": "UAB", "UCF": "UCF", "UCLA": "UCLA", "UCONN": "UConn", "UMASS": "Massachusetts",
    "UNLV": "UNLV", "USC": "USC", "USF": "South Florida", "UTAH": "Utah",
    "UTAH ST": "Utah State", "UTEP": "UTEP", "UTSA": "UTSA", "VA TECH": "Virginia Tech",
    "VANDERBILT": "Vanderbilt", "VIRGINIA": "Virginia", "W KENTUCKY": "Western Kentucky",
    "W MICHIGAN": "Western Michigan", "W VIRGINIA": "West Virginia", "WAKE": "Wake Forest",
    "WASH STATE": "Washington State", "WASHINGTON": "Washington", "WISCONSIN": "Wisconsin",
    "WYOMING": "Wyoming",
}  # "W GEORGIA" (West Georgia) intentionally omitted - not in CFBD FBS set.


def _norm(n):
    n = re.sub(r"[.'`]", "", str(n).lower().strip())
    n = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", n)
    return re.sub(r"\s+", " ", n)


# Where each side's real snap count lives. Offensive snaps are only published in the
# blocking export, but that export covers every offensive position (it grades a
# receiver's blocking too), so it is the snap source for all of them.
SNAP_COL = {"off": ("blocking", "snap_counts_offense"),
            "def": ("defense", "snap_counts_defense")}


def _snap_counts() -> pd.DataFrame:
    """[player_id, season, team_raw, side, snaps] - the real denominator.

    The depth weighting used to be `games`, on the two players with the most of them.
    Games is a participation flag, not playing time: a backup who appears on special
    teams in all twelve games outranks a starter who missed three.
    """
    rows = []
    for yr in YEARS:
        for side, (cat, col) in SNAP_COL.items():
            d = pd.read_csv(f"{PFF_DIR}/{cat}_{yr}.csv")
            s = d[["player_id", "team_name", col]].rename(columns={col: "snaps"})
            s["snaps"] = pd.to_numeric(s.snaps, errors="coerce").fillna(0.0)
            rows.append(s.assign(season=yr, side=side))
    s = pd.concat(rows, ignore_index=True)
    return (s.groupby(["player_id", "season", "team_name", "side"], as_index=False)
             .snaps.max().rename(columns={"team_name": "team_raw"}))


def load_player_grades() -> pd.DataFrame:
    """[player_id, pname, season, team, group, games, snaps, grade], one row per
    player-TEAM-season-side.

    PLAYER_ID IS THE KEY, and it did not used to be. The old version never read the
    column: it collapsed to one row per (normalized name, season, side), which does
    three separate wrong things. 743 of those keys hold two genuinely different
    players, and one of the two was deleted. 737 span two teams, so a name shared
    across programmes had one team's grade assigned to the other's roster. And a
    player who transferred mid-season could only appear once. Keying on the id PFF
    already publishes makes all three impossible rather than rare.

    The collapse that IS wanted - a quarterback appears in both the passing and the
    rushing export with the same offensive grade - is still done, on the id.
    """
    off_cats = ["passing", "rushing", "receiving", "blocking"]
    rows = []
    for yr in YEARS:
        for cat in off_cats:
            d = pd.read_csv(f"{PFF_DIR}/{cat}_{yr}.csv")
            if "grades_offense" not in d:
                continue
            sub = d[["player_id", "player", "position", "team_name",
                     "player_game_count", "grades_offense"]].dropna(
                         subset=["grades_offense"])
            rows.append(sub.rename(columns={"grades_offense": "grade"}).assign(
                season=yr, side="off"))
        d = pd.read_csv(f"{PFF_DIR}/defense_{yr}.csv")
        sub = d[["player_id", "player", "position", "team_name",
                 "player_game_count", "grades_defense"]].dropna(
                     subset=["grades_defense"])
        rows.append(sub.rename(columns={"grades_defense": "grade"}).assign(
            season=yr, side="def"))

    df = pd.concat(rows, ignore_index=True).rename(
        columns={"team_name": "team_raw", "position": "pos",
                 "player_game_count": "games"})
    df["pname"] = df["player"].map(_norm)
    df["team"] = df["team_raw"].map(TEAM_MAP)
    df["group"] = df["pos"].map(POS_MAP)

    unmapped = df[df.team.isna()].team_raw.value_counts()
    if len(unmapped):
        print(f"  [pff] {len(unmapped)} team names outside the FBS map, "
              f"{int(unmapped.sum())} rows dropped: {list(unmapped.index[:5])}")
    df = df.dropna(subset=["team", "group"])

    # one grade per player-team-season-side, taking the export where he appeared most
    df = df.sort_values("games", ascending=False).drop_duplicates(
        ["player_id", "season", "team_raw", "side"])

    df = df.merge(_snap_counts(), on=["player_id", "season", "team_raw", "side"],
                  how="left")
    # A graded player with no published snap count is real but rare (~1% of offensive
    # rows, all in the passing/receiving exports and absent from blocking). Falling
    # back to games keeps him in the weighting at roughly the right size instead of
    # weighting him zero, which is what a fillna(0) would silently do.
    missing = df.snaps.isna() | (df.snaps <= 0)
    if missing.any():
        med = df.loc[~missing].groupby("side").snaps.median()
        per_game = df.loc[~missing].snaps.sum() / max(df.loc[~missing].games.sum(), 1)
        df.loc[missing, "snaps"] = (df.loc[missing, "games"] * per_game).clip(lower=1.0)
        print(f"  [pff] {int(missing.sum())} of {len(df)} graded rows have no snap "
              f"count; imputed at {per_game:.1f}/game (median real {med.to_dict()})")

    return df[["player_id", "pname", "season", "team", "group", "games", "snaps",
               "grade"]]


def build_group_scores() -> pd.DataFrame:
    """[season, team, group, group_score]: snap-weighted prior-year grade of the
    season-N roster, per position group. Shared by talent + weight optimization.

    TWO THINGS CHANGED HERE.

    The prior-year join is on player_id. It was on the normalized name, which
    carried last season's grade to whoever happened to share a name this season.

    And the depth weighting is every contributor's snaps, not [1.0, 0.45, 0, 0, ...]
    over the top two by games. The old vector said an offensive line is its two
    busiest players and the other three are worth nothing, which is not what a line
    is; at receiver and in the secondary - where five and six players rotate - it
    threw away most of the group. Snaps are the weighting the question already
    implies: how much of this group's playing time is covered by whom.
    """
    g = load_player_grades()
    # a player's prior season is his own, wherever he played it; max over sides is
    # what it always was (a two-way player carries his better grade)
    prior = g.groupby(["player_id", "season"], as_index=False)["grade"].max()
    prior["season"] += 1
    prior = prior.rename(columns={"grade": "prior_grade"})

    # A season whose predecessor is not in the exports has nothing to look back on -
    # 2014 because it is the first, 2021 because 2020 is excluded - and scoring it
    # would produce a roster of players who all look like true freshmen.
    seasons = set(g.season)
    usable = sorted(N for N in seasons if (N - 1) in seasons)

    roster = g[g.season.isin(usable)].merge(prior, on=["player_id", "season"],
                                            how="left")
    matched = roster.prior_grade.notna().mean()
    print(f"  [pff] prior-year grade matched for {matched*100:.1f}% of roster rows "
          f"on player_id")

    # Replacement level per season and group: the snap-weighted 10th percentile of
    # the prior grades actually on the field there. A group with nobody graded used
    # to be DROPPED, which quietly rescaled that team's talent over the remaining
    # groups and so imputed the hole at the team's own average - the opposite of the
    # truth, since a group with no returning graded player is a weak one.
    have = roster.dropna(subset=["prior_grade"])
    repl = (have.groupby(["season", "group"]).prior_grade.quantile(0.10)
                .rename("repl"))

    rows = []
    for (N, team, grp), gg in roster.groupby(["season", "team", "group"]):
        has = gg.dropna(subset=["prior_grade"])
        r = repl.get((N, grp), np.nan)
        if has.snaps.sum() > 0:
            gs = float((has.snaps * has.prior_grade).sum() / has.snaps.sum())
            # the share of the group's snaps held by players with no prior grade -
            # true freshmen, JUCOs, FCS transfers - is credited at replacement
            unknown = float(gg.snaps.sum() - has.snaps.sum())
            if unknown > 0 and np.isfinite(r):
                tot = float(gg.snaps.sum())
                gs = (gs * has.snaps.sum() + r * unknown) / tot
        elif np.isfinite(r):
            gs = float(r)
        else:
            continue
        rows.append({"season": N, "team": team, "group": grp, "group_score": gs})
    return pd.DataFrame(rows)


def build_roster_talent(weights=PFF_OPT_WEIGHTS, group_scores=None) -> dict:
    """{season N: Series(team -> standardized roster-aware talent)}.
    Pass group_scores to reuse a precomputed table (e.g. with optimized weights)."""
    gs = build_group_scores() if group_scores is None else group_scores
    out = {}
    for N, sub in gs.groupby("season"):
        scores = {}
        for team, tg in sub.groupby("team"):
            num = sum(weights.get(r.group, 0) * r.group_score for r in tg.itertuples())
            den = sum(weights.get(r.group, 0) for r in tg.itertuples())
            if den > 0:
                scores[team] = num / den
        s = pd.Series(scores)
        out[N] = (s - s.mean()) / (s.std(ddof=0) or 1.0)
    return out


# How far down a group the snap-share profile is measured. Beyond this a player is
# taking a rounding error's worth of the group's playing time.
_MAX_RANK = 8


def depth_share_profile(g: pd.DataFrame) -> dict:
    """{(group, rank): mean share of the group's snaps held by its rank-th player}.

    Measured on the historical grades table, so the two-deep - which has no snaps -
    can be weighted by what playing time actually does at each depth slot rather than
    by an assumed 1.0 / 0.45.
    """
    d = g.copy()
    d["rank"] = d.groupby(["season", "team", "group"]).snaps.rank(
        ascending=False, method="first")
    d = d[d["rank"] <= _MAX_RANK]
    d["share"] = d.snaps / d.groupby(["season", "team", "group"]).snaps.transform("sum")
    prof = d.groupby(["group", "rank"]).share.mean()
    # renormalize within group so the listed slots carry the whole group
    prof = prof / prof.groupby("group").transform("sum")
    return {(grp, int(r)): float(v) for (grp, r), v in prof.items()}


def _join_prior_grade(td: pd.DataFrame, g25: pd.DataFrame) -> pd.DataFrame:
    """Attach each two-deep player's 2025 grade. Name-keyed, because the workbook
    carries no player id - so the join is TIERED AND LOGGED rather than assumed.

    It used to be one `merge(on='pname')` against a name-deduplicated grade table,
    which resolved every ambiguity by silently keeping whichever row sorted first.
    A name that belongs to two players is not a match; it is an unresolved case, and
    it is counted here instead of being turned into a number.
    """
    td = td.copy()
    td["prior_grade"] = np.nan
    tiers = [(["pname", "team", "group"], "name+team+group"),
             (["pname", "team"], "name+team"),
             (["pname"], "name only")]
    td["group"] = td.broad_group
    log = []
    for keys, label in tiers:
        need = td.prior_grade.isna()
        if not need.any():
            break
        # only unambiguous keys are allowed to resolve: two graded players sharing
        # the key means we do not know which one is listed
        cand = g25.groupby(keys, as_index=False).agg(
            grade=("grade", "max"), n=("player_id", "nunique"))
        uniq = cand[cand.n == 1].drop(columns="n")
        m = td.loc[need, keys].merge(uniq, on=keys, how="left")
        m.index = td.index[need]
        td.loc[need, "prior_grade"] = m.grade
        got = int(m.grade.notna().sum())
        ambiguous = int(cand[cand.n > 1].n.sum())
        log.append(f"{label} {got}" + (f" ({ambiguous} ambiguous)" if ambiguous else ""))

    unresolved = td[td.prior_grade.isna()]
    print(f"  [pff] 2026 two-deep grade join: " + ", ".join(log) +
          f"; {len(unresolved)} of {len(td)} unresolved "
          f"({len(unresolved)/len(td)*100:.1f}%, credited at replacement)")
    return td.drop(columns="group")


def build_2026_roster_talent(weights=PFF_OPT_WEIGHTS, qb_grades=None) -> dict:
    """{2026: Series(team -> standardized talent)} from the 2026 Ourlads two-deep
    (roster) x each player's 2025 PFF grade. Same construction as the historical
    roster talent, but the roster comes from the preseason two-deep (since 2026
    hasn't been played). Players without a 2025 grade are skipped (freshmen).

    qb_grades={normalized_name: grade}: opponent-adjusted QB WAR (rescaled to the
    PFF grade scale) replacing the PFF grade for matched QBs — 'use WAR instead of
    PFF at the one position CFBD can measure cleanly'. Non-QB positions keep PFF.
    """
    g = load_player_grades()
    g25 = g[g.season == 2025]

    td = pd.read_excel(require(TWODEEP_2026, "the 2026 two-deep workbook",
                               "CFB_TWODEEP_2026"), sheet_name="Weighted Two Deep")
    td = td[["team", "broad_group", "player_display", "depth"]].dropna(
        subset=["player_display", "broad_group"])
    td["pname"] = td["player_display"].map(_norm)
    td["depth"] = pd.to_numeric(td.depth, errors="coerce").fillna(2)

    td = _join_prior_grade(td, g25)
    if qb_grades:
        qb = td["broad_group"] == "QB"
        td.loc[qb, "prior_grade"] = td.loc[qb, "pname"].map(qb_grades).fillna(
            td.loc[qb, "prior_grade"])

    # The two-deep has no snaps - 2026 has not been played - so the weighting has to
    # come from how playing time HAS divided at each depth slot, measured on the same
    # grades table the historical side uses. That keeps the two sides the same object.
    # It used to be a flat 1.0 / 0.45, which is a guess, and a guess that says an
    # offensive line's five starters and five backups split 69/31 regardless of group.
    share = depth_share_profile(g)
    td["rank"] = td.groupby(["team", "broad_group"]).depth.rank(method="first")
    td["dw"] = [share.get((grp, min(int(r), _MAX_RANK)), 0.0)
                for grp, r in zip(td.broad_group, td["rank"])]

    repl25 = (g25.groupby("group").grade.quantile(0.10))

    scores = {}
    for team, tg in td.groupby("team"):
        num = den = 0.0
        for grp, gg in tg.groupby("broad_group"):
            if gg.dw.sum() <= 0:
                continue
            has = gg.dropna(subset=["prior_grade"])
            r = repl25.get(grp, np.nan)
            if has.dw.sum() > 0:
                gs = (has.dw * has.prior_grade).sum() / has.dw.sum()
                # a listed player with no 2025 grade is a true freshman or an
                # incoming FCS/JUCO transfer: replacement, not absent (see
                # build_group_scores for why absent is the wrong default)
                unknown = float(gg.dw.sum() - has.dw.sum())
                if unknown > 0 and np.isfinite(r):
                    gs = (gs * has.dw.sum() + r * unknown) / gg.dw.sum()
            elif np.isfinite(r):
                gs = float(r)
            else:
                continue
            pw = weights.get(grp, 0)
            num += pw * gs; den += pw
        if den > 0:
            scores[team] = num / den
    s = pd.Series(scores)
    return {2026: (s - s.mean()) / (s.std(ddof=0) or 1.0)}
