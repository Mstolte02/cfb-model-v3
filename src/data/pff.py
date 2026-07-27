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

PFF_DIR = "/Users/markstolte/Downloads/pff_exports"
TWODEEP_2026 = ("/Users/markstolte/Downloads/"
                "fbs_2026_two_deep_pfsn_full_position_weights.xlsx")
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


def load_player_grades() -> pd.DataFrame:
    """[pname, season, team, group, games, grade] - one row per player-season."""
    off_cats = ["passing", "rushing", "receiving", "blocking"]
    rows = []
    for yr in YEARS:
        for cat in off_cats:
            d = pd.read_csv(f"{PFF_DIR}/{cat}_{yr}.csv")
            if "grades_offense" not in d:
                continue
            sub = d[["player", "position", "team_name", "player_game_count",
                     "grades_offense"]].dropna(subset=["grades_offense"])
            for _, r in sub.iterrows():
                rows.append((r.player, yr, r.team_name, r.position,
                             r.player_game_count, r.grades_offense, "off"))
        d = pd.read_csv(f"{PFF_DIR}/defense_{yr}.csv")
        sub = d[["player", "position", "team_name", "player_game_count",
                 "grades_defense"]].dropna(subset=["grades_defense"])
        for _, r in sub.iterrows():
            rows.append((r.player, yr, r.team_name, r.position,
                         r.player_game_count, r.grades_defense, "def"))
    df = pd.DataFrame(rows, columns=["player", "season", "team_raw", "pos",
                                     "games", "grade", "side"])
    df["pname"] = df["player"].map(_norm)
    df["team"] = df["team_raw"].map(TEAM_MAP)
    df["group"] = df["pos"].map(POS_MAP)
    df = df.dropna(subset=["team", "group"])
    # one grade per player-season-side: the row with most games
    df = df.sort_values("games", ascending=False).drop_duplicates(["pname", "season", "side"])
    return df[["pname", "season", "team", "group", "games", "grade"]]


def _depth_weight(n):
    w = [1.0, 0.45]                          # starter, backup (top-2)
    return w[:n] + [0.0] * max(0, n - 2)


def build_group_scores() -> pd.DataFrame:
    """[season, team, group, group_score]: depth-weighted prior-year grade of the
    season-N roster, per position group. Shared by talent + weight optimization."""
    g = load_player_grades()
    prior = g.groupby(["pname", "season"], as_index=False)["grade"].max()
    rows = []
    seasons = set(g["season"])
    for N in sorted(seasons):
        if (N - 1) not in seasons:
            continue
        p = prior[prior["season"] == N - 1][["pname", "grade"]].rename(
            columns={"grade": "prior_grade"})
        roster = g[g["season"] == N].merge(p, on="pname", how="left")
        for (team, grp), gg in roster.groupby(["team", "group"]):
            gg = gg.sort_values("games", ascending=False)
            gg = gg.assign(dw=_depth_weight(len(gg)))
            has = gg.dropna(subset=["prior_grade"])
            if has["dw"].sum() == 0:
                continue
            gs = (has["dw"] * has["prior_grade"]).sum() / has["dw"].sum()
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
    grades25 = g[g["season"] == 2025].groupby("pname", as_index=False)["grade"].max()
    td = pd.read_excel(TWODEEP_2026, sheet_name="Weighted Two Deep")
    td = td[["team", "broad_group", "player_display", "depth"]].dropna(
        subset=["player_display", "broad_group"])
    td["pname"] = td["player_display"].map(_norm)
    td = td.merge(grades25.rename(columns={"grade": "prior_grade"}), on="pname", how="left")
    if qb_grades:
        qb = td["broad_group"] == "QB"
        td.loc[qb, "prior_grade"] = td.loc[qb, "pname"].map(qb_grades).fillna(
            td.loc[qb, "prior_grade"])
    td["dw"] = np.where(pd.to_numeric(td["depth"], errors="coerce") <= 1, 1.0, 0.45)

    scores = {}
    for team, tg in td.groupby("team"):
        num = den = 0.0
        for grp, gg in tg.groupby("broad_group"):
            has = gg.dropna(subset=["prior_grade"])
            if has["dw"].sum() == 0:
                continue
            gs = (has["dw"] * has["prior_grade"]).sum() / has["dw"].sum()
            pw = weights.get(grp, 0)
            num += pw * gs; den += pw
        if den > 0:
            scores[team] = num / den
    s = pd.Series(scores)
    return {2026: (s - s.mean()) / (s.std(ddof=0) or 1.0)}
