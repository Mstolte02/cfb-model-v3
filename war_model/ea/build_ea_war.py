"""A WAR build on EA College Football ratings instead of PFF grades, for 2026.

WHAT THIS SHARES WITH THE PFF BUILD, AND WHY THAT IS THE POINT. two_level_weights.py
splits the model into two questions: what is a concept worth (fitted against wins over
eleven seasons and 1,302 team-seasons) and which measurements represent that concept
(no outcome data involved). Only the second one is PFF-specific. So this swaps the
measurement layer to EA attributes and INHERITS the concept weights - which matters,
because the EA era has no fittable sample. EA serves only the current game, CFB 26 is
behind a robots.txt that disallows this crawler, and CFB 25 is archived nowhere, so
there is exactly one game-year available and no played season to fit against.

That also means THIS BUILD IS UNVALIDATED. CFB 27 rates the 2026 rosters, and 2026 has
not been played. Running it retrospectively against 2025 does not work either: CFB 27
only rates players who will be on a 2026 roster, so 2025's seniors are absent and the
400+ snap players - the ones who decide seasons - match at 53%, worse than the scrubs.
What comes out of here is comparable to the PFF build, not scoreable against it.

THREE THINGS EA DOES NOT GIVE, and what is done about each:

  playing time   EA rates everyone equally, which is its great advantage over PFF
                 (1,178 of 5,770 two-deep slots currently get an imputed value) and
                 also means a fourth-string guard would carry a starter's weight. The
                 PFF build gets volume from snap counts. Here depth is taken from EA's
                 own opinion - overall rank within team and position group - and turned
                 into snaps by the median a player of that rank actually played, from
                 our 2025 data. So EA decides who plays; history decides how much.

  discipline     6.0% of the fitted weight, and EA does not rate penalties at all.
                 Rather than drop it and let the weights sum to 0.94, its share is
                 redistributed across the concepts EA can measure, in proportion.

  a season       the z-scores need a population. Everything is standardized within
                 position group across all of FBS, which is what the PFF build does
                 within season.

Run: ../../venv/bin/python build_ea_war.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))

import two_level_weights as tlw          # noqa: E402
from build_massey import massey_matrix   # noqa: E402
from build_war import REPL_WIN_PCT       # noqa: E402
from build_roster_2026 import norm_name  # noqa: E402

WAR_DIR = os.path.dirname(HERE)

# EA's position labels -> the broad groups the rest of the pipeline uses.
POS_GROUP = {
    "QB": "QB", "HB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
    "LT": "OL", "LG": "OL", "C": "OL", "RG": "OL", "RT": "OL",
    "LEDG": "EDGE", "REDG": "EDGE", "DT": "DT",
    "MIKE": "LB", "SAM": "LB", "WILL": "LB",
    "CB": "CB", "FS": "SAF", "SS": "SAF",
    # kickers and punters play none of the concepts the model weights
    "K": None, "P": None,
}

# EA's school names against the CFBD set the rest of the model is indexed on. Eleven
# differ, and each one that is not fixed silently drops a team out of the Massey solve
# and out of every comparison downstream.
TEAM_ALIAS = {
    "Appalachian State": "App State", "Cal": "California", "Connecticut": "UConn",
    "FAU": "Florida Atlantic", "FIU": "Florida International", "Hawaii": "Hawai'i",
    "Miami (Ohio)": "Miami (OH)", "Middle Tennessee State": "Middle Tennessee",
    "San Jose State": "San José State", "UMass": "Massachusetts",
    "USF": "South Florida",
}

# The EA attributes that measure each football CONCEPT. Concept is the right level
# for the measurement question - what does coverage look like in EA's vocabulary - and
# the wrong level for the WEIGHTING question, which is why the facets below exist.
CONCEPT_ATTRS = {
    "passing":         ["throwPower", "throwAccuracyShort", "throwAccuracyMid",
                        "throwAccuracyDeep", "throwOnTheRun", "playAction", "awareness"],
    "qb_pressure":     ["throwUnderPressure", "breakSack"],
    "qb_rushing":      ["speed", "acceleration", "breakTackle", "bCVision"],
    "rushing":         ["carrying", "breakTackle", "trucking", "spinMove", "jukeMove",
                        "stiffArm", "bCVision", "speed"],
    "receiving":       ["catching", "shortRouteRunning", "mediumRouteRunning",
                        "deepRouteRunning"],
    "yac":             ["breakTackle", "jukeMove", "spinMove", "stiffArm", "speed"],
    "hands":           ["catching", "catchInTraffic", "spectacularCatch"],
    "pass_protection": ["passBlock", "passBlockPower", "passBlockFinesse"],
    "run_blocking":    ["runBlock", "runBlockPower", "runBlockFinesse",
                        "impactBlocking", "leadBlock"],
    "pass_rush":       ["powerMoves", "finesseMoves", "blockShedding", "strength"],
    "run_defense":     ["blockShedding", "tackle", "playRecognition", "pursuit",
                        "strength"],
    "coverage":        ["manCoverage", "zoneCoverage", "press", "playRecognition"],
    "tackling":        ["tackle", "hitPower", "pursuit"],
    "ball_security":   ["carrying"],
}

# EVERY PFF FACET IS SCOPED TO POSITIONS, AND THAT SCOPE CARRIES THE FIT'S OPINION
# ABOUT POSITIONAL VALUE. The first version of this build collapsed to concepts, so
# run_defense - one 17.4% weight - was shared across DT, EDGE, LB, CB and SAF in
# proportion to snap volume rather than by what the fit actually learned. The result
# drifted badly against the PFF build: linebackers +4.1 points of league WAR share,
# safeties +4.0, interior line -5.0, tight ends -4.2.
#
# The fit does not have one run_defense weight. It has DI_run at 3.5%, ED_run at 2.9%,
# LB_run at 2.8% and CB-S_run separately, and those differences are the answer to
# "which position's run defence matters most". So each PFF facet is rebuilt here with
# its own scope and its own fitted weight, and only the MEASUREMENT comes from EA.
FACET_GROUPS = {
    "QB": ["QB"], "RB": ["RB"], "WR": ["WR"], "TE": ["TE"], "OL": ["OL"],
    "DI": ["DT"], "ED": ["EDGE"], "LB": ["LB"], "CB": ["CB"], "S": ["SAF"],
    "CB-S": ["CB", "SAF"], "DI-ED": ["DT", "EDGE"], "CB-LB-S": ["CB", "LB", "SAF"],
    "QB-RB-TE-WR": ["QB", "RB", "TE", "WR"],
    "cfbd_pass_qb": ["QB"], "cfbd_run_qb": ["QB"], "cfbd_run_rb": ["RB"],
    "cfbd_recv_wr": ["WR"], "cfbd_recv_te": ["TE"], "cfbd_recv_rb": ["RB"],
    "cfbd_havoc_dl": ["DT"], "cfbd_havoc_lb": ["LB"],
    "cfbd_havoc_db": ["CB", "SAF"], "cfbd_cov_db": ["CB", "SAF"],
    "cfbd_tackle_lb": ["LB"], "cfbd_tackle_db": ["CB", "SAF"],
}


def facet_scope(name):
    """(groups, prefix) for a PFF facet name. cfbd_ facets are their own prefix."""
    if name.startswith("cfbd_"):
        return FACET_GROUPS.get(name), name
    pre = name.split("_")[0]
    return FACET_GROUPS.get(pre), pre


def inherited_weights():
    """Concept weights from the PFF fit, with discipline's share redistributed.

    EA rates no equivalent of a penalty, so that concept cannot be carried. Dropping it
    outright would leave the weights summing to 0.94 and quietly shrink every player's
    WAR by 6%; spreading it keeps the total honest and the relative ordering intact.
    """
    fw = pd.read_csv(f"{WAR_DIR}/hybrid_facet_weights.csv", index_col=0)["rf"]
    cmap = pd.Series(tlw.concept_map(list(fw.index))).reindex(fw.index)
    w = fw.groupby(cmap).sum()
    keep = w.reindex(CONCEPT_ATTRS.keys()).fillna(0.0)
    dropped = float(w.sum() - keep.sum())
    out = keep / keep.sum()
    return out, dropped


def facet_weights():
    """Per-FACET fitted weights, scoped to positions, with the facets EA cannot
    measure dropped and their share redistributed over the rest."""
    fw = pd.read_csv(f"{WAR_DIR}/hybrid_facet_weights.csv", index_col=0)["rf"]
    cmap = tlw.concept_map(list(fw.index))
    rows = []
    for name, wt in fw.items():
        concept = cmap.get(name)
        attrs = CONCEPT_ATTRS.get(concept)
        groups, pre = facet_scope(name)
        if not attrs or not groups or wt <= 0:
            continue                       # discipline, or a facet at exactly zero
        rows.append({"facet": name, "concept": concept, "prefix": pre,
                     "groups": groups, "attrs": attrs, "w": float(wt)})
    d = pd.DataFrame(rows)
    dropped = float(fw.sum() - d.w.sum())
    d["w"] = d.w / d.w.sum()
    return d, dropped


def volume_table():
    """Median 2025 snaps by (group, depth rank) - how much a player of that standing
    actually plays. Taken from our own history so EA is never asked a question about
    playing time, which it has no basis to answer."""
    w = pd.read_csv(f"{WAR_DIR}/hybrid_player_war.csv")
    w = w[(w.season == 2025) & (w.snaps > 0)].copy()
    from build_roster_2026 import PFF_TO_GROUP
    w["group"] = w.position.map(PFF_TO_GROUP)
    w = w[w.group.notna()]
    w["rank"] = w.groupby(["team", "group"]).snaps.rank(ascending=False, method="first")
    w["rank"] = w["rank"].clip(upper=6)
    return w.groupby(["group", "rank"]).snaps.median()


def load_ea():
    ea = pd.read_csv(f"{HERE}/ea_cfb27.csv")
    ea["team"] = ea.team.replace(TEAM_ALIAS)
    ea["group"] = ea.position.map(POS_GROUP)
    ea = ea[ea.group.notna()].copy()
    ea["key"] = ea.player.map(norm_name)
    # EA has no depth chart, so its own overall rating orders the room
    ea["rank"] = ea.groupby(["team", "group"]).overall.rank(ascending=False,
                                                            method="first").clip(upper=6)
    vol = volume_table()
    ea["snaps"] = [vol.get((g, r), np.nan) for g, r in zip(ea.group, ea["rank"])]
    ea["snaps"] = ea.snaps.fillna(ea.group.map(vol.groupby(level=0).min())).fillna(50.0)
    return ea


def facet_values(ea, fw):
    """Per player per FACET: a snap-weighted z of the averaged attributes, times
    volume - the value = z * snaps shape the PFF facets use.

    Scoped exactly as the PFF facet is. DI_run standardizes interior linemen against
    interior linemen; LB_run standardizes linebackers against linebackers; they are
    separate columns carrying separate fitted weights, which is the whole point.
    """
    rows = []
    for r in fw.itertuples():
        have = [a for a in r.attrs if a in ea.columns]
        d = ea[ea.group.isin(r.groups)]
        if not have or d.empty:
            continue
        d = d.copy()
        # standardize each attribute within position group before averaging: an 85
        # strength means something different at guard than at corner
        parts = []
        for a in have:
            g = d.groupby("group")[a]
            parts.append((d[a] - g.transform("mean")) / g.transform("std").replace(0, 1))
        d["score"] = np.mean(parts, axis=0)
        w_ = d.snaps.to_numpy(float)
        mu = np.average(d.score, weights=w_)
        sd = np.sqrt(np.average((d.score - mu) ** 2, weights=w_)) or 1.0
        d["z"] = (d.score - mu) / sd
        d["facet"] = r.facet
        d["concept"] = r.concept
        d["value"] = d.z * d.snaps
        rows.append(d[["id", "player", "team", "position", "group", "key", "overall",
                       "rank", "snaps", "facet", "concept", "z", "value"]])
    return pd.concat(rows, ignore_index=True)


def main():
    fw, dropped = facet_weights()
    w = fw.set_index("facet").w
    byc = fw.groupby("concept").w.sum().sort_values(ascending=False)
    print(f"inherited FACET weights: {len(fw)} facets, "
          f"{dropped*100:.1f}% (unmeasurable by EA) redistributed")
    print("  rolled up by concept:")
    for c, v in byc.items():
        print(f"    {c:<18}{v*100:5.2f}%")
    print("  rolled up by position scope:")
    for pre, v in fw.groupby("prefix").w.sum().sort_values(ascending=False).items():
        print(f"    {pre:<18}{v*100:5.2f}%")

    ea = load_ea()
    print(f"\nEA players used: {len(ea)} over {ea.team.nunique()} teams")

    cv = facet_values(ea, fw)
    print(f"player-facet rows: {len(cv)}")

    tot = (cv.groupby(["team", "facet"], as_index=False).value.sum()
             .pivot(index="team", columns="facet", values="value")
             .reindex(columns=w.index).fillna(0.0))
    sigma = tot.std(ddof=0)
    Z = (tot - tot.mean()) / sigma.replace(0, 1)
    f = pd.Series(Z.to_numpy() @ w.to_numpy(), index=Z.index)

    # ---- schedule -> Massey, the same solve the PFF build uses -----------------
    sched_raw = json.load(open(f"{ROOT}/data/raw/schedule_2026.json"))
    sched = pd.DataFrame([{"home_team": g["homeTeam"], "away_team": g["awayTeam"],
                           "season": 2026} for g in sched_raw])
    teams = sorted(t for t in f.index if
                   ((sched.home_team == t) | (sched.away_team == t)).any())
    missing = sorted(set(f.index) - set(teams))
    if len(missing) > 4:
        sys.exit(f"{len(missing)} EA teams are not in the 2026 schedule - almost "
                 f"certainly a naming mismatch, not a real absence: {missing[:12]}")
    if missing:
        print(f"  not on the 2026 FBS schedule, dropped: {missing}")
    fv = f.reindex(teams)
    fv = fv - fv.mean()
    M, _ = massey_matrix(sched, 2026, teams)
    A = M.copy(); A[-1, :] = 1.0
    b = fv.to_numpy(float).copy(); b[-1] = 0.0
    massey = pd.Series(np.linalg.solve(A, b), index=teams)

    slope = json.load(open(f"{WAR_DIR}/hybrid_wins_map.json"))["slope"]
    games = pd.Series({t: int(((sched.home_team == t) | (sched.away_team == t)).sum())
                       for t in teams})
    # c_t: how much a team's own f moves its rating, holding the rest fixed
    Minv = np.linalg.inv(A)
    c_t = pd.Series({t: Minv[i, i] for i, t in enumerate(teams)})

    cv = cv[cv.team.isin(teams)].copy()
    cv["w"] = cv.facet.map(w)
    cv["sigma"] = cv.facet.map(sigma)
    cv["c_t"] = cv.team.map(c_t)
    cv["games"] = cv.team.map(games)
    cv["waa"] = cv.games * slope * cv.c_t * cv.w * cv.value / cv.sigma

    pool = float(((0.5 - REPL_WIN_PCT) * games).sum())
    league = cv.groupby("facet").snaps.sum()
    per_snap = {c: pool * w[c] / league[c] for c in league.index if league[c] > 0}
    cv["repl"] = cv.facet.map(per_snap).fillna(0.0) * cv.snaps
    cv["war"] = cv.waa + cv.repl

    key = ["id", "player", "team", "position", "group", "overall", "key"]
    player = cv.groupby(key, as_index=False).agg(waa=("waa", "sum"), war=("war", "sum"),
                                                 snaps=("snaps", "max"))
    player.to_csv(f"{HERE}/ea_player_war_2026.csv", index=False)
    team = player.groupby("team", as_index=False).war.sum().sort_values("war",
                                                                        ascending=False)
    team["massey"] = team.team.map(massey)
    team["f"] = team.team.map(f)
    team.to_csv(f"{HERE}/ea_team_war_2026.csv", index=False)

    print(f"\nteams rated: {len(team)}   total EA WAR: {player.war.sum():.0f}")
    print("\ntop 15 teams by EA WAR:")
    print(team.head(15).round(3).to_string(index=False))
    print("\ntop 12 players by EA WAR:")
    print(player.nlargest(12, "war")[["player", "position", "team", "overall", "war"]]
          .round(3).to_string(index=False))
    print(f"\n-> ea_player_war_2026.csv, ea_team_war_2026.csv")


if __name__ == "__main__":
    main()
