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

# concept -> (EA attributes that measure it, groups that play it).
# The attributes are averaged after standardizing, so a concept measured by eight
# attributes is not thereby worth more than one measured by two - that is what the
# inherited concept weight decides.
CONCEPTS = {
    "passing":         (["throwPower", "throwAccuracyShort", "throwAccuracyMid",
                         "throwAccuracyDeep", "throwOnTheRun", "playAction",
                         "awareness"], ["QB"]),
    "qb_pressure":     (["throwUnderPressure", "breakSack"], ["QB"]),
    "qb_rushing":      (["speed", "acceleration", "breakTackle", "bCVision"], ["QB"]),
    "rushing":         (["carrying", "breakTackle", "trucking", "spinMove", "jukeMove",
                         "stiffArm", "bCVision", "speed"], ["RB"]),
    "receiving":       (["catching", "shortRouteRunning", "mediumRouteRunning",
                         "deepRouteRunning"], ["WR", "TE", "RB"]),
    "yac":             (["breakTackle", "jukeMove", "spinMove", "stiffArm", "speed"],
                        ["WR", "TE", "RB"]),
    "hands":           (["catching", "catchInTraffic", "spectacularCatch"],
                        ["WR", "TE"]),
    "pass_protection": (["passBlock", "passBlockPower", "passBlockFinesse"],
                        ["OL", "TE"]),
    "run_blocking":    (["runBlock", "runBlockPower", "runBlockFinesse",
                         "impactBlocking", "leadBlock"], ["OL", "TE", "RB"]),
    "pass_rush":       (["powerMoves", "finesseMoves", "blockShedding", "strength"],
                        ["DT", "EDGE", "LB"]),
    "run_defense":     (["blockShedding", "tackle", "playRecognition", "pursuit",
                         "strength"], ["DT", "EDGE", "LB", "CB", "SAF"]),
    "coverage":        (["manCoverage", "zoneCoverage", "press", "playRecognition"],
                        ["CB", "SAF", "LB"]),
    "tackling":        (["tackle", "hitPower", "pursuit"],
                        ["DT", "EDGE", "LB", "CB", "SAF"]),
    "ball_security":   (["carrying"], ["QB", "RB"]),
}


def inherited_weights():
    """Concept weights from the PFF fit, with discipline's share redistributed.

    EA rates no equivalent of a penalty, so that concept cannot be carried. Dropping it
    outright would leave the weights summing to 0.94 and quietly shrink every player's
    WAR by 6%; spreading it keeps the total honest and the relative ordering intact.
    """
    fw = pd.read_csv(f"{WAR_DIR}/hybrid_facet_weights.csv", index_col=0)["rf"]
    cmap = pd.Series(tlw.concept_map(list(fw.index))).reindex(fw.index)
    w = fw.groupby(cmap).sum()
    keep = w.reindex(CONCEPTS.keys()).fillna(0.0)
    dropped = float(w.sum() - keep.sum())
    out = keep / keep.sum()
    return out, dropped


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


def concept_values(ea):
    """Per player per concept: a snap-weighted z of the averaged attributes, times
    volume - the same value = z * snaps shape the PFF facets use."""
    rows = []
    for name, (attrs, groups) in CONCEPTS.items():
        have = [a for a in attrs if a in ea.columns]
        if not have:
            continue
        d = ea[ea.group.isin(groups)].copy()
        if d.empty:
            continue
        # standardize each attribute within its position group, then average: an
        # 85 speed means something different at guard than at corner
        parts = []
        for a in have:
            g = d.groupby("group")[a]
            parts.append(((d[a] - g.transform("mean")) / g.transform("std").replace(0, 1)))
        d["score"] = np.mean(parts, axis=0)
        w_ = d.snaps.to_numpy(float)
        mu = np.average(d.score, weights=w_)
        sd = np.sqrt(np.average((d.score - mu) ** 2, weights=w_)) or 1.0
        d["z"] = (d.score - mu) / sd
        d["concept"] = name
        d["value"] = d.z * d.snaps
        rows.append(d[["id", "player", "team", "position", "group", "key", "overall",
                       "rank", "snaps", "concept", "z", "value"]])
    return pd.concat(rows, ignore_index=True)


def main():
    w, dropped = inherited_weights()
    print(f"inherited concept weights (discipline {dropped*100:.1f}% redistributed):")
    for c, v in w.sort_values(ascending=False).items():
        print(f"  {c:<18}{v*100:5.2f}%")

    ea = load_ea()
    print(f"\nEA players used: {len(ea)} over {ea.team.nunique()} teams")

    cv = concept_values(ea)
    print(f"player-concept rows: {len(cv)}")

    tot = (cv.groupby(["team", "concept"], as_index=False).value.sum()
             .pivot(index="team", columns="concept", values="value")
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
    cv["w"] = cv.concept.map(w)
    cv["sigma"] = cv.concept.map(sigma)
    cv["c_t"] = cv.team.map(c_t)
    cv["games"] = cv.team.map(games)
    cv["waa"] = cv.games * slope * cv.c_t * cv.w * cv.value / cv.sigma

    pool = float(((0.5 - REPL_WIN_PCT) * games).sum())
    league = cv.groupby("concept").snaps.sum()
    per_snap = {c: pool * w[c] / league[c] for c in league.index if league[c] > 0}
    cv["repl"] = cv.concept.map(per_snap).fillna(0.0) * cv.snaps
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
