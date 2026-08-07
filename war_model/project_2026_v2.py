"""Stage 8b: 2026 projections, with the training population fixed.

The v1 model projects a true freshman who has never taken a snap as the best receiver
in the country, ahead of the actual best receiver in the country. Three defects
compound to produce that, and all three are population defects rather than modelling
ones:

  1. is_starter meant different things on each side of the model. In training it was
     rank_in_group <= 1 - the single highest-snap player at that position, decided by
     the outcome season. In serving it was depth == 1 on the two-deep, which is 2.8
     players per team at receiver and 4.6 on the line. So 9% of training receivers
     carried the flag against 50% of served ones, and those 9% averaged 1,038 snaps.
     The model learned "starter means team's workhorse" and then applied it to
     everybody listed on the first line.

  2. Only players who logged a snap were ever trained on. A freshman who redshirted
     produced no row, so the model estimated E[WAR | freshman who played] and served
     it to every freshman on the roster.

  3. class_num was approximated as prior_seasons + 1 in training but read from the
     real roster in serving, so the two were not the same variable.

The fix is to make both sides the same object: a two-deep. For every historical
team-position group we take the K players who actually occupied those slots, where K
is that group's slot count on the 2026 two-deep, and fill any short group from the
CFBD roster at zero WAR. Class comes from the CFBD roster in both.

AND THAT FIXED THE WRONG HALF OF DEFECT 1. Matching the DENSITY of the starter flag
left its MEANING different on the two sides: in training it still came from
snap_rank, which is the player's rank by snaps in the season being predicted, and at
serve time it was still somebody's opinion in August. The model learned what a
realised workhorse is worth and applied it to a listed one.

Measured on an otherwise identical model and the same 2025 holdout, the flag was
worth 0.064 of correlation - r = .617 without it against .681 with it. That 10% was
not accuracy; it was the model being told part of the answer. is_starter is gone and
`prior_rank` - rank within the team-position group by LAST season's snaps, computed
by the same expression on both sides - replaces it.
"""
import json, os, sys
import numpy as np
import pandas as pd

import artifacts
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from build_roster_2026 import norm_name, PFF_TO_GROUP
from facets import YEARS as WAR_YEARS

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = f"{HERE}/cfbd_cache"
CLASS_NUM = {"FR": 1, "SO": 2, "JR": 3, "SR": 4, "GR": 5}
PROJECTION_YEAR = 2026

# CFBD roster positions -> the two-deep's broad groups.
#
# GENERIC "OL" IS DELIBERATELY UNMAPPED. CFBD labels roughly nine in ten linemen "OL"
# with no tackle/interior distinction, and these rows are used for one thing only:
# topping a historical team-position room up to its two-deep slot count with
# zero-snap, zero-WAR filler, so each past season is the same shape of object as the
# 2026 chart. Guessing which room an unlabelled lineman belongs to would put a real
# person in a room he may not play in to fill a slot that exists to hold a zero, so
# they are dropped and the specific labels below are kept. The line is the one unit
# where this costs almost nothing: five linemen play essentially every snap, so an OT
# or IOL room is filled by real PFF players and rarely needs filler at all.
CFBD_TO_GROUP = {
    "QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
    "OT": "OT", "G": "IOL", "C": "IOL", "OG": "IOL",
    "DL": "DT", "DT": "DT", "NT": "DT", "DE": "EDGE", "EDGE": "EDGE",
    "LB": "LB", "CB": "CB", "DB": "CB", "S": "SAF",
}

# is_starter USED TO BE IN HERE AND IT WAS A LEAK.
#
# In training it was computed from snap_rank - the player's rank by snaps IN THE
# SEASON BEING PREDICTED. In serving it was depth == 1 on the preseason two-deep.
# The v1 note below records the density mismatch being fixed by giving both sides
# the same per-group count, and that fixed the wrong half of the problem: the flag
# still meant "finished the season as one of this group's busiest players" in
# training and "somebody listed him first in August" at serve time. The model
# learned the former, which is worth a great deal, and applied it to the latter,
# which is worth much less - so every listed starter was handed the value of a
# realised one, and the reported holdout r was measuring a task nobody can perform.
#
# `prior_rank` replaces it and is strictly ex ante: the player's rank within his
# team-position group by LAST season's snaps, computed identically on both sides.
# Listed two-deep depth is deliberately NOT a feature even though it exists at serve
# time, because it does not exist historically, and a column that means different
# things on the two sides of the model is the defect being removed.
FEATURES = ["war_lag1", "war_lag2", "war_lag3", "snaps_lag1", "snaps_lag2",
            "rate_lag1", "share_lag1", "prior_rank", "prior_seasons", "class_num",
            "is_transfer", "stars", "rating", "team_rating", "team_massey", "group_code"]

MAX_RANK = 8.0   # beyond this a player is a name on the list, not a rotation slot


# ------------------------------------------------------------------ slot counts
def slot_counts(ros26):
    """Slots and starter slots per team-position group, taken from the 2026 two-deep.

    These define the population on BOTH sides of the model, which is the whole point.
    """
    per_team = ros26.groupby(["team", "broad_group"]).agg(
        slots=("player", "size"), starters=("is_starter", "sum"))
    k = per_team.groupby("broad_group").slots.median().round().astype(int)
    s = per_team.groupby("broad_group").starters.median().round().astype(int)
    return k.to_dict(), s.to_dict()


# ------------------------------------------------------------------ roster zeros
def load_rosters(years, fbs_teams):
    """CFBD rosters, restricted to FBS and mapped onto the two-deep's groups.

    Also the source of real class years, which v1 had to approximate.
    """
    rows = []
    for y in years:
        p = f"{CACHE}/roster_{y}.json"
        if not os.path.exists(p):
            continue
        d = pd.DataFrame(json.load(open(p)))
        d = d[d.team.isin(fbs_teams)].copy()
        d["group"] = d.position.map(CFBD_TO_GROUP)
        d = d[d.group.notna()]
        d["player"] = (d.firstName.fillna("") + " " + d.lastName.fillna("")).str.strip()
        d["key"] = d.player.map(norm_name)
        # 'year' is the class (1-4); some rows carry a calendar year instead
        d["class_num"] = pd.to_numeric(d.year, errors="coerce")
        d.loc[(d.class_num < 1) | (d.class_num > 5), "class_num"] = np.nan
        rows.append(d[["team", "group", "player", "key", "class_num"]].assign(season=y))
    r = pd.concat(rows, ignore_index=True)
    return r.drop_duplicates(["season", "team", "group", "key"])


def load_cfbd_classes(year, fbs_teams):
    """[team, key, class_num] for one season, WITHOUT the position-group filter.

    load_rosters drops anyone whose CFBD position has no two-deep group, which is
    right when the rows exist to fill a position room and wrong here: a long snapper's
    class year is still his class year. This is the serving-side counterpart of the
    class column the training side reads, so it has to cover the same people the
    two-deep does, not the same people the population build does.

    Returns an empty frame if CFBD has not published the season, which is a real state
    and not an error - the caller falls back and says so.
    """
    p = f"{CACHE}/roster_{year}.json"
    if not os.path.exists(p):
        return pd.DataFrame(columns=["team", "key", "class_num"])
    d = pd.DataFrame(json.load(open(p)))
    if d.empty:
        return pd.DataFrame(columns=["team", "key", "class_num"])
    d = d[d.team.isin(fbs_teams)].copy()
    d["key"] = (d.firstName.fillna("") + " " + d.lastName.fillna(""
                )).str.strip().map(norm_name)
    d["class_num"] = pd.to_numeric(d.year, errors="coerce")
    d.loc[(d.class_num < 1) | (d.class_num > 5), "class_num"] = np.nan
    d = d[(d.key != "") & d.class_num.notna()]
    # a name that resolves to two players on one roster cannot be assigned a class
    n = d.groupby(["team", "key"]).class_num.nunique()
    d = d[~d.set_index(["team", "key"]).index.isin(n[n > 1].index)]
    return d.drop_duplicates(["team", "key"])[["team", "key", "class_num"]]


def build_population(w, rosters, K):
    """One row per two-deep slot per historical season, players and non-players alike.

    Within a team-position group the slots go to the K highest-snap players; if fewer
    than K played, the rest are filled from the roster at zero. That reproduces, for
    each past season, the same kind of object the 2026 two-deep is.
    """
    w = w.copy()
    w["key"] = w.player.map(norm_name)
    w["snap_rank"] = w.groupby(["season", "team", "group"]).snaps.rank(
        ascending=False, method="first")
    kept = w[w.snap_rank <= w.group.map(K).fillna(2)].copy()
    kept["played"] = True

    # roster members who did not take a snap, ranked behind everyone who did
    have = set(zip(kept.season, kept.team, kept.group, kept.key))
    ros = rosters[[k not in have for k in
                   zip(rosters.season, rosters.team, rosters.group, rosters.key)]].copy()
    # how many slots each group still has open
    filled = kept.groupby(["season", "team", "group"]).size().rename("n")
    ros = ros.merge(filled.reset_index(), on=["season", "team", "group"], how="left")
    ros["n"] = ros.n.fillna(0)
    ros["room"] = ros.group.map(K).fillna(2) - ros.n
    ros = ros[ros.room > 0]
    # no signal to order non-players by, so take them arbitrarily but reproducibly
    ros = ros.sort_values(["season", "team", "group", "key"])
    ros["slot"] = ros.groupby(["season", "team", "group"]).cumcount() + 1
    ros = ros[ros.slot <= ros.room]

    zeros = pd.DataFrame({
        "season": ros.season, "team": ros.team, "group": ros.group,
        "player": ros.player, "key": ros.key, "player_id": np.nan,
        "war": 0.0, "snaps": 0.0, "played": False,
        "snap_rank": ros.n + ros.slot,
    })
    return pd.concat([kept, zeros], ignore_index=True)


def build_history(war):
    w = war.copy()
    w["group"] = w.position.map(PFF_TO_GROUP)
    w = w[w.group.notna()]
    return w.groupby(["season", "player_id", "player", "team", "group"],
                     as_index=False).agg(war=("war", "sum"), snaps=("snaps", "sum"))


def _unique_by_key(frame, cols):
    """Name-keyed lookup that REFUSES an ambiguous key instead of picking one.

    The lag merges here were `drop_duplicates("key")` followed by a merge on the
    normalized name across all of FBS - so two players sharing a name silently handed
    one of them the other's WAR history, on a key with no team and no position in it.
    """
    n = frame.groupby("key").player_id.transform("nunique")
    ok = frame[n == 1].drop_duplicates("key")
    return ok[cols], int((n > 1).sum())


def make_training(pop, w, ratings, rec, rosters, S, seasons):
    """One row per slot per target season, with features known before the season."""
    w = w.copy()
    w["key"] = w.player.map(norm_name)
    # team-group snap share, the ex-ante role signal that replaces leaked rank
    w["share"] = w.snaps / w.groupby(["season", "team", "group"]).snaps.transform("sum")

    rows, refused = [], 0
    for t1 in seasons:
        cur = pop[pop.season == t1].copy()
        for lag in (1, 2, 3):
            h = w[w.season == t1 - lag]
            ren = {"war": f"war_lag{lag}", "snaps": f"snaps_lag{lag}",
                   "share": f"share_lag{lag}"}
            # PLAYER_ID FIRST. Most of the population carries one, and where it does
            # the join is exact; only the roster-filled slots (a man on the roster who
            # took no snap, so he has no PFF row this season) need the name.
            byid = h[["player_id", "war", "snaps", "share"]].rename(columns=ren)
            byid = byid[byid.player_id.notna()].drop_duplicates("player_id")
            cur = cur.merge(byid, on="player_id", how="left")

            need = cur[list(ren.values())[0]].isna()
            if need.any():
                cols = ["key", "war", "snaps", "share"]
                uniq, amb = _unique_by_key(h, cols)
                refused += amb
                uniq = uniq.rename(columns=ren)
                m = cur.loc[need, ["key"]].merge(uniq, on="key", how="left")
                m.index = cur.index[need]
                for c in ren.values():
                    cur.loc[need, c] = m[c]

        cur["prior_seasons"] = sum(cur[f"snaps_lag{l}"].notna() & (cur[f"snaps_lag{l}"] > 0)
                                   for l in (1, 2, 3)).astype(float)
        cur["rate_lag1"] = cur.war_lag1 / cur.snaps_lag1.replace(0, np.nan)
        prev, amb = _unique_by_key(w[w.season == t1 - 1], ["key", "team"])
        refused += amb
        cur = cur.merge(prev.rename(columns={"team": "prev_team"}), on="key", how="left")
        cur["is_transfer"] = ((cur.prev_team.notna()) & (cur.prev_team != cur.team)).astype(int)
        cur["target_season"] = t1
        rows.append(cur)
    tr = pd.concat(rows, ignore_index=True)
    print(f"  ambiguous name keys refused rather than guessed: {refused}")

    # EX-ANTE role: rank within the team-position group by LAST season's snaps.
    # Computed the same way on both sides of the model; see FEATURES.
    tr["prior_rank"] = (tr.groupby(["target_season", "team", "group"])
                          .snaps_lag1.rank(ascending=False, method="first",
                                           na_option="bottom")
                          .clip(upper=MAX_RANK))

    # real class from the CFBD roster, the same variable served in 2026
    cls = rosters[["season", "team", "key", "class_num"]].drop_duplicates(
        ["season", "team", "key"])
    tr = tr.merge(cls, on=["season", "team", "key"], how="left")

    tr = tr.merge(rec[["key", "stars", "rating"]].drop_duplicates("key"), on="key", how="left")
    tr["team_rating"] = tr.team.map(rec.groupby("committedTo").rating.mean())
    r = ratings[["season", "team", "massey"]].copy()
    r["season"] += 1
    tr = tr.merge(r.rename(columns={"massey": "team_massey"}), on=["season", "team"], how="left")
    return tr


def fit(seed=0):
    return HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, random_state=seed)


def main():
    from build_recruiting import load_recruits
    war = pd.read_csv(f"{HERE}/{artifacts.PLAYER_WAR}")
    ratings = pd.read_csv(f"{HERE}/{artifacts.TEAM_RATINGS}")
    recs = pd.read_csv(f"{HERE}/records.csv")
    rec = load_recruits()
    ros26 = pd.read_csv(f"{HERE}/roster_2026.csv")

    K, S = slot_counts(ros26)
    print("two-deep slots per team-position group (defines BOTH populations):")
    print("  " + "  ".join(f"{g}:{K[g]}/{S[g]}" for g in sorted(K)) + "   (slots/starters)")

    w = build_history(war)
    fbs = set(recs.team.unique())
    rosters = load_rosters(WAR_YEARS, fbs)
    print(f"\nCFBD roster rows (FBS only): {len(rosters)}")

    pop = build_population(w, rosters, K)
    print(f"two-deep-equivalent population: {len(pop)} slot-seasons")
    print(f"  of which never took a snap: {int((~pop.played).sum())} "
          f"({(~pop.played).mean()*100:.1f}%)   [v1 trained on ZERO of these]")

    # every season that has a usable prior season to learn a transition from
    targets = [y for y in WAR_YEARS if (y - 1) in set(WAR_YEARS)]
    tr = make_training(pop, w, ratings, rec, rosters, S, targets)
    print(f"target seasons: {targets}")
    groups = sorted(w.group.dropna().unique())
    gcode = {g: i for i, g in enumerate(groups)}
    tr["group_code"] = tr.group.map(gcode)
    tr["share_lag1"] = tr.share_lag1.fillna(0.0)
    print(f"\ntraining rows: {len(tr)}")
    print(f"  mean prior_rank      train {tr.prior_rank.mean():.2f}")

    # ---- holdout ---------------------------------------------------------
    trn, tst = tr[tr.target_season < 2025], tr[tr.target_season == 2025]
    m = fit().fit(trn[FEATURES], trn.war)
    pred = m.predict(tst[FEATURES])
    print(f"\nholdout 2025 (n={len(tst)}), trained on 2022-24:")
    print(f"  model          r = {np.corrcoef(pred, tst.war)[0,1]:.3f}  "
          f"MAE = {mean_absolute_error(tst.war, pred):.4f}")
    naive = tst.war_lag1.fillna(0)
    print(f"  carry-forward  r = {np.corrcoef(naive, tst.war)[0,1]:.3f}  "
          f"MAE = {mean_absolute_error(tst.war, naive):.4f}")

    noh = tst[tst.prior_seasons == 0]
    p2 = m.predict(noh[FEATURES])
    print(f"\n  no prior snaps (n={len(noh)}):  r = {np.corrcoef(p2, noh.war)[0,1]:.3f}  "
          f"MAE = {mean_absolute_error(noh.war, p2):.4f}")
    print(f"    calibration: predicted mean {p2.mean():.4f} vs actual {noh.war.mean():.4f}")

    # The report used to carry these numbers as literals, which meant they kept
    # describing an older build after the model changed. Emit them instead.
    grp_mean = trn.groupby("group_code").war.mean()
    base = tst.group_code.map(grp_mean)
    metrics = {
        "build": artifacts.BUILD,
        "holdout_n": int(len(tst)),
        "holdout_r": round(float(np.corrcoef(pred, tst.war)[0, 1]), 3),
        "holdout_mae": round(float(mean_absolute_error(tst.war, pred)), 4),
        "carry_r": round(float(np.corrcoef(naive, tst.war)[0, 1]), 3),
        "carry_mae": round(float(mean_absolute_error(tst.war, naive)), 4),
        "posmean_r": round(float(np.corrcoef(base, tst.war)[0, 1]), 3),
        "nohist_n": int(len(noh)),
        "nohist_r": round(float(np.corrcoef(p2, noh.war)[0, 1]), 3),
        "nohist_mae": round(float(mean_absolute_error(noh.war, p2)), 4),
        "nohist_base_mae": round(float(mean_absolute_error(
            noh.war, noh.group_code.map(grp_mean))), 4),
        "nohist_pred_mean": round(float(p2.mean()), 4),
        "nohist_actual_mean": round(float(noh.war.mean()), 4),
        "train_rows": int(len(tr)),
        "never_played_rows": int((~pop.played).sum()),
        "features": FEATURES,
        "ex_ante_only": True,
    }
    json.dump(metrics, open(f"{HERE}/projection_metrics.json", "w"), indent=1)

    # ---- refit and project 2026 -----------------------------------------
    final = fit().fit(tr[FEATURES], tr.war)
    r = ros26.copy()
    r["war_lag1"] = r.war_2025.where(r.snaps_2025 > 0)
    r["war_lag2"] = r.war_2024.where(r.snaps_2024 > 0)
    r["war_lag3"] = r.war_2023.where(r.snaps_2023 > 0)
    r["snaps_lag1"] = r.snaps_2025.where(r.snaps_2025 > 0)
    r["snaps_lag2"] = r.snaps_2024.where(r.snaps_2024 > 0)
    r["rate_lag1"] = r.war_lag1 / r.snaps_lag1
    w25 = w[w.season == 2025].copy()
    w25["key"] = w25.player.map(norm_name)
    w25["share"] = w25.snaps / w25.groupby(["team", "group"]).snaps.transform("sum")
    r["key"] = r.player.map(norm_name)
    r = r.merge(w25.drop_duplicates("key")[["key", "share"]].rename(
        columns={"share": "share_lag1"}), on="key", how="left")
    r["share_lag1"] = r.share_lag1.fillna(0.0)

    # CLASS COMES FROM THE CFBD ROSTER ON BOTH SIDES, which is what this file has
    # claimed since it was written and could not do until CFBD published 2026. While
    # that endpoint returned `[]` the serving side read the two-deep's own class
    # column instead, and the two are not the same variable: they disagree on 21.8%
    # of matched slots, almost all by a year, and Ourlads carries a GR tier that
    # CFBD's 1-4 scale does not. The model was trained on one and served the other.
    cls26 = load_cfbd_classes(PROJECTION_YEAR, fbs)
    r = r.merge(cls26, on=["team", "key"], how="left", suffixes=("", "_cfbd"))
    fallback = r["class"].map(CLASS_NUM)
    matched = r.class_num.notna()
    r["class_num"] = r.class_num.fillna(fallback)
    if cls26.empty:
        print(f"\n  [warn] no CFBD {PROJECTION_YEAR} roster; class falls back to the "
              f"two-deep, which is NOT the variable the model was trained on")
    else:
        print(f"\nclass from the CFBD {PROJECTION_YEAR} roster: {matched.sum()} of "
              f"{len(r)} slots ({matched.mean()*100:.1f}%); the rest keep the "
              f"two-deep's own class year")
        both = r[matched & fallback.notna()]
        print(f"  disagreed with the two-deep on {(both.class_num != fallback[both.index]).sum()} "
              f"of {len(both)} matched slots")

    # THE DISPLAYED CLASS IS THE ONE THE MODEL USED. final_report.py publishes this
    # column and make_html.py puts it in a table, so leaving it as the chart's own
    # label would print "SR" beside a projection conditioned on a junior - the exact
    # two-meanings-one-column defect the rest of this file exists to remove.
    #
    # GR IS THE ONE EXCEPTION, and it is not an exception to that rule. CFBD's scale
    # runs 1-4 and has no graduate tier at all, so a player the chart calls GR and
    # CFBD calls 4 is not two sources disagreeing - it is one source being coarser.
    # Collapsing him to SR would be discarding a true label to match a scale that
    # cannot express it, and it emptied the app's Graduates filter from 194 slots to
    # 18. The ±1-year cases below are real disagreements and do resolve to CFBD.
    shown = r.class_num.map({v: k for k, v in CLASS_NUM.items()})
    coarser = matched & (r["class"] == "GR") & (r.class_num == 4)
    r["class"] = shown.where(matched, r["class"]).mask(coarser, "GR")
    # the SAME ex-ante rank the training side computes: last season's snaps, ranked
    # within the team-position group. Not the two-deep's listed depth - see FEATURES.
    r["prior_rank"] = (r.groupby(["team", "broad_group"])
                        .snaps_lag1.rank(ascending=False, method="first",
                                         na_option="bottom")
                        .clip(upper=MAX_RANK))
    r["is_transfer"] = r.is_transfer.astype(int)
    r["group_code"] = r.broad_group.map(gcode)
    r["team_massey"] = r.team.map(ratings[ratings.season == 2025].set_index("team").massey)
    r["team_rating"] = r.team.map(rec.groupby("committedTo").rating.mean())
    if "stars" not in r or r.stars.isna().all():
        r = r.merge(rec[["key", "stars", "rating"]].drop_duplicates("key"), on="key", how="left")

    r["proj_war"] = final.predict(r[FEATURES])
    r["imputed"] = ~r.has_history
    r.to_csv(f"{HERE}/projections_2026_v2.csv", index=False)
    print(f"\n2026: {len(r)} slots, projected WAR total {r.proj_war.sum():.0f}")
    print("projections_2026_v2.csv written")


if __name__ == "__main__":
    main()
