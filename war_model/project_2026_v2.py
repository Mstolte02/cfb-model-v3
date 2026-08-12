"""Stage 8b: leakage-safe player projections for the published 2026 two-deep.

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

The first attempted fix made each historical room the size of the current two-deep,
but chose its members using target-season snap rank. That fixed density while still
revealing who later won playing time. The final fix trains on every published roster
member, including zero-snap players, and removes target-season role from both features
and population selection. Class comes from the CFBD roster on both sides.

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
# GENERIC "OL" IS DELIBERATELY UNMAPPED. CFBD does not say whether those players are
# tackles or interior linemen. Guessing would manufacture the position-group feature
# and label, so the projection uses only specific tackle/guard/center designations.
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
            "is_transfer", "stars", "rating", "team_massey", "group_code"]

MAX_RANK = 8.0   # beyond this a player is a name on the list, not a rotation slot


# ------------------------------------------------------------------ slot counts
def slot_counts(ros26):
    """Slots and starter slots per team-position group, taken from the 2026 two-deep.

    These describe serving and the post-projection historical team aggregation. They
    no longer select the player-model training population.
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


def build_population(w, rosters, K=None):
    """Every historical roster member, selected without target-season outcomes.

    The former population retained the K highest-snap players in the season being
    predicted.  Even though its features were lagged, that filter revealed who won
    the future playing-time competition.  The serving population is a published
    preseason depth chart, so the honest training analogue is the full published
    season roster: players with no target PFF row receive zero target WAR.

    ``K`` remains accepted for compatibility with older callers but is deliberately
    unused.  Position-room slot selection now happens only after players have been
    projected (see preseason_team_projection.py).
    """
    w = w.copy()
    w["key"] = w.player.map(norm_name)
    actual = (w.groupby(["season", "team", "group", "key"], as_index=False)
                .agg(player_id=("player_id", "first"), war=("war", "sum"),
                     snaps=("snaps", "sum")))
    base = rosters[["season", "team", "group", "player", "key"]].copy()
    pop = base.merge(actual, on=["season", "team", "group", "key"], how="left")
    pop["war"] = pop.war.fillna(0.0)
    pop["snaps"] = pop.snaps.fillna(0.0)
    pop["played"] = pop.snaps > 0
    return pop


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
    print("2026 two-deep slots per team-position group (serving/aggregation only):")
    print("  " + "  ".join(f"{g}:{K[g]}/{S[g]}" for g in sorted(K)) + "   (slots/starters)")

    w = build_history(war)
    fbs = set(recs.team.unique())
    rosters = load_rosters(WAR_YEARS, fbs)
    print(f"\nCFBD roster rows (FBS only): {len(rosters)}")

    pop = build_population(w, rosters, K)
    print(f"all-roster ex-ante population: {len(pop)} player-seasons")
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

    # CURRENT ELIGIBILITY COMES FROM THE CURRENT DEPTH CHART.  CFBD is still useful
    # as a fallback, but it is not allowed to overwrite a fresher source.  The old
    # order did exactly that and labelled Bear Bachmeier a freshman in 2026: CFBD's
    # roster said year 1 while both independent current charts, EA, and BYU's own
    # biography establish that 2025 was his true-freshman season.  This is a general
    # source-precedence fix, not a player-specific exception.
    cls26 = load_cfbd_classes(PROJECTION_YEAR, fbs)
    r = r.merge(cls26, on=["team", "key"], how="left", suffixes=("", "_cfbd"))
    chart_class = r["class"].map(CLASS_NUM)
    cfbd_class = r.class_num.copy()
    chart_matched = chart_class.notna()
    cfbd_matched = cfbd_class.notna()
    r["class_num"] = chart_class.fillna(cfbd_class)
    r["class_source"] = np.select(
        [chart_matched, ~chart_matched & cfbd_matched],
        ["current depth chart", "CFBD fallback"], default="unknown")
    if cls26.empty:
        print(f"\n  [warn] no CFBD {PROJECTION_YEAR} roster; current depth-chart "
              f"classes retained with no fallback for missing labels")
    else:
        both = chart_matched & cfbd_matched
        disagreements = int((chart_class[both] != cfbd_class[both]).sum())
        print(f"\nclass from current depth charts: {chart_matched.sum()} of {len(r)} "
              f"slots ({chart_matched.mean()*100:.1f}%); CFBD fills "
              f"{int((~chart_matched & cfbd_matched).sum())} missing labels")
        print(f"  CFBD disagreed on {disagreements} of {int(both.sum())} comparable "
              f"slots; current charts kept")

    # The displayed class and projection feature are deliberately the same resolved
    # value. Keeping the chart label also preserves its graduate tier, which CFBD's
    # coarser 1-4 field cannot express.
    shown = r.class_num.map({v: k for k, v in CLASS_NUM.items()})
    r["class"] = r["class"].where(chart_matched, shown)
    # the SAME ex-ante rank the training side computes: last season's snaps, ranked
    # within the team-position group. Not the two-deep's listed depth - see FEATURES.
    r["prior_rank"] = (r.groupby(["team", "broad_group"])
                        .snaps_lag1.rank(ascending=False, method="first",
                                         na_option="bottom")
                        .clip(upper=MAX_RANK))
    r["is_transfer"] = r.is_transfer.astype(int)
    r["group_code"] = r.broad_group.map(gcode)
    r["team_massey"] = r.team.map(ratings[ratings.season == 2025].set_index("team").massey)
    if "stars" not in r or r.stars.isna().all():
        r = r.merge(rec[["key", "stars", "rating"]].drop_duplicates("key"), on="key", how="left")

    r["proj_war"] = final.predict(r[FEATURES])
    r["imputed"] = ~r.has_history
    r.to_csv(f"{HERE}/projections_2026_v2.csv", index=False)
    print(f"\n2026: {len(r)} slots, projected WAR total {r.proj_war.sum():.0f}")
    print("projections_2026_v2.csv written")


if __name__ == "__main__":
    main()
