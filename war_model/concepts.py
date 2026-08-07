"""Football concepts, not features: within-concept PCA, concept ablation, quality interactions.

Four things were wrong with the way model_lab.py asked its questions.

1. INTERACTIONS WERE BUILT ON THE WRONG QUANTITY. A team facet total is
   value = z x snaps, summed and then standardized, so it carries quality AND volume.
   Multiplying two of those multiplies the volumes too: a team that throws a lot and
   runs a lot gets a large QB x RB interaction for no football reason at all. The
   interaction that means something is between QUALITIES - how good was the
   quarterback, how good was the protection - which is the snap-weighted mean z of
   each unit, with volume divided out. That is the player-level quantity.

2. LEAVE-ONE-FEATURE-OUT HAD NO POWER. The worst single removal cost .0006 RMSE,
   because 86 features contain six views of every throw and whichever one is removed,
   its correlates absorb the job. Removing a whole CONCEPT - every passing feature,
   every coverage feature - is the ablation that can actually move the number.

3. THE QUESTION WAS ABOUT FEATURES, NOT CONCEPTS. "Does WR_drop_rate help" is not a
   football question. "Does receiving help, given passing" is.

4. PCA WAS GLOBAL. A component mixing quarterback accuracy with linebacker tackling
   is not a latent football skill, it is an artefact of putting everything in one
   matrix. Within-concept PCA gives one interpretable score per concept.

Run: ./rbenv/bin/python concepts.py
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ the concepts
# A concept is a job on a football field, not a position and not a metric. Passing is
# the quarterback throwing; receiving is everyone catching; pass protection is
# everyone blocking on a pass. Position appears only where the job differs by it -
# interior pressure and edge pressure are the same concept, coverage by a corner and
# coverage by a safety are the same concept.
CONCEPTS = {
    "passing":         ["QB_pass", "QB_accuracy_pct", "QB_btt_rate", "QB_twp_rate",
                        "QB_completion_pct", "QB_positive_epa_pct", "QB_ypa",
                        "QB_qb_rating"],
    "qb_pressure":     ["QB_sack_pct", "QB_pressure_to_sack_rate"],
    "qb_rushing":      ["QB_run"],
    "rushing":         ["RB_run", "RB_ypa", "RB_yco_attempt", "RB_breakaway_pct",
                        "RB_elusive_rating"],
    "receiving":       ["WR_pass_route", "WR_offense", "WR_yprr", "WR_caught_pct",
                        "WR_contested_catch_rate", "WR_positive_epa_pct",
                        "WR_targeted_qb_rating", "WR_first_downs", "WR_touchdowns",
                        "WR_yards_per_reception", "WR_avg_depth_of_target",
                        "TE_pass_route", "TE_yprr", "TE_positive_epa_pct",
                        "TE_targeted_qb_rating", "TE_first_downs", "RB_pass_route"],
    "yac":             ["WR_yards_after_catch", "WR_avoided_tackles",
                        "TE_yards_after_catch", "RB_yards_after_catch"],
    "hands":           ["WR_hands_drop", "WR_drop_rate", "TE_hands_drop"],
    # The line is two position groups, so each blocking facet is named twice. Both
    # names have to be listed: concept_map() falls through to "_solo_<facet>" for
    # anything unclaimed, which keeps the facet in the model but takes it out of every
    # concept - and build_ea_war, which is keyed on concepts, then drops it entirely.
    # That is how 1,534 EA linemen silently went missing.
    "pass_protection": ["OT_pass_block", "OT_pressures_allowed", "OT_sacks_allowed",
                        "IOL_pass_block", "IOL_pressures_allowed",
                        "IOL_sacks_allowed", "TE_pass_block"],
    "run_blocking":    ["OT_run_block", "IOL_run_block", "TE_run_block",
                        "RB_run_block"],
    "pass_rush":       ["DI_pass_rush", "ED_pass_rush", "LB_pass_rush",
                        "CB-S_pass_rush", "DI_total_pressures", "ED_total_pressures",
                        "DI_sacks", "ED_sacks", "ED_hurries", "ED_qb_rating_against"],
    "run_defense":     ["DI_run", "ED_run", "LB_run", "CB-S_run", "DI_stops",
                        "LB_stops"],
    "coverage":        ["CB_coverage", "S_coverage", "LB_coverage", "CB_catch_rate",
                        "S_catch_rate", "CB_yards", "S_yards", "CB_qb_rating_against",
                        "CB_pass_break_ups", "CB-S_interceptions",
                        "CB_yards_after_catch"],
    "tackling":        ["LB_tackle", "CB-S_tackle", "DI-ED_tackle", "LB_tackles",
                        "LB_missed_tackle_rate", "CB-S_missed_tackle_rate"],
    "ball_security":   ["QB_hands_fumble", "RB_hands_fumble"],
    "discipline":      ["OT_penalties", "IOL_penalties", "DI-ED_defense_penalty",
                        "CB-LB-S_defense_penalty", "QB-RB-TE-WR_offense_penalty"],
}

# Which unit plays each concept, for the player-level quality scores. Concepts are
# played by the players the catalogue assigned to their features, so this is derived
# rather than declared - see unit_quality().

# Interactions worth asking about, as concept pairs. Stated as football claims:
# a quarterback needs protection, a back needs blocking, pressure and coverage feed
# each other, and receivers only matter if someone can throw.
CONCEPT_INTERACTIONS = [
    ("passing", "receiving"),
    ("passing", "pass_protection"),
    ("passing", "yac"),
    ("rushing", "run_blocking"),
    ("pass_rush", "coverage"),
    ("coverage", "tackling"),
    ("run_defense", "tackling"),
    ("pass_protection", "receiving"),
    ("passing", "rushing"),
    ("pass_rush", "run_defense"),
]


def load():
    fv = pd.read_parquet(f"{HERE}/candidate_values.parquet")
    Z = pd.read_csv(f"{HERE}/candidate_team_z.csv", index_col=[0, 1])
    recs = pd.read_csv(f"{HERE}/records.csv")
    nxt = recs[["season", "team", "adj_win_pct"]].copy()
    nxt["season"] -= 1
    nxt = nxt.rename(columns={"adj_win_pct": "y"}).set_index(["season", "team"])
    df = Z.join(nxt, how="inner").dropna(subset=["y"])
    return fv, df, [c for c in Z.columns if c in df.columns]


def concept_of(feature):
    for c, members in CONCEPTS.items():
        if feature in members:
            return c
    return None


def unit_quality(fv, index, team_map=None):
    """Per team-season, per concept: the snap-weighted MEAN z of that concept's players.

    This is the quality of the unit with volume divided out, which is what an
    interaction should be built from. The team facet total that model_lab used is
    sum(z * snaps) - a good unit that plays a lot and a mediocre unit that plays a lot
    are far apart on it for the wrong reason.
    """
    team_map = team_map or json.load(open(f"{HERE}/team_map.json"))
    fv = fv.copy()
    fv["team"] = fv.team_name.map(team_map)
    fv["concept"] = fv.facet.map(concept_of)
    fv = fv[fv.concept.notna()]
    num = (fv.assign(zs=fv.z * fv.snaps)
             .groupby(["season", "team", "concept"], as_index=False)
             .agg(zs=("zs", "sum"), sn=("snaps", "sum")))
    num["q"] = num.zs / num.sn.replace(0, np.nan)
    Q = num.pivot(index=["season", "team"], columns="concept", values="q")
    Q = Q.reindex(index).astype(float)
    # a concept a team has no qualifying players for sits at league average
    Q = Q.groupby(level="season").transform(lambda c: (c - c.mean()) / c.std(ddof=0))
    return Q.fillna(0.0)


def concept_pca(Z, names, seasons_fit, n_components=1):
    """One (or a few) components per concept, fitted on the training seasons only."""
    comps, loadings = {}, {}
    for c, members in CONCEPTS.items():
        cols = [m for m in members if m in names]
        if not cols:
            continue
        k = min(n_components, len(cols))
        sc = StandardScaler().fit(Z.loc[seasons_fit, cols])
        p = PCA(n_components=k).fit(sc.transform(Z.loc[seasons_fit, cols]))
        T = p.transform(sc.transform(Z[cols]))
        # orient so a positive score means "more of the concept", by aligning with the
        # mean of the members rather than leaving the sign to numpy
        for j in range(k):
            if np.corrcoef(T[:, j], Z[cols].mean(axis=1))[0, 1] < 0:
                T[:, j] *= -1
                p.components_[j] *= -1
            comps[f"{c}" if k == 1 else f"{c}_{j+1}"] = T[:, j]
        loadings[c] = {
            "members": cols,
            "explained": [float(v) for v in p.explained_variance_ratio_],
            "pc1": {m: float(l) for m, l in zip(cols, p.components_[0])},
        }
    return pd.DataFrame(comps, index=Z.index), loadings


def fit_ridge(X, y):
    return RidgeCV(alphas=np.logspace(-3, 4, 60)).fit(X, y)


def cv_rmse(X, y, seasons, cols=None):
    """Season-blocked CV RMSE on a column subset."""
    Xv = X if cols is None else X[:, cols]
    if Xv.shape[1] == 0:
        return float(np.sqrt(mean_squared_error(y, np.full(len(y), y.mean()))))
    P = np.full(len(y), np.nan)
    for s in sorted(set(seasons)):
        tr, te = seasons != s, seasons == s
        if tr.sum() < 200:
            continue
        sc = StandardScaler().fit(Xv[tr])
        m = fit_ridge(sc.transform(Xv[tr]), y[tr])
        P[te] = m.predict(sc.transform(Xv[te]))
    ok = ~np.isnan(P)
    return float(np.sqrt(mean_squared_error(y[ok], P[ok])))


def main():
    fv, df, names = load()
    y = df.y.to_numpy(float)
    seasons = df.index.get_level_values("season").to_numpy()
    Z = df[names]

    unassigned = [n for n in names if concept_of(n) is None]
    print(f"features {len(names)}  concepts {len(CONCEPTS)}  "
          f"unassigned {len(unassigned)}")
    if unassigned:
        print(f"  NOT IN ANY CONCEPT: {', '.join(unassigned)}")
    sizes = {c: len([m for m in ms if m in names]) for c, ms in CONCEPTS.items()}
    print("  " + ", ".join(f"{c} {n}" for c, n in sorted(sizes.items(),
                                                         key=lambda kv: -kv[1])))

    # ---- 1. within-concept PCA ------------------------------------------------
    print("\n" + "=" * 78)
    print("1. PCA WITHIN EACH CONCEPT — is each one a single latent skill?")
    print("=" * 78)
    C1, loadings = concept_pca(Z, names, Z.index, n_components=1)
    for c, L in sorted(loadings.items(), key=lambda kv: -kv[1]["explained"][0]):
        if len(L["members"]) < 2:
            continue
        top = sorted(L["pc1"].items(), key=lambda kv: -abs(kv[1]))[:4]
        print(f"  {c:<17} {len(L['members']):>2} features, "
              f"first component {L['explained'][0]*100:4.0f}%   "
              + ", ".join(f"{k} {v:+.2f}" for k, v in top))

    # ---- 2. concepts as the whole feature set ---------------------------------
    print("\n" + "=" * 78)
    print("2. CONCEPT SCORES vs 86 RAW FEATURES")
    print("=" * 78)
    Xraw = Z.to_numpy(float)
    base = cv_rmse(Xraw, y, seasons)
    Xc1 = C1.to_numpy(float)
    r_c1 = cv_rmse(Xc1, y, seasons)
    C2, _ = concept_pca(Z, names, Z.index, n_components=2)
    r_c2 = cv_rmse(C2.to_numpy(float), y, seasons)
    Q = unit_quality(fv, df.index)
    r_q = cv_rmse(Q.to_numpy(float), y, seasons)
    print(f"  86 raw features                      rmse {base:.4f}")
    print(f"  {Xc1.shape[1]} concept scores (1 PC each)          rmse {r_c1:.4f}"
          f"   {r_c1-base:+.4f}")
    print(f"  {C2.shape[1]} concept scores (2 PCs each)         rmse {r_c2:.4f}"
          f"   {r_c2-base:+.4f}")
    print(f"  {Q.shape[1]} unit qualities (snap-weighted mean z)  rmse {r_q:.4f}"
          f"   {r_q-base:+.4f}")

    # ---- 3. leave-one-CONCEPT-out --------------------------------------------
    print("\n" + "=" * 78)
    print("3. LEAVE-ONE-CONCEPT-OUT — removing every feature of a concept at once")
    print("=" * 78)
    rows = []
    for c in CONCEPTS:
        cols = [i for i, n in enumerate(names) if concept_of(n) != c]
        n_drop = len(names) - len(cols)
        if not n_drop:
            continue
        r = cv_rmse(Xraw, y, seasons, cols)
        rows.append({"concept": c, "features_dropped": n_drop, "rmse": r,
                     "delta": r - base})
    lo = pd.DataFrame(rows).sort_values("delta", ascending=False)
    print(f"  all concepts: rmse {base:.4f}")
    print(lo.round(5).to_string(index=False))

    # ---- 4. forward selection BY CONCEPT -------------------------------------
    print("\n" + "=" * 78)
    print("4. WHICH CONCEPTS EARN A PLACE — forward selection, concept at a time")
    print("=" * 78)
    chosen, remaining = [], [c for c in CONCEPTS if sizes.get(c)]
    best = cv_rmse(Xraw, y, seasons, [])
    print(f"  intercept only: rmse {best:.4f}")
    while remaining:
        scored = []
        for c in remaining:
            cols = [i for i, n in enumerate(names)
                    if concept_of(n) in set(chosen) | {c}]
            scored.append((cv_rmse(Xraw, y, seasons, cols), c))
        scored.sort()
        r, c = scored[0]
        gain = best - r
        chosen.append(c)
        remaining.remove(c)
        flag = "" if gain > 0.0005 else "   <- stops paying here"
        print(f"  +{c:<17} ({len(chosen):>2}) rmse {r:.4f}  gain {gain:+.4f}{flag}")
        best = r

    # ---- 5. player-level (quality) interactions -------------------------------
    print("\n" + "=" * 78)
    print("5. INTERACTIONS ON UNIT QUALITY, NOT TEAM TOTALS")
    print("=" * 78)
    qn = list(Q.columns)
    Qv = Q.to_numpy(float)
    q_base = cv_rmse(Qv, y, seasons)
    print(f"  {len(qn)} unit qualities alone: rmse {q_base:.4f}")
    irows = []
    for a, b in CONCEPT_INTERACTIONS:
        if a not in qn or b not in qn:
            continue
        X2 = np.column_stack([Qv, Qv[:, qn.index(a)] * Qv[:, qn.index(b)]])
        r = cv_rmse(X2, y, seasons)
        irows.append({"interaction": f"{a} x {b}", "rmse": r, "delta": r - q_base})
    it = pd.DataFrame(irows).sort_values("delta")
    print(it.round(5).to_string(index=False))
    keep = it[it.delta < -0.0002]
    if len(keep):
        cols = [Qv]
        for _, r in keep.iterrows():
            a, b = r.interaction.split(" x ")
            cols.append((Qv[:, qn.index(a)] * Qv[:, qn.index(b)])[:, None])
        r_all = cv_rmse(np.hstack(cols), y, seasons)
        print(f"\n  all {len(keep)} helpful interactions together: rmse {r_all:.4f}  "
              f"{r_all-q_base:+.4f}")
    else:
        print("\n  no interaction improves on the qualities alone")

    json.dump({
        "concepts": {c: [m for m in ms if m in names] for c, ms in CONCEPTS.items()},
        "unassigned": unassigned,
        "within_concept_pca": loadings,
        "representation": {
            "raw_86": base, "concept_1pc": r_c1, "concept_2pc": r_c2,
            "unit_quality": r_q,
            "n_concept_1pc": int(Xc1.shape[1]), "n_concept_2pc": int(C2.shape[1]),
            "n_unit_quality": int(Q.shape[1])},
        "leave_one_concept_out": json.loads(lo.round(6).to_json(orient="records")),
        "forward_by_concept": chosen,
        "quality_interactions": json.loads(it.round(6).to_json(orient="records")),
        "quality_base_rmse": q_base,
    }, open(f"{HERE}/concepts.json", "w"), indent=1)
    print("\n-> concepts.json")


if __name__ == "__main__":
    main()
