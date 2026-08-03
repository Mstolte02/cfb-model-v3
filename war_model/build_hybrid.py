"""Stages 2-3 again, on the PFF + CFBD facet set rather than PFF alone.

Selection was empirical (see horse_race.py / robustness.py): adding the full CFBD
facet set to the PFF set is worth +0.017 on both the same-season and the next-season
target, consistently across seeds. Adding only the QB facets captures about two thirds
of that. Subsetting further did not help, so the whole set goes in.

Facet names collide between the sources - both define pass_qb, cov_db, recv_wr and
others - so every CFBD facet carries a cfbd_ prefix. The two measure genuinely
different things under those shared names, which is exactly why keeping both helps.

Player identity is unified by normalized name within team-season: a CFBD player who
matches a PFF player shares his id and their facet values accumulate onto one row;
one who does not keeps a synthetic cfbd: id so that his contribution still reaches
his team's total instead of vanishing.

Writes the hybrid outputs under their own names, leaving the PFF-only artifacts in
place so the two can be compared.
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import nnls
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_predict

from facets import FACETS, YEARS
from cfbd_facets import CFBD_FACETS
from build_massey import massey_matrix
from build_war import team_sensitivity, REPL_WIN_PCT
from build_roster_2026 import norm_name

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM_MAP = json.load(open(f"{HERE}/team_map.json"))

# Both sources' position labels collapsed onto one coarse group, so a CFBD row can
# only join a PFF player who plays the same kind of football. PFF uses HB/DI/ED/T/G/C
# where CFBD uses RB/DL/DE/EDGE/OL, hence the two vocabularies in one table.
POS_GROUP = {
    "QB": "QB",
    "HB": "RB", "RB": "RB", "FB": "RB",
    "WR": "WR", "TE": "TE",
    "T": "OL", "G": "OL", "C": "OL", "OL": "OL", "OT": "OL", "OG": "OL",
    "DI": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "ED": "DL", "DE": "DL", "EDGE": "DL",
    "LB": "LB",
    "CB": "DB", "S": "DB", "DB": "DB",
}


def unified_facets():
    """One facet frame covering both sources, on a shared player identity.

    WAR_FEATURES=handbuilt reverts to the twenty facets I wrote by hand. The default
    is the 86 generated candidates from candidates.py, which fixes the coverage
    unfairness in that list: a receiver had three ways to earn value against a running
    back's seven, and 29% of his measured value came from run blocking.
    """
    source = os.environ.get("WAR_FEATURES", "candidates").lower()
    if source == "candidates" and os.path.exists(f"{HERE}/candidate_values.parquet"):
        p = pd.read_parquet(f"{HERE}/candidate_values.parquet")
        print(f"feature set: {p.facet.nunique()} generated candidates")
    else:
        p = pd.read_parquet(f"{HERE}/facet_values.parquet")
        print(f"feature set: {p.facet.nunique()} hand-built facets")
    p["team"] = p.team_name.map(TEAM_MAP)
    p["key"] = p.player.map(norm_name)
    p["uid"] = p.player_id.astype(str)
    p["source"] = "pff"

    c = pd.read_parquet(f"{HERE}/cfbd_facet_values.parquet")
    c = c.rename(columns={"cfbd_team": "team"})
    c["key"] = c.player.map(norm_name)
    c["facet"] = "cfbd_" + c.facet
    c["source"] = "cfbd"

    # Inherit the PFF id where name, team AND position group agree. Name+team alone
    # is not enough: Utah State had a QB and a WR both named Anthony Garcia, and the
    # looser merge handed the quarterback's passing value to the receiver, which was
    # enough on its own to make him the highest-projected receiver in the country.
    p["grp"] = p.position.map(POS_GROUP)
    c["grp"] = c.position.map(POS_GROUP)
    pk = (p.dropna(subset=["grp"]).drop_duplicates(["season", "team", "key", "grp"])
           [["season", "team", "key", "grp", "uid"]].rename(columns={"uid": "pff_uid"}))
    c = c.merge(pk, on=["season", "team", "key", "grp"], how="left")

    # SECOND PASS, because requiring the groups to agree was too strict and it split
    # 3,260 player-seasons across two identities - one PFF row with the snaps and one
    # synthetic cfbd: row with none. The two sources disagree about position far more
    # often than they disagree about who the player is, and by far the commonest case
    # (1,559 of those splits) is the edge rusher: PFF says ED, CFBD says LB. Colin
    # Simmons and Dylan Stewart both existed twice, and the 2026 roster join then
    # matched the empty half and called them players with no FBS history.
    #
    # So a mismatched pair may still merge when BOTH hold: the two groups are ones a
    # single player can plausibly be labelled with, and there is exactly ONE PFF player
    # of that name on that team that season. The uniqueness test is what keeps the
    # Anthony Garcia case dead - two players, one name, one team, one season - and the
    # adjacency list keeps it dead twice over, since QB and WR are not adjacent.
    ADJACENT = {frozenset(x) for x in [
        ("DL", "LB"),    # edge rusher: 3-4 outside backer or 4-3 end, same player
        ("LB", "DB"),    # nickel, rover, hybrid safety
        ("TE", "WR"),    # flex tight end split out wide
        ("RB", "TE"),    # H-back and fullback
        ("RB", "WR"),    # slot back
    ]}
    solo = p.dropna(subset=["grp"]).drop_duplicates(["season", "team", "key", "grp"])
    n_named = solo.groupby(["season", "team", "key"]).uid.transform("size")
    solo = (solo[n_named == 1][["season", "team", "key", "grp", "uid"]]
            .rename(columns={"uid": "alt_uid", "grp": "alt_grp"}))
    c = c.merge(solo, on=["season", "team", "key"], how="left")
    adj = [frozenset((a, b)) in ADJACENT
           for a, b in zip(c.grp.fillna(""), c.alt_grp.fillna(""))]
    take = c.pff_uid.isna() & c.alt_uid.notna() & pd.Series(adj, index=c.index)
    n_exact = int(c.pff_uid.notna().sum())
    c.loc[take, "pff_uid"] = c.loc[take, "alt_uid"]
    c = c.drop(columns=["alt_uid", "alt_grp"])

    c["uid"] = c.pff_uid.fillna("cfbd:" + c.player_id.astype(str))
    print(f"CFBD rows inheriting a PFF player id: {c.pff_uid.notna().mean()*100:.1f}% "
          f"({int(c.pff_uid.notna().sum())} of {len(c)})   "
          f"exact group {n_exact}, adjacent group {int(take.sum())}")

    cols = ["season", "uid", "player", "position", "team", "facet", "snaps", "z", "value", "source"]
    out = pd.concat([p[cols], c[cols]], ignore_index=True)

    # Empirical-Bayes shrinkage of z by snaps was tried here and removed. It made
    # every measure worse - team CV 0.843 -> 0.837, Massey r 0.672 -> 0.656, league
    # WAR no longer summing to wins above replacement - and it did not even fix what
    # it was aimed at: the backup share of QB WAR went UP, 13.6% -> 16.0%. Shrinking
    # z compresses the facet totals, sigma compresses with them, and f_contrib
    # (w * value / sigma) barely moves for backups while starters lose real value.
    # The volume term in value = z * snaps is already the right control for playing
    # time; a second one double-counts it.

    # The two sources label positions differently - PFF says CB/S/DI/ED/T/G/C where
    # CFBD says DB/DL/OL - so a matched player would otherwise split into two rows
    # under the same id and have his WAR divided between them. PFF's labels are the
    # finer-grained set and every downstream stage is keyed to them, so they win;
    # CFBD-only players keep their own label because nothing else is available.
    canon = (out[out.source == "pff"].drop_duplicates(["season", "uid"])
                .set_index(["season", "uid"]).position)
    lookup = pd.Series(
        canon.reindex(pd.MultiIndex.from_arrays([out.season, out.uid])).to_numpy(),
        index=out.index)
    out["position"] = lookup.fillna(out.position)

    # the display name should likewise be one string per player, not two spellings
    name = (out[out.source == "pff"].drop_duplicates(["season", "uid"])
               .set_index(["season", "uid"]).player)
    nlookup = pd.Series(
        name.reindex(pd.MultiIndex.from_arrays([out.season, out.uid])).to_numpy(),
        index=out.index)
    out["player"] = nlookup.fillna(out.player)
    # keep only teams with an FBS record, as the PFF-only build does
    recs = pd.read_csv(f"{HERE}/records.csv")
    keep = set(zip(recs.season, recs.team))
    out = out[[(s, t) in keep for s, t in zip(out.season, out.team)]]
    return out.reset_index(drop=True)


def snap_denominated_facets():
    """Facets whose denominator is a real snap count rather than a rate denominator.

    Detected from the denominator column name, so it stays correct if the catalogue is
    regenerated with new measurements: PFF names true snap counts *_snap_counts_* and
    pass__passing_snaps, while attempts / dropbacks / targets / routes / touches are
    all subsets of a player's snaps.
    """
    is_snaps = lambda d: bool(d) and ("snap_counts" in d or d.endswith("passing_snaps"))
    source = os.environ.get("WAR_FEATURES", "candidates").lower()
    if source == "candidates" and os.path.exists(f"{HERE}/candidate_catalogue.json"):
        cat = json.load(open(f"{HERE}/candidate_catalogue.json"))["catalogue"]
        return {c["name"] for c in cat if is_snaps(c.get("denominator"))}
    return {name for name, (_, snap_col, _, _) in FACETS.items() if is_snaps(snap_col)}


def deattenuate(ratings, fv, recs, slope):
    """Factor k such that k*WAA + replacement regresses on actual wins above
    replacement with slope 1 - i.e. the factor that puts WAR into units of wins.

    Solved rather than fitted in one pass: regressing the residual on WAA is itself an
    OLS on a noisy predictor and lands at 0.955, so the root of "slope of the whole
    identity equals 1" is taken directly.
    """
    from scipy.optimize import brentq
    prev = pd.read_csv(f"{HERE}/hybrid_player_war.csv") if os.path.exists(
        f"{HERE}/hybrid_player_war.csv") else None
    if prev is None:
        print("  [warn] no prior build to calibrate against; slope left attenuated.")
        return 1.0
    t = prev.groupby(["season", "team"], as_index=False).agg(
        war=("war", "sum"), waa=("waa", "sum"))
    t["repl"] = t.war - t.waa
    d = t.merge(recs[["season", "team", "adj_wins", "fbs_games"]],
                on=["season", "team"])
    d["actual"] = d.adj_wins - REPL_WIN_PCT * d.fbs_games
    f = lambda k: np.polyfit(k * d.waa + d.repl, d.actual, 1)[0] - 1.0
    try:
        k = float(brentq(f, 0.5, 5.0))
    except ValueError:
        print("  [warn] could not bracket the de-attenuation root; leaving slope as is.")
        return 1.0
    print(f"  de-attenuation solved on {len(d)} team-seasons: k = {k:.4f}")
    return k


def main():
    fv = unified_facets()
    facet_names = sorted(fv.facet.unique())
    print(f"hybrid facet rows: {len(fv)}   facets: {len(facet_names)}")

    recs = pd.read_csv(f"{HERE}/records.csv")
    sched = pd.read_csv(f"{HERE}/schedule.csv")

    tot = (fv.groupby(["season", "team", "facet"], as_index=False)["value"].sum()
             .pivot(index=["season", "team"], columns="facet", values="value")
             .reindex(columns=facet_names).fillna(0.0))
    Z = tot.groupby(level="season").transform(
        lambda c: (c - c.mean()) / c.std(ddof=0)).fillna(0.0)

    df = Z.join(recs.set_index(["season", "team"]), how="inner")
    X = df[facet_names].to_numpy(float)
    y = df.adj_win_pct.to_numpy(float)
    groups = df.index.get_level_values("season").to_numpy()

    # ---- facet weights ------------------------------------------------------
    # Fitted by ridge against the FOLLOWING season's wins, not by random-forest
    # importance against the same season's. Both parts of that changed for a reason
    # (weight_target_test.py):
    #
    #   same-season RF (was)  forward r .494   pass rush  7.8%   coverage 25.8%
    #   next-season ridge     forward r .534   pass rush 14.6%   coverage 12.9%
    #
    # Fitting on the same season loads the model onto facets contaminated by the
    # result. Coverage grade is the clearest case: it explains the season it happened
    # in better than pass rush does (r .55 vs .38) and repeats less than half as well
    # year over year (.21 vs .57), because a defence that is ahead forces obvious
    # passing downs and PFF rewards the coverage that follows. Weighting on that made
    # corners the most valuable non-quarterbacks in football and buried edge rushers,
    # which contradicts every published study of positional value - and it is also
    # what made the ratings echo the previous season.
    #
    # Ridge rather than forest importance because impurity importance inflates
    # whichever of a correlated pair gets split on first; ridge and permutation
    # importance both put pass rush above coverage, the forest alone does not.
    #
    # NON-NEGATIVITY IS THE DEFAULT, and it is the constraint that makes positional
    # value trustworthy. A negative weight says a player is penalised for being good
    # at a football skill, which is never what we mean. Unconstrained fits produce
    # them freely: PCA + ridge on these candidates gives 21 negative weights out of
    # 86, including contested-catch rate, pass break-ups and first downs. Those are
    # not findings, they are collinear features splitting signs. Non-negative ridge
    # solves the same penalised problem with w >= 0 via NNLS, costs .0006 RMSE against
    # the unconstrained fit, and produces:
    #
    #   QB 15.7%  EDGE 11.0%  WR 8.8%  CB 8.2%  RB 8.2%  LB 7.9%  OL 7.2%  DL 7.0%
    #
    # Edge rushers second only to quarterbacks and ahead of corners; receivers level
    # with backs. Both of those were wrong before and both are right now.
    #
    # NON-NEGATIVITY FIXED THE SIGNS AND LEFT THE SPLIT BROKEN. NNLS is a sparse
    # solver: faced with two near-duplicate facets it does not share the weight, it
    # sends one to exactly zero. That is what the 30 zeros above are. Bootstrapping
    # this fit over team-seasons shows the damage - the QB block total is stable at
    # 15.5% (5-95% band 11.1-19.1) while inside it QB_twp_rate takes anywhere from 7.6%
    # to 28.1%, QB_pass takes 0-12.6% and is zeroed outright in 16% of resamples, and
    # eight different facets hold the top QB slot at some point. Every block is like
    # this; 45 of 98 facets are zeroed in a quarter or more of fits.
    #
    # It does not hurt win prediction - the team totals come out the same either way -
    # but WAR is an ATTRIBUTION, so which twin won the coin flip decides who gets
    # credit. That is how turnover-worthy-play rate came to outweigh the passing grade
    # three to one, and how a tight end's receiving value came to rest on targeted QB
    # rating, a statistic about the quarterback throwing to him.
    #
    # TWOLEVEL IS THE DEFAULT NOW, and the trade it makes is deliberate. It fits
    # concept groups against wins and splits inside them by reliability rather than by
    # the fit (see two_level_weights.py). Measured against the flat fit by
    # weighting_compare.py:
    #
    #   forward r                        .523 -> .500      worse
    #   median rank sd over resamples     532 -> 254        half
    #   mean Spearman vs base ranking    .944 -> .985
    #   facets zeroed in >=25% of fits   45/98 -> 8/98
    #
    # Losing .023 of forward r to halve the resampling spread of the player rankings is
    # the right way round for this model. WAR is not a win predictor - the win
    # predictor is downstream of it, and cfb-model-v3 blends this in at 40% of one of
    # its six inputs. What WAR is FOR is attribution, and an attribution that moves
    # this much under a resample was not measuring the player.
    #
    # WAR_WEIGHTS: twolevel (default) | nonneg | next_season_ridge | same_season_rf | pca
    mode = os.environ.get("WAR_WEIGHTS", "twolevel")
    nxt = recs[["season", "team", "adj_win_pct"]].copy()
    nxt["season"] -= 1
    fwd = (df.reset_index()
             .merge(nxt.rename(columns={"adj_win_pct": "next_win_pct"}),
                    on=["season", "team"], how="inner"))
    Xf = fwd[facet_names].to_numpy(float)
    yf = fwd.next_win_pct.to_numpy(float)

    if mode == "same_season_rf":
        m = RandomForestRegressor(n_estimators=800, min_samples_leaf=3,
                                  random_state=0, n_jobs=-1).fit(X, y)
        w = pd.Series(m.feature_importances_, index=facet_names)
    elif mode == "pca":
        # PCA + ridge composes exactly back to per-facet weights, because every step
        # is linear: standardize, project, weight. Kept switchable, not default,
        # because of the sign problem described above.
        sc = StandardScaler().fit(Xf)
        pc = PCA(n_components=0.90, svd_solver="full").fit(sc.transform(Xf))
        rg = RidgeCV(alphas=np.logspace(-3, 4, 60)).fit(pc.transform(sc.transform(Xf)), yf)
        w = pd.Series((rg.coef_ @ pc.components_) / sc.scale_,
                      index=facet_names).clip(lower=0)
    elif mode == "twolevel":
        # facet_reliability.csv comes from uncertainty.py, which reads the parquet this
        # stage writes - so the first build on a fresh tree has to run once under
        # nonneg to seed it. It is not circular in substance: rho is computed from z
        # and snaps only and does not depend on the weights at all.
        import two_level_weights as tlw
        rel = tlw.load_reliability(fv, rebuild=True)
        w, tl = tlw.build(fwd.set_index(["season", "team"])[facet_names], yf, rel,
                          facets=facet_names)
        print(f"two-level: {tl['n_concepts']} concepts in {tl['n_groups']} groups, "
              f"lam={tl['lam']}   collinear and grouped: {tl['groups'] or 'none'}")
        for g, v in tl["group_weights"].items():
            print(f"    {g:<20} {v*100:5.1f}%")
    elif mode == "nonneg":
        # NNLS on [X; sqrt(lam) I] is the ridge normal equations with a non-negativity
        # constraint. lam from the season-blocked sweep in the notes above.
        lam = float(os.environ.get("WAR_NNLS_LAM", 1000.0))
        A = np.vstack([Xf, np.sqrt(lam) * np.eye(len(facet_names))])
        b = np.concatenate([yf - yf.mean(), np.zeros(len(facet_names))])
        coef, _ = nnls(A, b)
        w = pd.Series(coef, index=facet_names)
    else:
        m = RidgeCV(alphas=np.logspace(-2, 3, 40)).fit(Xf, yf)
        w = pd.Series(m.coef_, index=facet_names).clip(lower=0)
    n_zero = int((w == 0).sum())
    w = w / w.sum()
    print(f"facet weights: {mode}   {len(facet_names)} features, "
          f"{n_zero} zeroed, {int((w < 0).sum())} negative   "
          f"(fitted on {len(fwd) if mode != 'same_season_rf' else len(df)} team-seasons)")

    f_now = X @ w.reindex(facet_names).to_numpy()
    f_fwd = fwd[facet_names].to_numpy(float) @ w.reindex(facet_names).to_numpy()
    print(f"  weighted facet total vs this season's wins: "
          f"r = {np.corrcoef(f_now, y)[0,1]:.3f}")
    print(f"  weighted facet total vs NEXT season's wins: "
          f"r = {np.corrcoef(f_fwd, fwd.next_win_pct)[0,1]:.3f}")
    rf_cv = RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                  random_state=0, n_jobs=-1)
    p = cross_val_predict(rf_cv, X, y, cv=GroupKFold(n_splits=5), groups=groups)
    print(f"  season-blocked CV r (all facets -> adj win pct): "
          f"{np.corrcoef(p, y)[0,1]:.3f}")

    print("\ntop 12 facet weights:")
    for f, v in w.sort_values(ascending=False).head(12).items():
        print(f"  {f:<18} {v:.4f}")
    print(f"\n  CFBD share of total weight: "
          f"{w[[f for f in facet_names if f.startswith('cfbd_')]].sum()*100:.1f}%")

    # ---- Massey ratings on the hybrid f -----------------------------------
    sigma = tot.groupby(level="season").std(ddof=0)
    ratings = []
    for season in YEARS:
        zs = Z.xs(season, level="season")
        teams = sorted(t for t in zs.index
                       if ((recs.season == season) & (recs.team == t)).any())
        f = zs.loc[teams, facet_names].to_numpy(float) @ w.to_numpy()
        f = f - f.mean()
        M, _ = massey_matrix(sched, season, teams)
        A = M.copy(); A[-1, :] = 1.0
        b = f.copy(); b[-1] = 0.0
        ratings.append(pd.DataFrame({"season": season, "team": teams,
                                     "f": f, "massey": np.linalg.solve(A, b)}))
    ratings = pd.concat(ratings, ignore_index=True)
    out = ratings.merge(recs, on=["season", "team"])
    print(f"\nMassey rating vs adj win pct, all seasons: "
          f"r = {np.corrcoef(out.massey, out.adj_win_pct)[0,1]:.3f}")

    slope, intercept = np.polyfit(out.massey, out.adj_win_pct, 1)
    print(f"implied win pct = {slope:.4f} * massey + {intercept:.4f}  (OLS, attenuated)")

    # THAT SLOPE IS ATTENUATED, AND IT IS THE REASON WAR DID NOT EQUAL WINS.
    # Massey is an estimate with error in it, so an OLS fit of win pct on Massey is
    # biased toward zero - the textbook errors-in-variables result. Everything
    # downstream inherits the bias, because waa is linear in this slope, and the
    # symptom was that summed team WAR regressed on actual wins above replacement at
    # 1.308 rather than 1.000. Wins Above Replacement was not in units of wins.
    #
    # It was being papered over at DISPLAY time - export_viz multiplied each roster by
    # a hardcoded 1.64 to make the numbers "read as wins" - which was wrong twice. The
    # constant did not match the data (1.31 measured over 1,438 team-seasons), and a
    # slope-only expansion around the league mean drove the bottom of the distribution
    # through zero, so one roster displayed every player as a negative contributor.
    #
    # The replacement half was never the problem: league replacement credit comes to
    # 5,985.7 against 5,985.7 actual wins above replacement, 0.0% out. Only the spread
    # was short. So the correction is one number solved for here, not applied later:
    # the factor that makes summed team WAR regress on actual wins above replacement
    # with a slope of exactly 1. It comes out at 1.41, and afterwards the identity
    # holds at 1.0000 * WAR - 0.0023 with the correlation slightly BETTER than before
    # (.834 against .816). No display rescale is needed anywhere.
    slope *= deattenuate(out, fv, recs, slope)
    print(f"de-attenuated slope = {slope:.4f}")

    nxt = out[["season", "team", "massey", "adj_win_pct"]].copy()
    nxt["season"] -= 1
    j = out.merge(nxt, on=["season", "team"], suffixes=("", "_n1"))
    print(f"  Cor(rating, win pct year n+1): "
          f"{np.corrcoef(j.massey, j.adj_win_pct_n1)[0,1]:.3f}")

    # ---- WAA / WAR ---------------------------------------------------------
    sens, games = {}, {}
    for season in YEARS:
        teams = sorted(recs[recs.season == season].team)
        sens[season] = team_sensitivity(sched, season, teams)
        games[season] = recs[recs.season == season].set_index("team").fbs_games

    fv["sigma"] = [sigma.loc[s, f] for s, f in zip(fv.season, fv.facet)]
    fv["w"] = fv.facet.map(w)
    fv["f_contrib"] = fv.w * fv.value / fv.sigma
    fv["c_t"] = [sens[s][t] for s, t in zip(fv.season, fv.team)]
    fv["games"] = [games[s][t] for s, t in zip(fv.season, fv.team)]
    fv["waa"] = fv.games * slope * fv.c_t * fv.f_contrib

    # Replacement credit is payment for OCCUPYING A SNAP, so it can only be paid
    # through facets where the volume column is the player's own playing time. On the
    # CFBD defensive facets it is not: CFBD publishes no individual snap counts on
    # defense, so cfbd_facets.py denominates them by team defensive plays, and every
    # defender on the roster carries the same volume - 922 of them at Alabama in 2025,
    # for the fourteenth defensive back as much as for the starting corner. Paying
    # replacement credit per that snap column hands a 1-100 snap defender 67% of what
    # a 600+ snap defender collects, where playing time says it should be nearer 10%.
    #
    # The value term survives this and is left alone: z is production per team play,
    # so z * team_plays is total production above average, which is what it should be.
    # Only the per-snap credit is incoherent, so only it is withheld. The pool is
    # renormalized over the paying facets, which keeps league WAR summing to wins above
    # replacement instead of quietly losing their share of it.
    # Renormalizing per season rather than once also closes a leak that predates this:
    # a facet with no snaps in a season used to forfeit its share of that season's
    # pool instead of passing it on, so 2014 and 2015 - which have no CFBD data at all
    # - were quietly paying out 1.5% less replacement WAR than every other year. That
    # was survivable while 30 facets sat at zero weight; under two-level weights every
    # facet carries some, and every gap would leak.
    team_volume = {f"cfbd_{n}" for n, (_, vol, _) in CFBD_FACETS.items()
                   if vol == "team_def_plays"}
    paying = [f for f in facet_names if f not in team_volume]

    pool_rows = []
    for season in YEARS:
        r = recs[recs.season == season]
        pool = float(((0.5 - REPL_WIN_PCT) * r.fbs_games).sum())
        league_snaps = fv[fv.season == season].groupby("facet").snaps.sum()
        present = [f for f in paying if league_snaps.get(f, 0.0) > 0]
        w_present = float(w[present].sum()) or 1.0
        for facet in present:
            pool_rows.append({"season": season, "facet": facet,
                              "per_snap": pool * w[facet] / w_present
                                          / league_snaps[facet]})
    per_snap = pd.DataFrame(pool_rows).set_index(["season", "facet"]).per_snap
    fv["repl_credit"] = [per_snap.get((s, f), 0.0)
                         for s, f in zip(fv.season, fv.facet)] * fv.snaps
    fv["war"] = fv.waa + fv.repl_credit

    key = ["season", "uid", "player", "position", "team"]
    war = fv.groupby(key, as_index=False).agg(
        waa=("waa", "sum"), war=("war", "sum"))
    # Snaps are reported from PFF rows only. CFBD's volume column is a reconstructed
    # play count on a different denominator - targets rather than routes, team plays
    # rather than individual snaps - so adding the two together would produce a number
    # that is not a snap count of anything.
    #
    # AND THEY ARE A MAX, NOT A SUM. Summing was the obvious thing and it is wrong: the
    # facet-level column is each facet's own DENOMINATOR, and a facet's denominator is
    # whatever that measurement divides by - attempts, dropbacks, targets, routes,
    # touches, or a snap count. A quarterback is covered by thirteen facets, so summing
    # counted his season eight or nine times over and reported Julian Sayin with 4,454
    # snaps in 2025 against a real 715. It scaled with how many facets happen to cover a
    # position, which is a property of the catalogue and not of the player.
    #
    # The max over the facets that are denominated by an ACTUAL snap count is the
    # player's snap count, because every other denominator is a subset of his snaps.
    # 92% of player-seasons have at least one such facet; the rest fall back to the max
    # over all of them, which is a floor rather than a fiction.
    snapish = snap_denominated_facets()
    pff = fv[fv.source == "pff"]
    real = (pff[pff.facet.isin(snapish)].groupby(key, as_index=False).snaps.max())
    anyv = (pff.groupby(key, as_index=False).snaps.max()
               .rename(columns={"snaps": "snaps_any"}))
    war = war.merge(real, on=key, how="left").merge(anyv, on=key, how="left")
    war["snaps"] = war.snaps.fillna(war.snaps_any).fillna(0.0)
    war = war.drop(columns=["snaps_any"])
    # the downstream stages key on player_id
    war = war.rename(columns={"uid": "player_id"})

    # per-facet WAR, in the same shape build_war writes, so the analysis stages that
    # decompose a player's value can read either build
    facet_war = fv.pivot_table(index=key, columns="facet", values="war",
                               aggfunc="sum").fillna(0.0)
    facet_war.index = facet_war.index.rename({"uid": "player_id"})
    facet_war.to_csv(f"{HERE}/hybrid_player_war_by_facet.csv")
    fv.rename(columns={"uid": "player_id"}).to_parquet(f"{HERE}/hybrid_facet_war.parquet")

    war.to_csv(f"{HERE}/hybrid_player_war.csv", index=False)
    out.to_csv(f"{HERE}/hybrid_team_ratings.csv", index=False)
    w.rename("rf").to_frame().to_csv(f"{HERE}/hybrid_facet_weights.csv")
    tot.to_csv(f"{HERE}/hybrid_team_facet_totals.csv")
    json.dump({"slope": float(slope), "intercept": float(intercept)},
              open(f"{HERE}/hybrid_wins_map.json", "w"), indent=1)

    print(f"\nplayer-seasons: {len(war)}   total WAR: {war.war.sum():.0f}")
    print("\ntop 12 by WAR, 2025:")
    print(war[war.season == 2025].nlargest(12, "war")[
        ["player", "position", "team", "waa", "war"]].to_string(index=False))
    print("\nhybrid artifacts written")


if __name__ == "__main__":
    main()
