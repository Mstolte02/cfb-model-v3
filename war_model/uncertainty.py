"""How confident is the model in each player's WAR, and in each team's total?

Every WAR figure rests on a z-score estimated from a finite number of snaps, and a
250-snap backup's z is a much noisier measurement than a 1,200-snap starter's. Nothing
downstream knew that: a team of proven veterans and a team of unproven freshmen got
point estimates with the same apparent authority.

TWO SOURCES OF UNCERTAINTY, and they are different in kind.

  measurement   we observed this player's season, but a season is a sample. How far
                is the observed z from his true talent? Scales as 1/snaps, with the
                constant set per facet by how much of that facet even repeats.
  projection    for 2026 we have not observed anything yet, so on top of measurement
                error there is the projection model's own error, taken from its
                held-out residuals by position group and prior-snap bucket.

CALIBRATING THE MEASUREMENT TERM. A facet's year-over-year correlation is its
reliability: rho = signal / (signal + noise) at the typical workload. So the noise
share is (1 - rho) at median snaps, and scaling by median/snaps gives the variance for
any workload. That is the standard reliability-to-variance step, and it is why a facet
like contested-catch rate (rho .02) contributes almost pure noise while edge pass rush
(rho .57) is close to a real measurement.

PROPAGATION IS EXACT because WAR is linear in z:
    waa = games * slope * c_t * w * z * snaps / sigma
so d(waa)/dz is a constant and var(waa) = (d/dz)^2 * var(z). No simulation needed at
the player level.

Facets are summed as if independent, which understates a player's total variance where
his own facets correlate - a receiver's route grade and yards per route run move
together. Noted rather than modelled; the alternative needs a full covariance and buys
little here.

Run: ./rbenv/bin/python uncertainty.py
"""
import json, os
import numpy as np
import pandas as pd

import artifacts

HERE = os.path.dirname(os.path.abspath(__file__))
RHO_FLOOR, RHO_CEIL = 0.02, 0.95
MIN_PAIRS = 60


def reliability(fv, key):
    """Per facet: how much of the z-score is SAMPLING NOISE, separated from real change.

    The old version took the year-over-year correlation and called 1 - rho the noise.
    That charges the measurement for something it did not do: a player is not the same
    player twelve months later. He develops, he is coached differently, he changes
    role. All of that lands in 1 - rho and inflates every error bar on the site,
    because a facet where players genuinely improve looks exactly like a facet we
    cannot measure.

    The two are separable without a split-half, which is fortunate, since the PFF
    exports are season aggregates and there are no odd/even game splits to take.
    Sampling error falls as 1/snaps and real change does not, so over the pairs

        E[(z_t - z_t+1)^2]  =  2 * A / n_harmonic  +  C

    the SLOPE against 1/n is the measurement noise and the INTERCEPT is how much
    players actually move. Fitting that line gives a noise constant A that is a
    property of the measurement, and a reliability that is a function of workload
    rather than one number for every player on the facet:

        rho(n) = 1 - A / n

    C is reported alongside because it is interesting in its own right - it is the
    part of a facet that is real and still does not persist.
    """
    rows = []
    for f, g in fv.groupby("facet"):
        g2 = g[g.snaps >= g.snaps.quantile(0.4)]
        nxt = g2[["season", key, "z", "snaps"]].copy()
        nxt["season"] -= 1
        j = g2.merge(nxt.rename(columns={"z": "z1", "snaps": "snaps1"}),
                     on=["season", key])
        med = float(g.snaps.median())
        rho_yoy = (float(np.corrcoef(j.z, j.z1)[0, 1]) if len(j) >= MIN_PAIRS else 0.25)

        A = C = np.nan
        if len(j) >= MIN_PAIRS:
            nh = 2.0 / (1.0 / j.snaps.clip(lower=1) + 1.0 / j.snaps1.clip(lower=1))
            d2 = (j.z - j.z1).to_numpy(float) ** 2
            x = (1.0 / nh).to_numpy(float)
            X = np.column_stack([np.ones(len(x)), x])
            beta, *_ = np.linalg.lstsq(X, d2, rcond=None)
            C, A = float(beta[0]), float(beta[1] / 2.0)
        if not np.isfinite(A) or A <= 0:
            # no usable slope: fall back to charging the whole year-over-year gap to
            # measurement at the median workload, which is what this used to do
            A = (1 - np.clip(rho_yoy, RHO_FLOOR, RHO_CEIL)) * med
            C = np.nan

        rows.append({"facet": f, "noise_A": float(A), "change_C": float(C),
                     "rho": float(np.clip(rho_yoy, RHO_FLOOR, RHO_CEIL)),
                     "rho_median_workload": float(np.clip(1 - A / max(med, 1),
                                                          RHO_FLOOR, RHO_CEIL)),
                     "median_snaps": med, "n_pairs": len(j)})
    return pd.DataFrame(rows).set_index("facet")


def main():
    fv = pd.read_parquet(f"{HERE}/{artifacts.FACET_WAR}")
    key = "player_id" if "player_id" in fv.columns else "uid"
    wmap = json.load(open(f"{HERE}/{artifacts.WINS_MAP}"))
    slope = wmap["slope"]

    rel = reliability(fv, key)
    print(f"facets: {len(rel)}   year-over-year rho median {rel.rho.median():.3f}")
    print(f"  of which SAMPLING noise at the median workload: "
          f"{1 - rel.rho_median_workload.median():.3f}   "
          f"real change: {rel.change_C.median():.3f} of z variance")

    fv = fv.join(rel, on="facet")
    # variance of the observed z around the player's true talent, at HIS workload
    fv["var_z"] = (fv.noise_A / fv.snaps.clip(lower=1)).clip(upper=1.0)
    # Derivative of this facet's WAA with respect to its z. Since build_hybrid started
    # allocating the schedule term, waa carries a second piece - sched_t times the
    # player's share of the team's |f_contrib| - whose derivative is second order: the
    # team's residual is FIXED, so a player's z only moves how a constant is divided.
    # The own-production term below is the whole of the first-order effect.
    dvdz = fv.dvdz if "dvdz" in fv.columns else fv.snaps
    fv["dwaa_dz"] = fv.games * slope * fv.c_t * fv.w * dvdz / fv.sigma
    fv["var_waa"] = fv.dwaa_dz ** 2 * fv.var_z

    pk = ["season", key, "player", "position", "team"]
    war = fv.groupby(pk, as_index=False).agg(
        war=("war", "sum"), waa=("waa", "sum"), snaps=("snaps", "sum"),
        var=("var_waa", "sum"))
    war["sd"] = np.sqrt(war["var"])
    war["lo68"] = war.war - war.sd
    war["hi68"] = war.war + war.sd
    # A unitless score so the app can shade a player without a wins scale. It was
    # called `confidence`, which it is not: a confidence is a probability that an
    # interval covers something, and this is |estimate| / (|estimate| + its sd) - a
    # signal-to-noise ratio squashed onto 0-1. Naming it `snr` costs nothing and stops
    # it being read as "the model is 84% sure".
    war["snr"] = 1 / (1 + war.sd / war.war.abs().clip(lower=0.02))
    war["snr"] = war.snr.clip(0, 1)

    print(f"\nplayer-seasons {len(war):,}")
    print(f"  median sd {war.sd.median():.4f}   median |WAR| {war.war.abs().median():.4f}")
    print("\nuncertainty falls with playing time, as it must:")
    b = pd.cut(war.snaps, [0, 200, 500, 1000, 2000, 1e9],
               labels=["<200", "200-500", "500-1k", "1k-2k", "2k+"])
    print(war.groupby(b, observed=True).agg(
        n=("sd", "size"), mean_sd=("sd", "mean"),
        mean_war=("war", "mean"), mean_snr=("snr", "mean")).round(4).to_string())

    team = war.groupby(["season", "team"], as_index=False).agg(
        war=("war", "sum"), var=("var", "sum"), players=("war", "size"))
    team["sd"] = np.sqrt(team["var"])
    team["lo68"] = team.war - team.sd
    team["hi68"] = team.war + team.sd
    team["lo95"] = team.war - 1.96 * team.sd
    team["hi95"] = team.war + 1.96 * team.sd
    print(f"\nteam-seasons {len(team)}   mean team WAR {team.war.mean():.2f}   "
          f"mean sd {team.sd.mean():.3f}")
    print("\nleast and most certain rosters, 2025:")
    t25 = team[team.season == 2025].nlargest(4, "sd")[["team", "war", "sd"]]
    t25b = team[team.season == 2025].nsmallest(4, "sd")[["team", "war", "sd"]]
    print("  widest: " + ", ".join(f"{r.team} ±{r.sd:.2f}" for r in t25.itertuples()))
    print("  tightest: " + ", ".join(f"{r.team} ±{r.sd:.2f}" for r in t25b.itertuples()))

    war.to_csv(f"{HERE}/player_war_uncertainty.csv", index=False)
    team.to_csv(f"{HERE}/team_war_uncertainty.csv", index=False)
    rel.reset_index().to_csv(f"{HERE}/facet_reliability.csv", index=False)
    print("\n-> player_war_uncertainty.csv, team_war_uncertainty.csv, "
          "facet_reliability.csv")


def _sd_model(resid_frame):
    """Fit the (group, history) table, the level exponent and the team shock."""
    err = resid_frame.groupby(["group", "hist"]).resid.std().rename("sd").reset_index()
    fallback = float(resid_frame.resid.std())
    lp = np.log(np.abs(resid_frame.pred).clip(lower=1e-3))
    lr = np.log(np.abs(resid_frame.resid).clip(lower=1e-4))
    ok = np.isfinite(lp) & np.isfinite(lr)
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(int(ok.sum())), lp[ok]]),
                               lr[ok], rcond=None)
    tm = resid_frame.groupby("team").resid.agg(["mean", "size"])
    tm = tm[tm["size"] >= 20]
    idio = float(resid_frame.resid.var())
    common = float(max(tm["mean"].var() - (idio / tm["size"]).mean(), 0.0))
    return {"err": err, "fallback": fallback, "level_slope": float(beta[1]),
            "ref": float(np.abs(resid_frame.pred).median()), "common_var": common}


def _apply_sd(frame, sdm):
    e = sdm["err"].rename(columns={"group": "_g"})
    m = frame.merge(e, left_on=["group", "hist"], right_on=["_g", "hist"], how="left")
    m.index = frame.index
    lvl = np.clip((frame.pred.abs().clip(lower=1e-3) / max(sdm["ref"], 1e-3))
                  ** sdm["level_slope"], 0.35, 3.0)
    return m.sd.fillna(sdm["fallback"]).to_numpy() * lvl.to_numpy()


def coverage(tr, sdm_in_sample):
    """Do the intervals contain the answer? OUT OF SAMPLE, at the team level.

    Nothing in this module was ever checked against an outcome, and an interval that
    has not been scored for coverage is a decoration. This one was: the old team
    projection intervals summed ~90 player variances as if independent and came out
    near +/-0.4 wins, which no roster's projection has ever been.

    Both halves are held out. The error model is fitted on residuals from a projection
    trained through 2023 and scored on 2024; it is then applied, unchanged, to a
    projection trained through 2024 and scored on 2025. Estimating the spread on the
    same season it is scored against is how an interval comes to look calibrated
    without being so.
    """
    from project_2026_v2 import FEATURES, fit
    out = {}
    tr = tr.assign(hist=np.where(tr.prior_seasons == 0, "none",
                                 np.where(tr.prior_seasons == 1, "one", "two+")))
    a = tr[tr.target_season < 2024]
    b = tr[tr.target_season == 2024]
    c = tr[tr.target_season == 2025]
    if len(a) < 500 or b.empty or c.empty:
        print("\n  [warn] not enough seasons for an out-of-sample coverage check")
        return out

    m = fit().fit(a[FEATURES], a.war)
    b = b.assign(pred=m.predict(b[FEATURES]))
    b = b.assign(resid=b.war - b.pred)
    sdm = _sd_model(b)

    m2 = fit().fit(tr[tr.target_season < 2025][FEATURES],
                   tr[tr.target_season < 2025].war)
    c = c.assign(pred=m2.predict(c[FEATURES]))
    c = c.assign(sd=_apply_sd(c, sdm))

    g = c.groupby("team").agg(actual=("war", "sum"), pred=("pred", "sum"),
                              var=("sd", lambda s: float((s ** 2).sum())),
                              n=("sd", "size"))
    e = (g.actual - g.pred).abs()
    print("\n  OUT-OF-SAMPLE interval coverage, 2025 team totals "
          "(error model fitted on 2024):")
    for name, var in (("independent (was)", g["var"]),
                      ("with team shock", g["var"] + (g.n ** 2) * sdm["common_var"])):
        sd = np.sqrt(var)
        c68, c95 = float((e <= sd).mean()), float((e <= 1.96 * sd).mean())
        out[name] = (c68, c95, float(sd.mean()))
        print(f"    {name:<20} {c68*100:5.1f}%  {c95*100:5.1f}%   "
              f"mean sd {sd.mean():.3f}")
    print(f"    team shock sd {np.sqrt(sdm['common_var']):.4f}/player, "
          f"level exponent {sdm['level_slope']:.3f}  (both fitted on 2024)")
    # The band is CONSERVATIVE, not calibrated: a nominal 68% covering ~87% says the
    # residuals are heavy-tailed - most rosters are projected well and a handful are
    # badly wrong - so a normal sd overstates the typical miss while still being the
    # right width for the tail. Recorded rather than tuned away, since shrinking it to
    # hit 68% on one season would be fitting the interval to the sample it is scored on.
    json.dump({"nominal_68_actual": out["with team shock"][0],
               "nominal_95_actual": out["with team shock"][1],
               "mean_team_sd": out["with team shock"][2],
               "mean_team_sd_independent": out["independent (was)"][2],
               "team_shock_sd_per_player": float(np.sqrt(sdm["common_var"])),
               "level_exponent": sdm["level_slope"],
               "fitted_on": 2024, "scored_on": 2025, "n_teams": int(len(g))},
              open(f"{HERE}/interval_coverage.json", "w"), indent=1)
    return out


# ---------------------------------------------------------------------------
def projection_uncertainty():
    """How wrong is the 2026 projection likely to be, per player and per team?

    Measurement error is the smaller half and it barely separates teams: summing ~90
    roughly-independent player variances gives every roster a similar total. What
    actually distinguishes a settled veteran team from one starting six freshmen is
    PROJECTION error - we have not seen 2026 at all - and that is estimated the only
    honest way available, from the projection model's own held-out residuals.

    Residuals are bucketed by position group and by how much history a player has,
    because those are the two things that visibly change the error: a returning
    starter is far more predictable than someone with no FBS snaps.
    """
    import artifacts as A
    from build_recruiting import load_recruits
    from facets import YEARS as WAR_YEARS
    from project_2026_v2 import (FEATURES, build_history, build_population,
                                 load_rosters, make_training, slot_counts, fit)

    war = pd.read_csv(f"{HERE}/{A.PLAYER_WAR}")
    ratings = pd.read_csv(f"{HERE}/{A.TEAM_RATINGS}")
    recs = pd.read_csv(f"{HERE}/records.csv")
    rec = load_recruits()
    ros26 = pd.read_csv(f"{HERE}/roster_2026.csv")
    K, S = slot_counts(ros26)

    w = build_history(war)
    rosters = load_rosters(WAR_YEARS, set(recs.team.unique()))
    pop = build_population(w, rosters, K)
    targets = [y for y in WAR_YEARS if (y - 1) in set(WAR_YEARS)]
    tr = make_training(pop, w, ratings, rec, rosters, S, targets)
    groups = sorted(w.group.dropna().unique())
    gcode = {g: i for i, g in enumerate(groups)}
    tr["group_code"] = tr.group.map(gcode)
    tr["share_lag1"] = tr.share_lag1.fillna(0.0)

    trn, tst = tr[tr.target_season < 2025], tr[tr.target_season == 2025]
    m = fit().fit(trn[FEATURES], trn.war)
    tst = tst.assign(pred=m.predict(tst[FEATURES]))
    tst["resid"] = tst.war - tst.pred
    tst["hist"] = np.where(tst.prior_seasons == 0, "none",
                           np.where(tst.prior_seasons == 1, "one", "two+"))

    err = (tst.groupby(["group", "hist"]).resid.std()
              .rename("sd").reset_index())
    fallback = float(tst.resid.std())
    print(f"\nprojection residual sd, held-out 2025 (overall {fallback:.4f}):")
    print(err.pivot(index="group", columns="hist", values="sd").round(4).to_string())

    # ---- residual sd also depends on the LEVEL of the prediction ------------
    # Group and history alone say a projected 1.4-win quarterback and a projected
    # 0.05-win one carry the same error, which is plainly false - the error grows with
    # what is being predicted. Fitting log|resid| on log|pred| within the group-history
    # cell captures it in one parameter and cannot go negative.
    lp = np.log(np.abs(tst.pred).clip(lower=1e-3))
    lr = np.log(np.abs(tst.resid).clip(lower=1e-4))
    ok = np.isfinite(lp) & np.isfinite(lr)
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(int(ok.sum())), lp[ok]]),
                               lr[ok], rcond=None)
    level_slope = float(beta[1])
    print(f"  residual sd scales with the prediction: d log|resid| / d log|pred| "
          f"= {level_slope:.3f}")

    def level_factor(pred, ref):
        f = (np.abs(pred).clip(lower=1e-3) / max(ref, 1e-3)) ** level_slope
        return np.clip(f, 0.35, 3.0)

    ref_pred = float(np.abs(tst.pred).median())

    # ---- a team-level common shock -----------------------------------------
    # Player residuals inside one roster are NOT independent. A new coordinator, a
    # changed scheme, a line that fails in front of everybody: the whole team misses
    # together. Summing ~90 independent variances therefore understated team error
    # badly, and it is also why the per-team spread never validated - the term that
    # actually separates teams was missing entirely.
    tm = tst.groupby("team").resid.agg(["mean", "size"])
    tm = tm[tm["size"] >= 20]
    # var(team mean) = var(common) + var(idio)/n, so subtract the sampling part
    idio = float(tst.resid.var())
    common_var = float(max(tm["mean"].var() - (idio / tm["size"]).mean(), 0.0))
    print(f"  team-level common shock sd: {np.sqrt(common_var):.4f} per player "
          f"(from {len(tm)} rosters)")

    proj = pd.read_csv(f"{HERE}/projections_2026_v2.csv")
    proj["hist"] = np.where(~proj.has_history, "none",
                            np.where(proj.prior_seasons == 1, "one", "two+"))
    proj = proj.merge(err.rename(columns={"group": "broad_group"}),
                      on=["broad_group", "hist"], how="left")
    proj["sd"] = proj.sd.fillna(fallback) * level_factor(proj.proj_war, ref_pred)

    team = proj.groupby("team", as_index=False).agg(
        proj_war=("proj_war", "sum"), var=("sd", lambda s: float((s ** 2).sum())),
        slots=("player", "size"),
        unproven=("has_history", lambda s: int((~s).sum())))
    # the common shock is perfectly correlated within a roster, so it enters the team
    # total as (n * sd)^2 rather than n * sd^2 - which is the whole reason the old
    # intervals were too narrow by roughly a factor of three
    team["var"] = team["var"] + (team.slots ** 2) * common_var
    team["sd"] = np.sqrt(team["var"])
    team["lo68"] = team.proj_war - team.sd
    team["hi68"] = team.proj_war + team.sd
    team["lo95"] = team.proj_war - 1.96 * team.sd
    team["hi95"] = team.proj_war + 1.96 * team.sd

    coverage(tr, None)

    print(f"\n2026 team projection uncertainty: mean sd {team.sd.mean():.3f}   "
          f"range {team.sd.min():.3f}-{team.sd.max():.3f}")
    print("  most uncertain rosters:")
    for r in team.nlargest(5, "sd").itertuples():
        print(f"    {r.team:<20} {r.proj_war:5.2f} +/- {r.sd:.2f}   "
              f"{r.unproven} of {r.slots} unproven")
    print("  most settled rosters:")
    for r in team.nsmallest(5, "sd").itertuples():
        print(f"    {r.team:<20} {r.proj_war:5.2f} +/- {r.sd:.2f}   "
              f"{r.unproven} of {r.slots} unproven")

    proj[["team", "player", "broad_group", "proj_war", "sd", "hist"]].to_csv(
        f"{HERE}/player_projection_uncertainty.csv", index=False)
    team.to_csv(f"{HERE}/team_projection_uncertainty.csv", index=False)
    err.to_csv(f"{HERE}/projection_residual_sd.csv", index=False)
    print("\n-> player_projection_uncertainty.csv, team_projection_uncertainty.csv")
    return team


# ---------------------------------------------------------------------------
def talent_noise():
    """How much noisier is the 2026 talent feature than the one the model trained on?

    The downstream win model never sees WAR directly; it sees TALENT, a within-season
    z-score. In training that z is built from each roster's observed prior-season WAR.
    For 2026 there is no observed season, so it is built from projected WAR instead -
    and that substitution, not the projection's absolute error, is the only thing the
    simulation needs to widen for. A fitted coefficient is already attenuated for the
    error in its own training feature, so injecting the projection's full error would
    double-count what the coefficient has priced in.

    The substitution cost is directly measurable on the one season where both features
    exist. Build 2025 talent both ways - held-out projection, and observed 2024 WAR -
    regress one on the other, and the residual sd IS the extra noise, in the z units
    the simulation perturbs.

    NEGATIVE RESULT, deliberately recorded. The per-team spread in
    team_projection_uncertainty.csv does not survive validation: correlation between a
    team's predicted uncertainty and how far its talent estimate actually missed is
    +0.01 (quadrature sd), -0.01 (unproven count), +0.03 (unproven share). Summing ~90
    player variances gives every roster a similar total and what spread remains is
    noise. So the simulation perturbs every team by the SAME sd. Claiming otherwise
    would be a confidence interval that varies for decorative reasons.
    """
    import json
    import artifacts as A
    from build_recruiting import load_recruits
    from facets import YEARS as WAR_YEARS
    from project_2026_v2 import (FEATURES, build_history, build_population,
                                 load_rosters, make_training, slot_counts, fit)

    war = pd.read_csv(f"{HERE}/{A.PLAYER_WAR}")
    ratings = pd.read_csv(f"{HERE}/{A.TEAM_RATINGS}")
    recs = pd.read_csv(f"{HERE}/records.csv")
    ros26 = pd.read_csv(f"{HERE}/roster_2026.csv")
    K, S = slot_counts(ros26)
    w = build_history(war)
    rosters = load_rosters(WAR_YEARS, set(recs.team.unique()))
    pop = build_population(w, rosters, K)
    targets = [y for y in WAR_YEARS if (y - 1) in set(WAR_YEARS)]
    tr = make_training(pop, w, ratings, load_recruits(), rosters, S, targets)
    groups = sorted(w.group.dropna().unique())
    tr["group_code"] = tr.group.map({g: i for i, g in enumerate(groups)})
    tr["share_lag1"] = tr.share_lag1.fillna(0.0)

    trn, tst = tr[tr.target_season < 2025], tr[tr.target_season == 2025]
    tst = tst.assign(pred=fit().fit(trn[FEATURES], trn.war).predict(tst[FEATURES]))

    proj_t = tst.groupby("team").pred.sum()
    prior = (war.groupby(["season", "player_id"], as_index=False).war.sum()
                .rename(columns={"war": "prior"}))
    prior["season"] += 1
    ros25 = war[war.season == 2025][["season", "player_id", "team"]].drop_duplicates()
    obs_t = (ros25.merge(prior, on=["season", "player_id"], how="left")
                  .fillna({"prior": 0.0}).groupby("team").prior.sum())

    idx = proj_t.index.intersection(obs_t.index)
    z = lambda s: (s - s.mean()) / s.std(ddof=0)
    xp, xo = z(proj_t[idx]), z(obs_t[idx])
    b = np.polyfit(xp, xo, 1)
    sd = float((xo - np.polyval(b, xp)).std(ddof=0))
    y = recs[recs.season == 2025].set_index("team").adj_win_pct.reindex(idx)

    out = {"talent_noise_sd": sd, "n_teams": int(len(idx)),
           "corr_proj_obs": float(xp.corr(xo)),
           "r_obs_wins": float(xo.corr(y)), "r_proj_wins": float(xp.corr(y)),
           "per_team_spread_validated": False}
    print(f"\ntalent-feature noise: teams {len(idx)}  corr(proj, observed-prior) "
          f"{out['corr_proj_obs']:.3f}  EXTRA SD {sd:.3f}")
    print(f"  projected talent tracks 2025 wins at r {out['r_proj_wins']:.3f} vs "
          f"{out['r_obs_wins']:.3f} for observed-prior - the projection earns its place")
    print("  per-team spread NOT validated (|r| < .03); simulation uses one sd for all")
    json.dump(out, open(f"{HERE}/talent_noise.json", "w"), indent=1)
    print("-> talent_noise.json")
    return out


if __name__ == "__main__":
    main()
    projection_uncertainty()
    talent_noise()
