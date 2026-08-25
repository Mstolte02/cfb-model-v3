"""Central configuration for the improved CFB predictive model.

Design choice that fixes the leakage in the original notebooks: the model is
FORWARD-LOOKING. A game in season N is predicted from each team's season (N-1)
advanced stats. This (a) matches the doc's real use case ("2024 stats -> 2025
projection") and (b) is genuinely leakage-free, because the features for a game
never include that game's own outcome.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"

# --- External data, none of which is in git ----------------------------------
# The PFF exports are licensed and the two-deep workbook is bought; both live
# outside the repo. They used to be absolute paths written into five different
# modules, which meant a machine without them got a DIFFERENT MODEL rather than an
# error - `except Exception: print(warn)` around each loader let the pipeline finish
# with the WAR talent silently missing from the blend. A run that cannot see its
# inputs must fail, not quietly produce a plausible number on fewer of them.
#
# Override any of these with the matching environment variable.
_SIBLING_EXTERNAL = ROOT.parent / "source-data"
_DEFAULT_EXTERNAL = (_SIBLING_EXTERNAL if _SIBLING_EXTERNAL.exists()
                     else Path.home() / "Downloads")
CFB_EXTERNAL = Path(os.environ.get("CFB_EXTERNAL", _DEFAULT_EXTERNAL))
PFF_DIR = Path(os.environ.get("PFF_DIR", CFB_EXTERNAL / "pff_exports"))
GAMES_CSV = Path(os.environ.get(
    "CFB_GAMES_CSV", CFB_EXTERNAL / "CFB_Data" / "data" / "games.csv"))
TWODEEP_2026 = Path(os.environ.get(
    "CFB_TWODEEP_2026",
    CFB_EXTERNAL / "fbs_2026_two_deep_pfsn_full_position_weights.xlsx"))
PFSN_MASTER = Path(os.environ.get(
    "CFB_PFSN_MASTER", CFB_EXTERNAL / "cfb_pfsn_all_raw_numbers_master.xlsx"))
LOGO_DIR = Path(os.environ.get("CFB_LOGO_DIR", CFB_EXTERNAL / "cfb_logos"))


# THE SITE SHIPS ONE BUILD. It used to ship several - a PFF-only one beside the
# default, each a complete build written with its own `_suffix` so the header could
# switch between them - and with it went SITE_VARIANTS, variant_suffix() and the
# CFB_WAR_VARIANT guard that stopped a mismatched run producing a plausible
# ratings_pff.json out of the wrong numbers. There is nothing left to keep in step.
# ROSTER_VARIANT below is unrelated: it is a set of hyperparameters, not a build.


def require(path, what, env):
    """Return `path`, or explain exactly what is missing and which variable moves it.

    Every external input goes through here. The alternative - a truthiness check at
    the call site that falls through to a default - is what produced runs whose
    printed accuracy described a model with one of its three talent inputs absent.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{what} not found at {p}. Set {env} to its location. "
            f"This input is not optional; the model is not the same model without it.")
    return p

# Season we are projecting (the whole point of this build).
PROJECTION_YEAR = 2026

# To predict season N we need stats from N-1. So pull stats for these years...
STAT_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
# ...and train on games from these (each uses the PRIOR year's stats).
GAME_YEARS = [2021, 2022, 2023, 2024, 2025]
# Hold out the most recent completed season for honest out-of-sample evaluation.
TEST_GAME_YEAR = 2025

# --- Talent layer (doc §B/§C: regression to a talent-informed mean) ----------
# Talent rating for year N (recruiting composite, known preseason -> no leakage)
# anchors where each team's prior-year stats get shrunk toward.
TALENT_YEARS = [2021, 2022, 2023, 2024, 2025]
# Shrinkage weight lambda: 0 = ignore talent (raw prior stats), 1 = pure talent.
# Tuned on held-out data in scripts/compare.py; this is the default for rank.py.
SHRINKAGE_LAMBDA = 0.65  # best on held-out 2025 (see scripts/compare.py)
# 2026 talent isn't published yet; fall back to this year's talent as a proxy.
PROJECTION_TALENT_FALLBACK_YEAR = 2025

# --- Calibrated mean (doc §C, improved) --------------------------------------
# Instead of regressing stats toward raw talent, regress toward a fitted
# preseason projection of each stat from: prior-year stat, talent, returning
# production, and prior-year Pythagorean win expectation.
RETURNING_YEARS = [2021, 2022, 2023, 2024, 2025]
PROJECTION_RETURNING_FALLBACK_YEAR = 2025
# Pythagorean exponent for win expectation = PF^x / (PF^x + PA^x).
# ~2.37 is the standard football value (Football Outsiders); tunable.
PYTHAG_EXP = 2.37
# Years we need game scores for, to compute Pythagorean (prior-season) inputs.
PYTHAG_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

# --- Regression of the O/D composites toward the roster baseline (doc §C) -----
# O_adj = (1 - lam*u)*O + lam*u*(b_o * talent_z), and likewise for D. lam = 0 trusts
# last season completely; lam = 1 sends a team all the way to its roster baseline.
#
# u USED TO BE (1 - returning production) - shrink the low-continuity teams harder.
# It is now FLAT (see UNCERTAINTY_USE_RETURNING), so lam IS the shrinkage.
#
# Both changes came out of scripts/rating_vs_realized.py, which scores the thing
# being complained about rather than a proxy for it. Every previous sweep of this
# knob (loso3, roster_vs_results) scored it on game outcomes, and that surface is
# FLAT - the whole lambda grid spans 0.0010 Brier, because a game's win probability
# is dominated by the talent coefficient and barely notices weight moving between
# the O/D composites and the roster baseline underneath them.
#
# The complaint was never about win probability. It was that a team can carry a
# top-10 DEFENCE RATING off one strong season while its roster says average. So
# score the rating: correlate each team's ENTERING-season O/D against what that unit
# actually PRODUCED that season, both opponent-adjusted. That target discriminates.
#
#   effective shrink   0.00    0.11*    0.35    0.55    0.65    0.70    0.75    0.90
#   defence r         .5531   .5688    .6010   .6270   .6346   .6356   .6339   .6054
#   offence r         .5473   .5682    .6044   .6361   .6478   .6515   .6532   .6402
#   * what shipped before: lam 0.25 x mean u 0.451
#
# 0.70 is defence's optimum by correlation and offence's by RMSE, and sits within
# 0.002 r of both sides' peaks - the two sides were checked separately and do not
# want different values, so splitting them would be fitting noise. Stable: all five
# seasons improve monotonically off zero and each season's own optimum is in
# [0.65, 0.85].
#
# It costs the win model nothing: LOSO Brier 0.2036 at lam 0.65-0.75 against 0.2034
# before, inside the noise this surface has always had. Only lam = 1.0 breaks it
# (0.2112), where O/D collapse onto pure talent and go collinear with the talent
# feature - so 0.70 is a genuine interior optimum on BOTH targets, not a corner.
UNCERTAINTY_LAMBDA = 0.70

# Returning production does not identify which teams regress. At MATCHED average
# shrinkage, weighting u by (1 - returning production) is worse than not weighting it
# at all - defence r 0.597 against 0.615 - so the variation it adds is noise. Set
# True to restore the old behaviour; src/matchup.uncertainty_u is the one definition.
UNCERTAINTY_USE_RETURNING = False

# --- Opponent (strength-of-schedule) adjustment (doc §4.4 / "Adj_" stats) -----
# Iterative SRS on the O/D composites. 0 = raw composites. See scripts/loso_oppadj.py.
#
# 1.0 was the EDGE of the tuning grid, which is not evidence that it is the optimum.
# Extended past it (solving the fixed point exactly, since alpha = 1 is where the
# iteration's singularity sits and alpha > 1 is otherwise ill-defined):
#
#   alpha    0.00   0.50   0.75   1.00   1.25   1.50   2.00   3.00
#   Brier   .2118  .2106  .2096  .2086  .2208  .2248  .2202  .2257
#
# A genuine interior optimum - it falls off a cliff past 1.0, not a plateau. The
# 25-iteration loop also never formally converges (the mean drifts ~0.006/iteration
# along the constant vector forever) but the SHAPE does, and the final
# re-standardization removes the drift: 25 vs 400 iterations correlates 1.00000 with
# a max rank difference of 2. Left alone.
#
# THE ABOVE WAS MEASURING THE SOLVER. That drift is not a cosmetic nuisance: the
# update is a power iteration on a row-stochastic operator whose spectral radius is
# exactly alpha, so alpha = 1.0 sits ON the singularity and everything above it
# diverges geometrically. The re-standardization at the end hid the divergence and
# turned it into a "cliff", and no sweep run that way could have found an optimum
# above 1.0 whether or not one existed.
#
# oppadj.solve_srs now solves (I - alpha*P*M*P) x = P x0 directly, on the complement
# of the constant vector - the one direction that cannot matter, since adding a
# number to every rating changes nothing and the caller z-scores the result. Re-swept
# on the exact solution:
#
#   alpha    0.00   0.60   0.70   0.75   0.85   0.90   1.00   1.25   1.50
#   Brier   .2121  .2095  .2090  .2088  .2085  .2085  .2135  .2215  .2422
#
# The optimum is a broad plateau from about 0.75 to 0.90 and 1.00 IS NOW WORSE THAN
# ANY OF IT - .2135 against .2085 - because the exact solve at alpha = 1 is the
# near-singular system the iteration was papering over. Shipping 0.85: on the flat
# part, with margin from the singularity rather than sitting on it.
#
# This value is still selected on the same LOSO it is reported against, which is
# what scripts/nested_cv.py exists to correct - see the note there for the honest
# forward number.
#
# TWO REFINEMENTS WERE TESTED AND REJECTED (scripts/per_stat_oppadj.py):
#
#   per-stat instead of per-composite. Three features are true pairs (havoc, red-zone
#   TD, pressure) and could be adjusted against their own counterpart rather than
#   against the whole opposing composite. Implemented in oppadj.adjust_per_stat and
#   measured: Brier .2035 -> .2031, inside the noise, and the ratings it produces
#   correlate .998-.999 with the composite ones, moving teams 2 places on average.
#   The composite correction already captures it. Kept as code, not shipped.
#
#   splitting home from away. The home effect is real and large - 0.51 team-sd on
#   13,726 team-games - but a season rating only inherits it insofar as teams differ
#   in home-game share, and they barely do (mean .524, sd .069). Implied bias sd is
#   0.035 team-sd against the opponent adjustment's 0.559, i.e. 16x smaller, and
#   correcting for it moves teams 1 place. It cancels within a season.
OPP_ADJ_ALPHA = 0.85

# --- Talent signal: blend roster-aware PFF talent with CFBD recruiting ---------
# PFF roster-aware talent (this year's roster x last year's grades) is the strongest
# single talent signal; CFBD adds the orthogonal recruiting axis. 50/50 blend beat
# either alone on LOSO (scripts/validate_roster_talent.py).
TALENT_BLEND = 0.5  # weight on PFF roster-aware (1-w on CFBD)

# --- v3: player WAR as a third talent axis -------------------------------------
# talent = (1-WAR_BLEND) * (TALENT_BLEND mix of PFF/CFBD) + WAR_BLEND * WAR, where
# WAR is the summed prior-year WAR of this year's roster from war_model/
# (PFF grades + CFBD PPA -> Massey -> wins above replacement).
#
# WAR does NOT beat the PFF grade signal as a replacement - swapped in for it, LOSO
# 2022-25 is a wash (Brier 0.2066 either way, scripts/validate_war_talent.py). It
# earns its place only as an ADDITION, because it carries the CFBD play-value signal
# and a depth weighting that falls out of snap counts rather than a fitted position
# weight vector. Swept in scripts/sweep_war_talent.py:
#
#   WAR share   0.00     0.15     0.25     0.35     0.50     0.80
#   Brier      .2048    .2042   .2040    .2041    .2046    .2059
#
# Re-derived after the facet reweighting changed what WAR measures
# (scripts/roster_vs_results.py, a joint sweep of WAR share against the uncertainty
# lambda that controls how far teams regress off last season's results):
#
#   WAR share   0.00     0.25     0.40     0.55     0.70     1.00
#   Brier      .2050    .2039   .2038    .2041    .2046    .2059
#
# The optimum moved 0.25 -> 0.40 once WAR stopped being fitted to the season it was
# describing. The accuracy difference between the two is .0001, which is nothing -
# what changed is that the evidence now supports the higher share rather than merely
# tolerating it. WAR alone is still clearly worse than the blend (.2059).
#
# The companion question - whether last season's results carry too much weight - is
# NOT supported. Sweeping the uncertainty lambda at the best WAR share:
# 0.00 -> .2040, 0.25 -> .2038, 0.50 -> .2038, 0.75 -> .2041, 1.00 -> .2049.
# It already sits in the flat optimum; pushing further onto the roster costs accuracy.
# Set to 0.0 to disable; the frame falls back cleanly if the WAR build is absent.
WAR_BLEND = 0.40

# --- EA ratings for lightly-played players --------------------------------------
# Below this many prior-season snaps the 2026 projection takes EA College Football's
# RANKING of the player instead of its own, quantile-mapped onto our WAR distribution
# within his position group so only the ordering changes and the wins scale stays ours.
# 0 disables it and the model reads projections_2026_v2.csv as before.
#
# The case for it: on the 2025 holdout our projection correlates .634 with what a
# no-prior-snap player actually did against .805 for a 600+ snap one, and EA is most
# different exactly there - agreeing .526 on no-history players against .903 on
# established starters. EA also rates 86% of the players under this threshold, where we
# are otherwise imputing from a positional prior.
#
# THE CASE AGAINST IT, WHICH IS NOT SETTLED. Nothing measured shows EA is more ACCURATE
# in the tail, only that it is different and not merely restating recruiting (19% of its
# view of a no-history player). The one accuracy test that exists - EA's CFB 26 team
# ratings against 2025 results - had EA behind, .42 against our .51. Settling it needs
# CFB 26 launch player ratings, which are behind a robots.txt that disallows this
# crawler. See war_model/ea/gap_analysis.py.
# EA is no longer an input. The projection is PFF facets, CFBD play value and
# production only; 0 disables the snap-threshold blend that handed unproven players
# EA's ordering. Kept as a knob rather than deleted so the comparison can be rerun.
EA_BLEND_SNAPS = 0

# --- v3.1: features retired after a collinearity audit --------------------------
# scripts/feature_audit.py + scripts/feature_sets.py. The six inputs were checked for
# redundancy three ways - correlation and VIF on the fitted design, leave-one-feature-
# out LOSO, and forward selection - and two of them were not earning their place:
#
#   pythag      r = +0.61 with O and +0.63 with D, the highest VIF in the set (3.0).
#               It is built from points scored and allowed, which is what the O and D
#               composites already measure. Dropping it: Brier 0.2054 -> 0.2053,
#               accuracy 0.676 -> 0.680.
#   fp_margin   actively costs accuracy. Dropping it: Brier 0.2054 -> 0.2045, and
#               forward selection adds it last and gets worse (0.2045 -> 0.2054).
#
# Dropping both: Brier 0.2047, four features, no VIF above ~1.9. The remaining four
# are NOT redundant with each other, which was the other half of the question:
# returning is nearly orthogonal to everything (max |r| 0.21) and talent is the single
# most valuable feature in the set (dropping it costs +0.0040 Brier).
#
# pythag stays in src/projection.py, where it is a predictor of next season's O/D -
# a different job from being a model feature.
#
# talent joined them when UNCERTAINTY_LAMBDA went to 0.70. At that shrinkage the O/D
# composites are 70% talent BY CONSTRUCTION, so a standalone talent column is close
# to a second copy of them: r(off_edge, talent) = 0.87 and VIF = 7.6, against the 3.0
# that was enough to retire pythag. The ridge responds the way it always does to a
# duplicated column - it flips talent NEGATIVE (+0.50 -> -0.22) and roughly triples
# O/D to compensate. Predictions are indifferent (LOSO Brier 0.2035 with or without
# it) but the fitted weights stop meaning anything, and the app would have shown a
# negative talent weight for a model that leans on talent harder than ever.
#
#   k     r(O,talent)  VIF   talent coef      Brier keeping it / dropping it
#   0.11     0.50      1.7      +0.59              0.2035
#   0.55     0.76      3.9      +0.20              0.2037 / 0.2033
#   0.70     0.87      7.6      -0.22              0.2035 / 0.2035
#
# Talent has NOT left the model - it is inside O and D at 70%, which is why dropping
# the column costs nothing. It is still exported for display and still drives the
# shrink; only the redundant second coefficient is gone.
# Set to [] to restore the original six.
DROPPED_FEATURES = ["fp_margin", "pythag", "talent"]

# Team-level features pulled from CFBD /stats/season/advanced.
# `higher_is_better=False` features get sign-flipped so larger == stronger.
# These map to the doc's §1A inputs; PFF pressure% / explosive rush-pass splits
# are NOT in CFBD free and are intentionally omitted (would require PFF data).
# Feature set after data-driven assembly (scripts/feature_analysis.py + explore.py):
# explosiveness dropped (~0 signal); 0.7 collinearity prune keeping the better
# predictor; high-signal CFBD stats (havoc, line yards) + TruMedia stats (red-zone
# TD, PFF pressure) added. fp_margin is a TEAM-LEVEL term (not in O/D composites).
# 5 offense / 5 defense + fp_margin. CFBD-native vs TruMedia split tracked below.
FEATURES = {
    # --- offense composite (O) ---
    "off_success_rate":  {"higher_is_better": True},   # CFBD: Success Rate
    "off_rush_ppa":      {"higher_is_better": True},   # CFBD: Offensive Rush EPA
    "off_havoc":         {"higher_is_better": False},  # CFBD: disruption allowed
    "off_rz_td":         {"higher_is_better": True},   # TruMedia: Red Zone TD%
    "off_press_allowed": {"higher_is_better": False},  # TruMedia: PFF pressure allowed
    # --- defense composite (D) ---
    "def_ppa":           {"higher_is_better": False},  # CFBD: EPA/play allowed
    "def_line_yds":      {"higher_is_better": False},  # CFBD: line yards allowed
    "def_havoc":         {"higher_is_better": True},   # CFBD: disruption generated
    "def_press":         {"higher_is_better": True},   # TruMedia: PFF pressure generated
    "def_rz_td":         {"higher_is_better": False},  # TruMedia: Red Zone TD allowed
    # --- team-level term (not in O/D) ---
    "fp_margin":         {"higher_is_better": True},   # TruMedia: field position margin
}

# Which features come from CFBD (have non-null every season) vs TruMedia (2021-25,
# left-joined; missing years filled to neutral at standardization).
CFBD_FEATURES = ["off_success_rate", "off_rush_ppa", "off_havoc",
                 "def_ppa", "def_line_yds", "def_havoc"]
TEAM_LEVEL_FEATURES = ["fp_margin"]  # excluded from the O/D composites

# L2 (Ridge) inverse-strength candidates, chosen by CV in train.py (doc §4.3).
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]

# --- v2: logistic + margin-model ensemble -------------------------------------
# p = ENSEMBLE_W * p_logistic + (1 - ENSEMBLE_W) * Phi(pred_margin / sigma).
# Margin (MOV) carries strength info the binary outcome discards. LOSO 2022-25:
# Brier 0.2048 -> 0.2044, log-loss 0.5930 -> 0.5919, acc 67.4 -> 67.7% (flat
# plateau w in [0, 0.5]; 0.4 keeps the classification-fit logistic in the mix).
# Isotonic calibration was dropped in the same pass (it only added noise:
# 0.2052 -> 0.2048 Brier; the linear logistic is already well calibrated).
ENSEMBLE_W = 0.4

# --- v3.6: football scores are not smooth ---------------------------------------
# Points arrive in 7s, 3s, 6s and 8s, so margins pile up on 3 and 7 and avoid 4 and 5.
# scripts/score_shape.py fits the actual distribution of the margin given the predicted
# one (a kernel over past games, no shape assumed) and a table of real final scorelines,
# then asks whether either is worth using. The answer is different for the two, which is
# why they ship separately.
#
# WIN PROBABILITY: NOT ADOPTED. Swapping the empirical distribution in for the normal
# inside the ensemble is a small loss, and using it alone is a slightly bigger one.
# LOSO 2021-25, refitting everything inside each fold:
#
#   arm                  Brier    log-loss   accuracy
#   current (normal)    .2036      .5907       .6730
#   PMF in ensemble     .2038      .5916       .6794
#   PMF alone           .2040      .5927       .6769
#
#   PMF in ensemble vs current, per season Brier +0.0002 +/- 0.0004, better in 2 of 5.
#
# That is not a surprise on reflection: a win probability only cares which side of zero
# the margin lands on, and the normal already gets the mass either side of zero right.
# Everything the empirical distribution knows is about WHERE inside each half the margin
# lands, and the binary outcome throws exactly that away. So the win model is unchanged.
#
# KEY NUMBERS: ADOPTED, and this is where the normal is not merely equivalent but wrong.
# Out-of-sample frequency the margin lands on each value. The normal column integrates
# its density over the half-point either side of the integer, at the fold's own fitted
# sigma, so it is a like-for-like claim about the same event:
#
#   margin        3       7       4      10      14      21
#   actual     .1055   .0880   .0360   .0473   .0401   .0371
#   normal     .0383   .0365   .0380   .0343   .0306   .0229
#   empirical  .1064   .0884   .0364   .0481   .0404   .0371
#
# The normal is off by 2.8x on the single most common margin in the sport. Worse, it
# ranks 4 ABOVE 3 - it has to, being unimodal and smooth - when a 3-point game is
# nearly three times as common as a 4-point one. No amount of refitting sigma repairs
# that; the shape is wrong, not the scale.
#
# DISPLAYED SCORES: ADOPTED. The app used to round each side of the implied score
# independently, which produces 28-24 for a game that far more often finishes 27-24.
# Picking the most likely real scoreline near the predicted total and margin instead:
#
#                MAE per side   exact margin   exact score
#   rounding         9.746          .0257         .0014
#   score table      9.755          .0518         .0064
#
# The same accuracy about the level, twice as often exactly right about the margin and
# four times as often exactly right about the score. The table's pick lands on a 3- or
# 7-point margin 58% of the time against 19% of games finishing there, and that gap is
# correct rather than a defect: a projected score is a MODAL prediction and a mode
# concentrates. The margin distribution's own most likely value is 3 or 7 for 90% of
# games, so the table is if anything less concentrated than the distribution it is
# drawn from.
#
# 0 disables both and the app falls back to rounding; artifacts/score_shape.json is
# regenerated by scripts/score_shape.py and exported into viz/data/model.json.
SCORE_SHAPE = 1

# --- v2: ROSTER-WEIGHTED rating variant ---------------------------------------
# Alternative projection lens that leans on THIS year's roster rather than last
# year's results: 70% PFF roster-aware talent (2026 two-deep x 2025 grades) and
# full §C uncertainty shrinkage (teams with low returning production regress all
# the way to their talent baseline). LOSO 2022-25: Brier 0.2053 vs 0.2044 for
# the balanced default, same 67.7% accuracy — a small, known trade for a view
# that isn't just an echo of last season's standings.
# NO LONGER SHIPPED (v3.5). This was the "roster-weighted" lens - a second rating
# variant that leaned 70% on two-deep talent with full continuity shrinkage. It was a
# knowingly worse backtest (Brier .2053 against .2043) kept as an alternative view, and
# the app now presents one set of numbers on the 38/38/25 talent blend instead. The
# scripts still accept a `roster` argument if it is ever wanted back; nothing in the
# shipped pipeline passes it.
ROSTER_VARIANT = {"talent_blend": 0.7, "unc_lambda": 1.0}

ARTIFACTS.mkdir(parents=True, exist_ok=True)
DATA_RAW.mkdir(parents=True, exist_ok=True)
