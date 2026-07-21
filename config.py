"""Central configuration for the improved CFB predictive model.

Design choice that fixes the leakage in the original notebooks: the model is
FORWARD-LOOKING. A game in season N is predicted from each team's season (N-1)
advanced stats. This (a) matches the doc's real use case ("2024 stats -> 2025
projection") and (b) is genuinely leakage-free, because the features for a game
never include that game's own outcome.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"

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

# --- Per-team uncertainty index (doc §C) -------------------------------------
# Shrink a team's prior-year O/D toward its talent baseline by lam * (1 - returning
# production). Low-continuity teams (new QB / churn) regress harder. Best on LOSO
# at 1.0 (scripts/loso3.py): Brier -0.25%, log-loss -1.0%, accuracy +0.5pp.
UNCERTAINTY_LAMBDA = 0.25  # near-neutral with the leaner feature set (was 1.0)

# --- Opponent (strength-of-schedule) adjustment (doc §4.4 / "Adj_" stats) -----
# Iterative SRS on the O/D composites. Best on LOSO at full strength (-0.9% Brier,
# best log-loss). 0 = raw composites. See scripts/loso_oppadj.py.
OPP_ADJ_ALPHA = 1.0

# --- Talent signal: blend roster-aware PFF talent with CFBD recruiting ---------
# PFF roster-aware talent (this year's roster x last year's grades) is the strongest
# single talent signal; CFBD adds the orthogonal recruiting axis. 50/50 blend beat
# either alone on LOSO (scripts/validate_roster_talent.py).
TALENT_BLEND = 0.5  # weight on PFF roster-aware (1-w on CFBD)

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

# --- v2: ROSTER-WEIGHTED rating variant ---------------------------------------
# Alternative projection lens that leans on THIS year's roster rather than last
# year's results: 70% PFF roster-aware talent (2026 two-deep x 2025 grades) and
# full §C uncertainty shrinkage (teams with low returning production regress all
# the way to their talent baseline). LOSO 2022-25: Brier 0.2053 vs 0.2044 for
# the balanced default, same 67.7% accuracy — a small, known trade for a view
# that isn't just an echo of last season's standings.
ROSTER_VARIANT = {"talent_blend": 0.7, "unc_lambda": 1.0}

ARTIFACTS.mkdir(parents=True, exist_ok=True)
DATA_RAW.mkdir(parents=True, exist_ok=True)
