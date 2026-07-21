# CFB Predictive Model v2 (copy of ~/cfb-model with accuracy + viz upgrades)

## What's new in v2 (July 2026)

**Accuracy (validated by LOSO 2022–25, `scripts/loso_experiments*.py`):**

| change | Brier | log-loss | accuracy |
|---|---|---|---|
| v1 baseline | 0.2052 | 0.5975 | 0.673 |
| drop isotonic calibration (added noise; logistic already calibrated) | 0.2048 | 0.5930 | 0.674 |
| + ridge **margin-model ensemble** (`p = 0.4·p_logistic + 0.6·Φ(margin/σ)`) | **0.2044** | **0.5919** | **0.677** |

Held-out 2025: Brier 0.2045 → **0.2038**. Also tested and NOT adopted (no gain):
two-year O/D priors, Platt calibration, talent-blend/uncertainty-λ/pythag-exponent
retuning, MOV sample weighting. Bug fix: Air Force & Navy were silently dropped
(no 247 composite) — now kept via a low-percentile talent fallback.

**New pipeline (run in order):**

```bash
./venv/bin/python -m scripts.train             # retrain (now saves margin model too)
./venv/bin/python -m scripts.rank              # 2026 power ratings
./venv/bin/python -m scripts.simulate_playoff  # 20k-sim Monte Carlo -> CFP odds
./venv/bin/python -m scripts.prepare_logos     # map/download team logos
./venv/bin/python -m scripts.export_viz        # export data for the web app
python3 -m http.server 8642 -d viz             # open http://localhost:8642
```

**Playoff Monte Carlo (`scripts/simulate_playoff.py`):** simulates all 761
FBS-vs-FBS games on the real 2026 schedule + CCGs, then applies the confirmed
**2026-27 CFP format**: 12 teams; auto-bids for ACC/Big 12/Big Ten/SEC champs
(any ranking) + the highest-ranked Group of 6 team (champ or not); 7 at-large;
straight seeding, top-4 byes, first round at the higher seed, fixed bracket.
The committee ranking is proxied by `10·win% + 1.0·rating_z + 0.75·SOS_z`,
weights fit against the final 2025 CFP ranking (Spearman ρ = 0.923).

**Roster-weighted lens (July 2026):** a second rating variant that leans on the
2026 roster instead of 2025 results — 70% two-deep PFF talent (vs 50%) and full
§C uncertainty shrinkage (λ=1.0: low-continuity teams regress all the way to
their talent baseline). LOSO cost is known and small: Brier 0.2053 vs 0.2044,
same 67.7% accuracy. Generate with `scripts.rank roster`,
`scripts.simulate_playoff 20000 roster`, `scripts.export_viz roster`
(`*_roster` artifacts); the web app has a LENS toggle to flip between the two.
Params in `config.ROSTER_VARIANT`.

**Web app (`viz/`):** four views — Top 25 power ratings, playoff projection
(most-likely bracket + full odds), a client-side matchup simulator (win prob /
spread / total / projected score for any two FBS teams, any venue; JS math
verified equal to the Python model to 4 decimals), and a **Ratings Lab** editor.
Logos come from `~/Downloads/cfb_alt_logos` (all 138 FBS teams; 101 gaps
auto-filled from the ESPN CDN via `scripts/prepare_logos.py`).

**Ratings Lab (editable inputs):** every team's six model inputs (Off, Def,
FieldPos, Pythag, Talent, Returning — the frozen model's per-team feature vector)
are editable in a table. The scoring math stays fixed; edits change only the
inputs, and power ratings, ranks, the matchup sim, and the playoff Monte Carlo
all recompute **client-side** from them. Power is a neutral round-robin over the
136 rated teams (matches the backend to rounding with no edits); the playoff
re-sim is a faithful JS port of `scripts/simulate_playoff.py` (verified: 4k-sim
JS output matches the 20k-sim Python baseline within RNG). Edits persist in
localStorage, per lens, with per-row and global reset. Requires
`viz/data/schedule.json` (emitted by `scripts.export_viz`).

**Roster editing (`scripts/serve.py`) — player-level depth-chart edits.** A small
stdlib-only local server (no Flask) that serves the same `viz/` app *and* a roster
API. Run it instead of `http.server`:

```bash
./venv/bin/python -m scripts.serve      # serves viz/ + API on :8642 (~20s to load)
```

It exposes each team's 2026 two-deep (Ourlads) with every player's 2025 PFF grade
at `GET /api/rosters`. In the Ratings Lab, a 📋 button on each team opens the depth
chart: edit a grade, promote/demote depth, change position, or add/remove players.
`POST /api/recompute` re-derives **talent[2026]** from the edited roster (the only
input a 2026 roster touches) and rebuilds the frame with the **frozen** model math,
returning new vectors; the client recomputes ratings/playoff from there. Only
talent depends on the roster, so opponent-adjusted O/D, the talent→O/D slopes,
Pythagorean and returning are loaded **once** at startup — each edit is
milliseconds. Verified: the no-edit baseline reproduces the exported vectors to
0.00000; dropping Ohio State's QB1 grade 93→60 cut talent 1.56→1.00, power
86.4→83.0, rank #4→#6, and playoff odds 76%→59%. Roster edits persist in
localStorage and re-apply via the API on load. Without the server (plain
`http.server`), the app degrades gracefully: the 📋 buttons hide and the six
direct inputs remain editable.

---

# Original README (v1) — still accurate below except where noted above

A clean, leakage-free, calibrated reimplementation of the **win-probability core**
of the methodology doc, fed by **fresh CollegeFootballData.com (CFBD) pulls**.
Built to improve on the original `CFB_Pred_Model` notebooks.

## What this fixes vs. the original notebooks

| # | Issue in original | Fix here |
|---|---|---|
| 1 | Feature selection ran on the **full dataset before the train/test split** (leaks test data) | Forward-looking design: season N games predicted from season **N-1** stats; held-out test season |
| 2 | Final logistic fit on **unstandardized** features (penalty applied unevenly) | All features z-scored within season before fitting |
| 3 | Calibration was **measured but never applied** | Isotonic calibrator is fit on train, **saved, and applied** at predict time |
| 4 | L1 used despite doc specifying **L2/Ridge** | L2 logistic, `C` chosen by cross-validation on Brier |
| 5 | Power rankings set **every team to home** | Round-robin on a **neutral field** (`is_home=0`) |

## Quickstart

```bash
cd ~/cfb-model
# .env already contains CFBD_API_KEY (gitignored)
./venv/bin/python -m scripts.train     # pull data, train (matchup model), evaluate, save
./venv/bin/python -m scripts.compare   # 1-season: RAW vs TALENT vs CALIBRATED strength
./venv/bin/python -m scripts.loso      # LOSO CV of the strength methods
./venv/bin/python -m scripts.loso2     # LOSO CV: per-stat calibrated vs matchup model
./venv/bin/python -m scripts.rank      # 2026 power ratings + example matchup
```

## Model architecture: true matchup-adjusted, team-level (default)

Each team collapses to **two ratings** per season — an offensive rating `O` and a
defensive rating `D` (means of the standardized offensive / defensive stats). A
game is modeled as a real matchup, offense against the opponent's defense:

```
off_edge = O_home(N-1) - D_away(N-1)      # home offense vs away defense
def_edge = D_home(N-1) - O_away(N-1)      # home defense vs away offense
+ pythag_diff(N-1), talent_diff(N), returning_diff(N), home-field
```

The L2 logistic fits these on game outcomes (**target B**), which both blends the
prior-year ratings with the talent/returning/Pythagorean priors (= regression to
a calibrated mean) and weights offense vs defense. Probabilities are isotonic-
calibrated. This replaces the earlier per-stat *like-vs-like* vector, which wasn't
modeling true matchups.

Learned weights (trained on 2021–24): `talent` is the largest single coefficient
(~0.50), but prior-performance signals collectively (`off_edge`+`def_edge`+
`pythag` ≈ 0.59) outweigh it — matching the diagnostic that performance predicts
next year better than talent.

### Per-team uncertainty index (doc §C, `scripts/loso3.py`)

Teams with little returning production (new QB / roster churn) get their prior-year
O/D ratings pulled toward their talent baseline: `u = 1 - returning_fraction`, and
`O_adj = (1 - λu)·O + λu·(b_o·talent_z)`. LOSO over 2021–25 picks **λ=1.0** (default):

| | Brier | Log-loss | Accuracy |
|---|---|---|---|
| Uncertainty OFF | 0.2120 | 0.6272 | 0.662 |
| **Uncertainty ON (λ=1.0)** | **0.2115** | **0.6207** | **0.667** |

Small but consistent (−1.0% log-loss, +0.5pp accuracy). It also makes the model
*trust the ratings more*: pre-shrinking moves weight onto the O/D edges (0.18→0.30,
0.13→0.19) and off the flat talent term (0.50→0.46).

### Does the matchup structure help? (LOSO, `scripts/loso2.py`)

| Model | Brier | Log-loss | Accuracy |
|---|---|---|---|
| Per-stat calibrated | 0.2131 | **0.6211** | 0.661 |
| **Matchup (default)** | **0.2120** | 0.6272 | 0.662 |

Marginally better Brier (wins 4/5 folds), tied accuracy, slightly worse log-loss
— **essentially a wash for binary win prob.** That's expected: linear off/def
edges nearly collapse to a strength differential. The structural payoff is
interpretability + it's the correct base for **margins/spreads (doc §5.2)** and
interaction terms, where matchup detail actually matters.

## Regression to a *calibrated* mean (doc §C, improved)

Raw prior-season stats over-rate G5 over-performers, so we regress each team's
stats toward a preseason expectation. Two ways were built and tested:

- **TALENT shrinkage:** `adjusted_z[f] = (1-λ)·prior_z[f] + λ·b[f]·talent_z`,
  where talent is the CFBD recruiting composite (leakage-free, known preseason).
- **CALIBRATED projection (default):** a per-stat OLS fit to predict the season's
  actual stat from four preseason signals — prior-year stat, talent, **returning
  production**, and **Pythagorean win expectation** (`PF^x/(PF^x+PA^x)`, luck-
  adjusted). We regress toward this fitted value. Why: talent alone explains only
  R²≈0.09 of next-year strength; the blend reaches R²≈0.31 — a ~3× more accurate
  mean.

### Why we trust the calibrated version: LOSO, not one season

A single held-out season can't separate methods that differ by ~0.2% Brier — on
2025 alone, TALENT actually edged CALIBRATED. Leave-one-season-out CV
(`scripts/loso.py`, coefficients re-fit per fold) averages over 2021–2025:

| Method | Brier | Log-loss | Accuracy | vs RAW |
|---|---|---|---|---|
| RAW (prior stats) | 0.2199 | 0.6360 | 0.646 | — |
| TALENT (λ=0.65) | 0.2146 | 0.6237 | 0.655 | −2.4% Brier |
| **CALIBRATED (default)** | **0.2131** | **0.6211** | **0.661** | **−3.1% Brier** |

CALIBRATED wins on all three metrics and beats TALENT in 3 of 5 folds (it loses
only 2021 and 2025 — and 2025 is exactly the season the one-shot test had used).

> **2026 inputs:** CFBD hasn't loaded 2026 talent or returning production yet
> (the endpoints return empty for 2026). Returning production uses the real 2026
> numbers (Bill Connelly / ESPN) in `data/returning_2026.csv`, names reconciled to
> CFBD — this is the conceptually correct "entering-2026" continuity signal, not
> the off-by-one 2025 proxy. Talent still falls back to the 2025 composite until
> CFBD loads 2026. Both auto-switch to CFBD when it publishes. Note: returning
> production is a roster/continuity metric known *preseason*, so 2026 values exist
> now; performance stats and Pythagorean correctly use the completed 2025 season.

## How it works

1. **Data** (`src/data/`): pulls `/stats/season/advanced` and `/games` from CFBD,
   cached as JSON in `data/raw/`. No synthetic fallback — fails loudly without data.
2. **Features** (`src/features.py`): z-score each season's advanced stats, flip
   defensive signs so larger = stronger. A game's feature vector is
   `home_prior_stats - away_prior_stats`.
3. **Model** (`src/model.py`): L2 logistic + a learned home-field term, then an
   isotonic calibration map. Saved to `artifacts/model.json`.
4. **Predict** (`src/predict.py`): single matchups and neutral-field power ratings.

## Honest results (calibrated model, LOSO over 2021–2025)

- Brier ≈ 0.213, log-loss ≈ 0.621, accuracy ≈ 0.66 — **out of sample** (LOSO mean).
- Lower than the original notebooks' reported numbers because those were
  in-sample / leakage-inflated. This is the real forward-looking baseline:
  a preseason projection (prior stats + talent + returning production +
  Pythagorean) with no in-season info or injuries. Vegas-level Brier is
  ~0.20–0.21 for comparison.

## Features used (CFBD `/stats/season/advanced`)

`off_ppa`, `off_pass_ppa` (EPA/dropback), `off_rush_ppa`, `off_success_rate`,
`off_explosiveness`, `off_pts_per_opp` (finishing), `def_ppa`, `def_success_rate`,
`def_explosiveness`, `def_havoc`.

**Not reproducible from CFBD free** (would need PFF / TruMedia): `PFFPressured%`,
explosive **rush/pass splits**, `RZTD%` exact, down-and-distance scoring rates.

## Margin / spread model (doc §5.2, `scripts/spreads.py`)

Predicts each side's points as offense-vs-opponent-defense
(`points ~ O_scorer + D_opponent + home`, Ridge), then derives **spread** and
**total**. A win prob falls out via a normal model on the margin
(`P = Φ(margin/σ)`), which we use only as a coherence check.

LOSO over 2021–25:

| | value |
|---|---|
| Margin MAE | **14.85 pts** (home-field baseline 15.95 → **−7%**) |
| Margin RMSE | 18.82 |
| Implied win-prob Brier | 0.2207 (vs dedicated logistic 0.2115) |

Real signal over baseline, but short of Vegas (~10.5 MAE) — expected for a
*preseason-only* projection (no in-season form, injuries, or final depth charts).
The implied win prob sits right next to the logistic, confirming the two models
are coherent; use the **logistic for win prob, the spread model for margins/totals.**
Example 2026 lines: Ohio St −15 / Notre Dame −16.4 vs an average team; neutral
Oregon by ~4.6 over Alabama (total ~50).

## Experiments & findings

- **EWMA recency weighting (`scripts/loso_ewma.py`)** — weighting recent games of
  the prior season more heavily does **not** help next-season prediction; flat
  equal-weight wins (best EWMA +0.43% Brier vs flat). Late-season "form" is a
  noisier sample; the full-season average is the more stable carry-over signal.
- **Offseason regression (`scripts/plot_regression.py`)** — preseason projection
  vs end-of-prior-season rating has **slope ≈ 0.79** (~21% pull to the mean),
  roughly linear. See `artifacts/regression_plot.png`.
- **Feature assembly (`scripts/feature_analysis.py`)** — explosiveness dropped
  (~0 predictive signal); 0.7 collinearity prune keeping the better predictor of
  each cluster; distinct high-signal stats added. Final 6 features (3 O / 3 D, all
  pairwise |r|<0.7): `off_success_rate`, `off_rush_ppa`, `off_havoc`, `def_ppa`,
  `def_line_yds`, `def_havoc`. Net effect: same accuracy as the old 10-feature set
  with half the features (a parsimony win, not an accuracy win); the uncertainty
  index went near-neutral (λ 1.0 → 0.25).

## Model diagnostics (scripts/diagnostics.py)
Validity checks on the final model (out-of-fold where it matters):
- **Collinearity:** all VIF < 3 (clean). **Significance:** all features p<0.05 except
  pythag (p=0.06, kept — helps LOSO). **Overfit:** in-sample vs OOF Brier gap only
  +0.004. **Autocorrelation:** Durbin-Watson 2.04 (none over time).
- **Calibration:** isotonic is now fit on **cross-validated (out-of-fold)**
  predictions (not in-sample), fixing the over-confidence (OOF slope 0.87 → 0.89,
  Brier flat). The in-sample-calibration overfit is gone.
- **Remaining (minor):** spread model heteroscedastic (expected for football —
  point spreads fine, only intervals/totals need a variance model); mild team
  residual clustering (the static-preseason limitation the in-season Elo layer
  would address).

### Mean-regression formulas (scripts/loso_meanreg.py)
Holding average shrinkage constant, the *shape* (global / returning / variance /
rating-dependent / combined) is within ~0.001 Brier of each other and of no
shrinkage — the L2 blend already does regression-to-mean implicitly. Rating-
dependent is worst (offseason regression is linear). No explicit formula adopted.

## Roadmap (next improvements, in priority order)

1. ~~Regression-to-mean + talent prior (doc §C)~~ ✅ **done**.
2. ~~Calibrated mean (talent + returning production + Pythagorean)~~ ✅ **done**.
3. ~~True matchup-adjusted, team-level model (offense vs opponent defense)~~ ✅ **done**.
4. ~~Per-team uncertainty index (doc §C)~~ ✅ **done** (λ=1.0).
5. ~~Margin / spread + totals model (doc §5.2)~~ ✅ **done**.
6. ~~TruMedia feature expansion (RZ TD%, PFF pressure, field position)~~ ✅ **done**.
7. ~~Opponent / strength-of-schedule adjustment (doc §4.4)~~ ✅ **done** (SRS, α=1.0).
8. ~~Mean-regression formulas (doc §C)~~ ✅ tested — no form helps (kept implicit).
9. ~~In-season Elo layer core (doc §5.3)~~ ✅ **done** (−9% Brier, MOV updates).
### Elo parameter tuning (scripts/tune_elo.py)
- **K-factor: decaying 50→30 over the season is best** (just edges constant K=40).
  The by-week split confirms the intuition: high K helps EARLY (wk1-4: K50 0.1750 vs
  K30 0.1760), low K helps LATE (wk10+: K35 0.1899 vs K50 0.1914 — high K hurts late).
  Net pooled gain is small because the early/late effects partly offset and the MOV
  multiplier already adapts updates.
- **Home field: team-specific** beats constant — `HFA = 65 + 25·z(home-minus-away
  margin)`, shrunk. Best combo (K decay + team-HFA) → ~0.1855.
- No market blend by design (the goal is a market-distinct signal for betting edge).

## Roadmap (next improvements, in priority order)

10. **Elo refinements (doc §5.3):** PGWE (continuous postgame) [deferred by request];
    productionize live Elo + Elo→spread into rank/spreads. No market blend (by design).

### LOSO Brier progression (preseason model)
0.2115 (orig CFBD) → 0.2112 (+TruMedia) → 0.2093 (+opponent adj) →
**0.2048 (+roster-aware talent blend)**; log-loss → ~0.60.

### In-season Elo layer (scripts/backtest_elo.py) — the biggest jump
Seeding Elo from the preseason model and updating after each game (MOV multiplier)
cuts Brier **~9% overall** vs the static model, robust across 2022-25 (−5.9 to
−12.2% per season), growing through the year (wk1-4 0.180→0.175, wk5-9 0.212→0.190,
wk10+ 0.217→0.191). No blend needed — Elo starts at the preseason rating and only
improves. Fixes the static-preseason team-clustering the diagnostics flagged.

### Talent signal — roster-aware PFF + CFBD blend (DEFAULT)
- Historical *team-level* PFSN talent ≈ flat/worse than CFBD (redundant with prior
  performance already in the model).
- **Roster-aware PFF talent** (season-N roster × N-1 grades, transfer-aware) with
  position weights **re-optimized on PFF grades** (NNLS vs win%; far more balanced
  than PFSN's QB/CB-heavy weights), **blended 50/50 with CFBD**, beats CFBD on all
  metrics (LOSO 2022-25): Brier 0.2096→**0.2048**, log-loss 0.6123→**0.5998**, acc
  0.664→**0.676**. Roster turnover is the value-add; CFBD stays as the orthogonal
  recruiting anchor. 2026 uses the Ourlads two-deep × 2025 PFF grades.
- **Signal map (scripts/compare_signals.py):** three independent axes — prior
  performance (pythag≈prior_strength r=0.92), recruiting (CFBD, orthogonal),
  continuity (returning, ~0 corr with all). PFF_roster (r=0.46 vs win%) is the
  strongest talent signal and only 0.36-correlated with CFBD → complementary.
- **Interactions / nonlinearity:** screened all pairwise interactions + quadratic +
  tempo×efficiency → none materially useful; model is additive/linear. Logs N/A
  (features are standardized z-scores).
- **pythag vs O/D:** the 0.92 redundancy is with raw prior_strength, NOT a model
  feature; dropping pythag *hurt* LOSO (0.2093→0.2098), so it's kept.
