# Historical recovery and ensemble comparison

October 12, 2026. Research only; production continues to average four member probabilities equally.

## What was recovered

The actual local checkout under `C:/Users/marks/Projects/04 - College Football Model`
contains the previously missing external source tree. Both purchased workbooks,
2014–2025 player PFF exports, historical team-season and weekly PFF tables, CFBD
caches, expanded-WAR trees, and earlier multi-season prediction exports were found.
Frames for 2021–2025 were regenerated through the production loaders and saved
privately under `artifacts/historical_research/`; no licensed rows are published.
The committed production preseason WAR file is byte-identical to the external
`war_all_facets_v52/preseason_team_war.csv`. The older `war_all_facets` file differs
and was not substituted. The output JSON records hashes and coverage counts.

Local PFF access successfully retrieved a 2025 team directory (265 rows), 2024
blocking (5,808 rows), and historical 2025 offense/defense tables (281 rows each).
The fresh Actions probe also recovered CFBD but still has no `PFF_API_KEY` secret.
See `audit/output/source_recovery_status.json` for both status-only probes.
The local cache and fresh Actions advanced-game responses have 3,314 and 3,316
rows respectively: source retrieval vintages differ even when credentials work.

## Baseline and chronological contract

Fresh member fits reuse the recorded tuning selections made inside each earlier-season
training pool, bypassing the unversioned pickle cache. Sanitized selections are now
committed so the experiment no longer depends on an ignored JSON artifact. Preseason
projection training uses earlier years; full season roster membership remains a
retrospective approximation to the actual preseason roster.

The lean v5.3 PFF/base architecture was replayed on 2,189 games in 2023–2025.
Canonical runtime probabilities reconcile with independently fitted pandas stacks
on eligible exact-table PFF weeks to at most 6.2e-12. Missing historical tables
use the base stack; the 2026 carry-forward policy is not retroactively imposed.
The current 2026 full production architecture is separately reconciled against all
138 published power ratings. These are two different baseline checks.

The historical in-season WAR signal is excluded. Its saved calibration uses
leave-season-out rules: the 2023 rule was trained on 2022, 2024 and 2025.
That avoids its own year but uses future years, so it is unsuitable for a strictly
chronological comparison. A complete current-WAR multi-season baseline still needs
earlier-year-only calibration of the player filter, facet weights and production
conversion, followed by fresh team histories and member stack fits.

2023 supplies 729 out-of-fold member predictions for training. Test 2024 uses only
2023; test 2025 uses 2023–2024. All alternatives are evaluated on the identical 1,460
games in 2024–2025. New logistic scalers and coefficients and constrained blend
weights use training rows only. Penalties were fixed before scoring; there is no
held-out hyperparameter search or automatic candidate selection. Inherited baseline
PCA/residualization and standardization use each year's known preseason team universe;
this is not a complete refit of every upstream transform on a pooled training-only
universe. Other upstream WAR parameter provenance still needs auditing.

The feature-union candidate uses the actual deduplicated union of the four preseason
feature lists, replacing aggregate member preseason logits, plus member-specific score
updates, shared available form and home field. It is one final logistic probability.
This is substantially more complete than the earlier aggregate-live-input experiment.
It retains the existing score update mechanics rather than estimating a new state model.

## Matched results

| Candidate | Brier | Δ vs equal | Log loss | 2024 Brier | 2025 Brier |
|---|---:|---:|---:|---:|---:|
| Equal probability average (lean baseline) | 0.178352 | +0.000000 | 0.528725 | 0.181445 | 0.175163 |
| Learned constrained probability weights | 0.178352 | -0.000000 | 0.528725 | 0.181448 | 0.175161 |
| Regularized member logit model | 0.178681 | +0.000329 | 0.529605 | 0.182136 | 0.175120 |
| Deduplicated preseason feature union + live inputs | 0.179824 | +0.001472 | 0.534503 | 0.183488 | 0.176047 |
| Member logit model + opposing O×D | 0.178714 | +0.000362 | 0.529821 | 0.182104 | 0.175221 |
| Feature union + opposing O×D | 0.179890 | +0.001538 | 0.534761 | 0.183509 | 0.176160 |
| Feature union + nonlinear opposing gap | 0.179878 | +0.001526 | 0.534589 | 0.183396 | 0.176252 |

Learned weights stay close to 25% each and are effectively tied with equal averaging.
The logit model improves slightly in 2025 but loses more in 2024. The full feature
union loses in both seasons at the fixed penalty. Opposing-unit products and nonlinear
gaps do not repair that loss. The 2026 product result did not generalize in this
recovered comparison. This does not rule out other carefully regularized feature
unions or interaction families, and is not evidence for a production replacement.

The product `O_A*D_B - O_B*D_A` and nonlinear gap
`g(O_A-D_B) - g(O_B-D_A)`, `g(x)=x*abs(x)`, negate when teams swap.
Logistic models have no intercept or centering, so neutral probabilities remain
complementary. Offense and defense are standardized signals, not predicted points.

The JSON reports log loss, calibration bins/ECE, every season-week score, paired
season-week bootstrap intervals and whole-season bootstrap intervals. The paired
intervals include zero for all alternatives. There are only two evaluated seasons;
week blocks do not capture all team dependence, and these intervals do not correct
for historical candidate selection. Retrospective source pulls do not establish
prospective betting performance. Original timestamped historical source snapshots
cannot be recovered from current API responses alone.

## Remaining dependencies and decision

Keep production unchanged. Remaining work is specific: reconstruct chronological
in-season WAR rules and audit all upstream facet/conversion fits; establish source
vintages or collect prospective shadow predictions; configure the absent repository
PFF secret for future captures. The unique workbooks are available locally but are
not public or API-regenerable. API credentials and licensed raw tables stay private.

## Reproduce

`python -m scripts.historical_ensemble_research` with the external source tree and
the existing Python dependencies. It regenerates frames privately and emits aggregate
JSON, game-level predictions and this report. It never writes production model,
kickoff ledgers, frozen predictions or locked bets. Tests:
`python -m unittest tests.test_historical_ensemble_research tests.test_pff_carry_forward tests.test_ensemble_blend_research tests.test_ensemble_replay`.
Use `PYTHON` to select the interpreter when running the Node X-Ray tests on Windows.
