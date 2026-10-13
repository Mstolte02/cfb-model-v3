# Chronological in-season WAR and opponent interactions

Decision recorded October 12, 2026. Keep the current production model unchanged. Rebuilt in-season WAR has mixed, uncertain incremental value; these results establish neither a reliable gain nor a reason to remove it. Do not promote the tested interactions. Retain the passing–coverage gap and after-contact–tackling product as research leads with their linear controls.

## What was rebuilt

The complete **in-season PFF facet-WAR measurement and calibration chain** was rebuilt
in isolated directories for 2022–2025. For each target season, earlier seasons alone
fit facet consolidation/PCA, reliability, facet weights, team scales and WAA conversion.
Raw player measurements use earlier-season weighted moments; the target year does not
normalize its own inputs. Player Kalman priors and the conversion to WAR use earlier
years. Opponent strength for an observed window uses games through that window's cut,
and postseason rows with reset week numbers are excluded.

Player windows end at weeks 3, 6 and 9. Completed later windows supply calibration
labels only for subsequent seasons. 2022 supplies the first calibration history;
2023 fits its rule on 2022, 2024 on 2022–2023, and 2025 on 2022–2024. Team histories
contain 402, 402 and 408 team/cut rows respectively. A cut becomes eligible only
when game week is strictly greater than the cut. There are no future-year
leave-one-season-out rules or inherited full-history normalization in this update.

**Scope limit:** the four members' original preseason `war_projected` and
`player_prod_war` inputs retain the recovered expanded-WAR production lineage.
Their earlier global upstream transforms were not rebuilt here. This is not a claim
that every input in the entire production model is now chronologically clean.
The comparison isolates the rebuilt in-season update conditional on those shared
preseason inputs. A fully end-to-end preseason WAR audit remains a dependency for
stronger production claims.

## Evaluation contract

All 25 candidates use identical games and four equally weighted member probabilities.
Preseason member coefficients are refitted on earlier years using the previously
saved earlier-year selections. Unit scaling, logistic coefficients and regularization
are fitted without outer test labels. Each stack selects C from
0.01, 0.03, 0.1, 0.3 and 1 by earlier chronological validation log loss.
The first inner split is weeks before 9 versus weeks 9 onward when only one year is
available. This selection is repeated equally for every candidate and member.

There are 729 warmup predictions in 2023, then **1,460 evaluated games**:
741 in 2024 and 719 in 2025. The team stack needs 2023's earlier-calibrated WAR to
learn a nonzero WAR coefficient, so 2023 is not credited as independent WAR evidence.

The `no_inseason_war` control still includes preseason WAR. It is a newly tuned,
matched ablation of the lean stack, not an exact replay of the already published
2026 production model or its old fitted penalties. In this research all games use
the same lean columns; absent PFF offense is zero-imputed rather than switching
to production's no-PFF base stack. This differs from the prior historical runtime
reconciliation and explains why its baseline score is slightly different.

## Opposing units

Six predefined unit pairings are tested: passing EPA versus coverage EPA allowed;
rushing EPA versus rushing EPA allowed; pressure allowed versus pressure generated;
yards before contact versus yards before contact allowed; explosive pass rate versus
explosive pass rate allowed; yards after contact versus yards after contact allowed.
Defensive signs are oriented so higher means stronger. Every feature uses the exact
prior-week PFF export. Missing interactions become zero; missing standardized main
effects use the training mean. Scales use only the earlier training observations.

Each pairing has linear O/D differences, an opposing product, and a nonlinear gap.
The product is O_home D_away − O_away D_home. The gap is
g(O_home − D_away) − g(O_away − D_home), where g(x)=x|x|.
Nonlinear candidates include their identical linear main effects. The combined
families contain all six pairs. These terms reverse sign when sides swap and preserve
complementary probabilities. Separately standardized O and D are not predicted points.

## Scores on common held-out games

Lower Brier/log loss/ECE are better. Higher AUC/accuracy are better. Calibration
intercept should approach 0 and slope 1; these are diagnostics fitted on evaluation
predictions, never used to adjust those predictions. ECE uses ten fixed bins.
Confidence errors are incorrect picks among predictions at least 80% confident;
the selected population changes across candidates, so this is not a paired causal comparison.

| Candidate | Brier | Log loss | AUC | Accuracy | Cal. intercept | Cal. slope | ECE | Confident error (n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_inseason_war | 0.178451 | 0.529276 | 0.801070 | 72.40% | 0.0382 | 1.0443 | 1.50% | 10.43% (460) |
| chronological_war | 0.178399 | 0.529577 | 0.801017 | 72.60% | 0.0442 | 1.0219 | 1.26% | 10.49% (467) |
| war_units_linear | 0.180138 | 0.534371 | 0.797304 | 72.12% | 0.0462 | 1.0167 | 1.86% | 10.41% (461) |
| war_units_product | 0.180188 | 0.534701 | 0.797434 | 72.33% | 0.0306 | 1.0594 | 2.33% | 9.82% (438) |
| war_units_gap | 0.180341 | 0.534568 | 0.796690 | 71.99% | 0.0438 | 1.0131 | 2.01% | 10.89% (450) |
| war_units_product_gap | 0.180567 | 0.535543 | 0.796616 | 72.19% | 0.0315 | 1.0503 | 2.51% | 10.28% (428) |
| no_war_units_product | 0.180439 | 0.534911 | 0.796665 | 72.74% | 0.0243 | 1.0808 | 3.34% | 9.22% (423) |
| passing_coverage_linear | 0.178675 | 0.530230 | 0.800260 | 72.47% | 0.0467 | 1.0208 | 1.51% | 10.00% (470) |
| passing_coverage_product | 0.178861 | 0.530724 | 0.799816 | 72.19% | 0.0451 | 1.0203 | 2.09% | 10.32% (465) |
| passing_coverage_gap | 0.178425 | 0.529424 | 0.800653 | 72.74% | 0.0486 | 1.0130 | 1.66% | 10.50% (476) |
| rushing_front_linear | 0.178814 | 0.530597 | 0.800359 | 72.74% | 0.0456 | 1.0185 | 1.88% | 10.58% (463) |
| rushing_front_product | 0.178831 | 0.530711 | 0.800368 | 72.81% | 0.0451 | 1.0184 | 1.67% | 10.78% (464) |
| rushing_front_gap | 0.178801 | 0.530563 | 0.800453 | 72.53% | 0.0447 | 1.0197 | 1.69% | 10.50% (457) |
| protection_pass_rush_linear | 0.178763 | 0.530184 | 0.800050 | 72.67% | 0.0447 | 1.0115 | 1.02% | 10.29% (476) |
| protection_pass_rush_product | 0.178743 | 0.530160 | 0.799922 | 72.53% | 0.0460 | 1.0113 | 1.51% | 10.29% (476) |
| protection_pass_rush_gap | 0.179030 | 0.530630 | 0.799362 | 72.05% | 0.0452 | 1.0004 | 1.50% | 10.04% (468) |
| run_blocking_front_linear | 0.178584 | 0.530034 | 0.800643 | 72.81% | 0.0484 | 1.0120 | 1.23% | 10.94% (466) |
| run_blocking_front_product | 0.178813 | 0.530672 | 0.800184 | 72.33% | 0.0469 | 1.0147 | 1.45% | 10.56% (464) |
| run_blocking_front_gap | 0.178841 | 0.530656 | 0.800008 | 72.67% | 0.0501 | 1.0081 | 1.21% | 10.73% (466) |
| explosive_pass_prevention_linear | 0.178534 | 0.529924 | 0.800891 | 72.12% | 0.0470 | 1.0202 | 1.95% | 10.87% (469) |
| explosive_pass_prevention_product | 0.178564 | 0.529953 | 0.800845 | 72.05% | 0.0469 | 1.0167 | 2.65% | 11.28% (470) |
| explosive_pass_prevention_gap | 0.178578 | 0.529980 | 0.800773 | 72.05% | 0.0476 | 1.0200 | 2.10% | 10.80% (463) |
| after_contact_tackling_linear | 0.178989 | 0.531372 | 0.799876 | 72.60% | 0.0477 | 1.0076 | 1.33% | 10.57% (473) |
| after_contact_tackling_product | 0.178727 | 0.530615 | 0.800500 | 72.67% | 0.0460 | 1.0178 | 1.24% | 9.91% (464) |
| after_contact_tackling_gap | 0.178980 | 0.531428 | 0.800223 | 72.74% | 0.0447 | 1.0117 | 1.05% | 10.28% (467) |

## Stability

| Candidate | 2024 Brier | 2025 Brier | 2024 log loss | 2025 log loss |
|---|---:|---:|---:|---:|
| no_inseason_war | 0.181378 | 0.175435 | 0.535193 | 0.523177 |
| chronological_war | 0.181871 | 0.174822 | 0.537168 | 0.521754 |
| war_units_linear | 0.184538 | 0.175604 | 0.544229 | 0.524210 |
| war_units_product | 0.183743 | 0.176524 | 0.543196 | 0.525946 |
| war_units_gap | 0.184639 | 0.175912 | 0.544143 | 0.524700 |
| war_units_product_gap | 0.184215 | 0.176808 | 0.544189 | 0.526632 |
| no_war_units_product | 0.183612 | 0.177169 | 0.542271 | 0.527325 |
| passing_coverage_linear | 0.181883 | 0.175369 | 0.537316 | 0.522926 |
| passing_coverage_product | 0.181782 | 0.175851 | 0.537187 | 0.524064 |
| passing_coverage_gap | 0.181359 | 0.175401 | 0.536275 | 0.522364 |
| rushing_front_linear | 0.182329 | 0.175190 | 0.538080 | 0.522884 |
| rushing_front_product | 0.182356 | 0.175197 | 0.538299 | 0.522890 |
| rushing_front_gap | 0.182312 | 0.175183 | 0.538031 | 0.522867 |
| protection_pass_rush_linear | 0.182309 | 0.175108 | 0.538031 | 0.522097 |
| protection_pass_rush_product | 0.182272 | 0.175106 | 0.537903 | 0.522180 |
| protection_pass_rush_gap | 0.182262 | 0.175700 | 0.537893 | 0.523144 |
| run_blocking_front_linear | 0.182450 | 0.174600 | 0.538640 | 0.521166 |
| run_blocking_front_product | 0.182739 | 0.174766 | 0.539601 | 0.521471 |
| run_blocking_front_gap | 0.182958 | 0.174598 | 0.539874 | 0.521156 |
| explosive_pass_prevention_linear | 0.182643 | 0.174300 | 0.538796 | 0.520782 |
| explosive_pass_prevention_product | 0.182531 | 0.174475 | 0.538548 | 0.521096 |
| explosive_pass_prevention_gap | 0.182752 | 0.174276 | 0.538948 | 0.520739 |
| after_contact_tackling_linear | 0.182944 | 0.174914 | 0.539999 | 0.522480 |
| after_contact_tackling_product | 0.182642 | 0.174692 | 0.539270 | 0.521695 |
| after_contact_tackling_gap | 0.183031 | 0.174806 | 0.540078 | 0.522513 |

WAR's Brier change versus no in-season WAR is −0.000052; log loss changes +0.000301.
Accuracy increases from 72.40% to 72.60%, AUC is essentially unchanged, and ECE improves
from 1.50% to 1.26%. The confident error rate moves from 10.43% to 10.49%.
WAR hurts both proper scores in 2024 and helps both in 2025.

Paired season/week bootstrap 95% intervals are [−0.000954, +0.000890] for WAR's
Brier difference and [−0.002118, +0.002825] for its log-loss difference.
Both include zero. There are 31 season/week blocks but only two season blocks;
season resampling is too sparse to supply strong confidence. Weekly breakdowns,
calibration bins, balanced accuracy, entropy, fitted coefficients, selected penalties
and all paired comparisons are in the JSON artifact.

The all-pair interaction models lose to the simple WAR model and to their all-pair
linear control on pooled Brier and log loss. The targeted passing/coverage gap
improves on its matching linear control (ΔBrier −0.000249, Δlog loss −0.000805),
but intervals include zero and season-level performance is mixed.
The after-contact/tackling product improves on its matching linear control in
both seasons (pooled ΔBrier −0.000263, Δlog loss −0.000757). Its paired log-loss
interval [−0.001330, −0.000174] excludes zero, but its Brier interval includes zero
and it still loses to the simple WAR model overall. This is a research lead,
not a production winner. Twenty-five inspected candidates and uncorrected paired
intervals cannot establish confirmatory significance; cross-week team dependence
also remains. The targeted pair tests followed inspection of the combined-family
results, so they are explicitly exploratory.

## Production and X-Ray

No production model, team ratings, locked bets, or immutable prediction rows were
rewritten. Existing in-season WAR remains active provisionally; this experiment
does not establish its benefit. No new interactions were promoted. Scheduled
updates continue as configured.

The X-Ray WAR lab includes all 2,189 historical games and all 25 candidates.
For each of four members it shows the exact feature × fitted-coefficient log-odds
contributions. Their sum maps through the logistic function, and the four resulting
probabilities average to the displayed ensemble prediction. JSON chunks load only
for the selected season/week. JavaScript tests verify every published prediction
and swapped-side reciprocity. The original live X-Ray remains the active-model view.

The repository PFF secret is configured and was verified by successful Actions run
[38340063670](https://github.com/Mstolte02/cfb-model-v3/actions/runs/38340063670).
Credentials and licensed raw tables are private. Retrospective downloads are not
archived as-of-kickoff source snapshots; historical injuries were not reconstructed.
CFBD play-value facets lack player-window sources and are omitted from this update.

## Reproduce

With the recovered private source tree and Python dependencies installed:

```shell
python -m scripts.chronological_war_rebuild
python -m scripts.war_matchup_chronological
python -m scripts.report_chronological_war
python -m unittest tests.test_chronological_war tests.test_inseason_war tests.test_historical_ensemble_research tests.test_ensemble_blend_research tests.test_pff_carry_forward
node --test tests/test_market_board.cjs tests/test_power_xray.cjs tests/test_war_research.cjs
```

Private fit manifests and calibration artifacts live under `artifacts/chronological_war`.
Public results: [full metrics and provenance](output/chronological_war_matchups.json),
[game predictions](output/chronological_war_matchup_predictions.csv), and
[X-Ray](https://mstolte02.github.io/cfb-model-v3/power-xray/war-research.html).
