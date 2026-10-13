# PFF carry-forward, ensemble blends and opposing-unit audit

October 11, 2026. Exploratory research, not a model promotion.

## Operational fix

PFF now uses the newest nonempty cutoff strictly before the prediction slate.
Week 7 currently uses data through Week 5 until Week 6 arrives. This keeps both PFF
and in-season WAR active. It never reads same-week or future-week tables. The
published policy boundary is slate 7: previously graded slates keep their original
lookup behavior. All 329 replayed historical game predictions and frozen betting
files were verified unchanged. The explorer displays the actual PFF cutoff and
whether it was carried forward.

## What was tested

The first two weeks (100 games) train the initial candidate models. Weeks 3–6
supply 229 matched test games. Each held-out week trains on strictly earlier weeks;
all candidates receive the same games. The four pretrained member models remain
fixed. This is a chronological meta-model experiment, not a historical refit of
the complete preseason pipelines. Penalties were fixed before scoring, without
using test-week labels. No candidate is automatically promoted.

| Candidate | Brier ↓ | Change vs equal ↓ | Log loss ↓ | Weeks better |
|---|---:|---:|---:|---:|
| Current equal probability average | 0.179253 | +0.000000 | 0.530333 | 0/4 |
| Equal logit average → one probability | 0.179214 | -0.000039 | 0.530151 | 3/4 |
| Learned probability blend | 0.178316 | -0.000937 | 0.527659 | 4/4 |
| One fitted model of four member logits | 0.180343 | +0.001090 | 0.532264 | 2/4 |
| One fitted model of combined live inputs | 0.176752 | -0.002501 | 0.526685 | 2/4 |
| Logit stack + nonlinear O/D gap | 0.181627 | +0.002374 | 0.536324 | 2/4 |
| Logit stack + opposing O×D interaction | 0.175199 | -0.004054 | 0.518946 | 4/4 |
| Combined inputs + nonlinear O/D gap | 0.175834 | -0.003419 | 0.523116 | 2/4 |
| Combined inputs + opposing O×D interaction | 0.175792 | -0.003461 | 0.523579 | 2/4 |
| Single member: full_new_war | 0.177563 | -0.001690 | 0.525488 | 3/4 |
| Single member: full_new_war_curvature | 0.177933 | -0.001320 | 0.526542 | 3/4 |
| Single member: full_new_war_production | 0.181360 | +0.002107 | 0.536090 | 0/4 |
| Single member: full_new_war_curvature_production | 0.181160 | +0.001907 | 0.535355 | 1/4 |

The learned blend improves in all four evaluated weeks; its weights move away from
the production-feature variants. For the final held-out week its weights are
40.6%, 32.7%, 15.0%, 11.7% (base, curvature, production, both). Those are diagnostic
weights fitted through Week 5, not new production weights.

The combined-input model improves overall but only in two of four weeks. The
strongest point estimate here is the logit stack with an opposing O×D product,
which improves in all four weeks. The member probability correlations range from
0.9929 to 0.9973, so the ensemble's members are very similar.

## What “one model” means here

The logit stack feeds four individual member logits into one regularized logistic
model and applies one final sigmoid. The combined-input model instead fits a
single logistic model to member-specific preseason differences, result-update
differences, active form inputs, PFF, WAR, and home field. Repeated inputs retain
their per-member columns; regularization is part of the stated experiment. Neither
can reconstruct the original preseason feature union from the public payload,
which stores the preseason scores as aggregates. The recovered original team
frames now support the fuller union comparison in HISTORICAL_ENSEMBLE_RESEARCH.md.

## Is production a matchup model?

Production is predominantly an additive team-strength-difference model. With
positive defense values representing better defense, its efficiency terms are

`bO * (O_A - O_B) + bD * (D_A - D_B)`.

This can be rearranged into two opposing-side advantages:

`(bO * O_A - bD * D_B) - (bO * O_B - bD * D_A)`.

Thus renaming the comparisons “offense vs defense” alone adds no new information.
What is absent in the live stack is an explicit interaction that lets an offense's
impact depend nonlinearly on the specific defense. The four-member average also
introduces a nonlinear aggregation, but it is not an explicit O×D matchup term.
The existing UI's position matchup displays do not establish that those crosses
enter the live probability.

Two antisymmetric features were tested:

- Nonlinear gap: `g(O_A - D_B) - g(O_B - D_A)`, where `g(x)=x*abs(x)`.
- Opposing product: `O_A*D_B - O_B*D_A`.

Offense uses PFF when available, otherwise CFBD form; defense uses CFBD form.
Both are standardized composites, not common-unit estimates of points. Missing
coverage or fewer than two form games zeros the interaction. The signals are
averaged over members; their four current form series happen to be identical.
Swapping teams negates each input and preserves complementary probabilities.

At the current cutoff, Notre Dame has PFF offense 1.599 and CFBD defense 2.187;
Alabama has PFF offense 1.881 and CFBD defense 1.254. These are separate standardized
scales. The linear live terms reward Alabama's offensive edge and Notre Dame's
defensive edge; the tested cross terms additionally allow that pairing itself to
matter. They are research-only.

## Limits and next validation

This is one partial season, four held-out week clusters, and multiple candidates.
The PFF/WAR cutoffs include later retrievals, so chronological outcome splitting
does not make this a prospective point-in-time backtest. The interval estimates in
the JSON resample weeks; four clusters cannot support strong significance claims.
The original run could not access the historical inputs. On October 12 the actual
local source tree was recovered; see HISTORICAL_ENSEMBLE_RESEARCH.md for the newer
multi-season comparison and the remaining upstream chronology limitations.
Older checked-in v3/v4 prediction CSVs are not a substitute for this ensemble.

Keep the existing production combination while the candidates are shadowed or
rerun on restored multi-season forward predictions. Do not declare a winner from
this sample. A full union-of-preseason-features comparison still requires those
training inputs; the current result concerns the available live inputs only.

## Reproduce

`python3 -m scripts.ensemble_blend_research`

Outputs: `audit/output/ensemble_blend_2026.json` and
`audit/output/ensemble_blend_2026_predictions.csv`. The JSON includes every fold,
weights, probability correlations, metrics, and exploratory week-bootstrap intervals.

Checks: `python3 -m unittest tests.test_pff_carry_forward tests.test_ensemble_blend_research`
and `node --test tests/test_power_xray.cjs tests/test_market_board.cjs`.
