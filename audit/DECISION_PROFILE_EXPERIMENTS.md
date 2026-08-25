# Play-call decision profile experiments (coach phase 6)

## Scope and status

`COACH_EFFECTS_EXPERIMENTS.md` closed phases 0–5 by rejecting every coach-derived
production feature, and listed what had not been tested: pass rate over expected,
fourth-down aggressiveness relative to situation, personnel packages, a unified
aggressiveness profile, decision×roster and decision×coach interactions, and whether
decision metrics mediate the estimated coach effect. This phase tests all of those
that the available data can express.

**Status: complete. No decision feature clears the predictive adoption bar; nothing
ships. The phase did surface an unrelated candidate — nonlinear talent — that beats
the shipping core on the static metric. It was tested as phase 7 in
`CURVATURE_EXPERIMENTS.md` and is also not adopted.**

## What could not be tested

Personnel-package usage (11/12/21 personnel), formation diversity and
package-specific efficiency are **not available in any source this project holds**.
CFBD's `/plays` carries down, distance, field position, score, clock and timeouts but
no personnel or formation charting; the PFF exports are season-aggregate player stats;
TruMedia is season-level team style. Testing that bullet requires a charting feed
(SIS or PFF Ultimate) that is not licensed here. It remains untested, not rejected.

## Data

`src/data/plays.py` adds a slim play cache: 628,193 regular-season plays over
2021–2025, reduced from CFBD's week-scoped `/plays` to the columns a decision depends
on. Raw weekly JSON stays in `data/raw` as a convenience cache only; the gzipped
per-season CSV is the reproducible artifact.

**Play-type vocabulary is not stable across seasons.** CFBD introduced
`Pass Completion` alongside `Pass Reception` and `Punt Return` alongside `Punt` in
2025 only. The first build silently discarded 1,431 real snaps from the holdout
season — 812 passes and 649 punts — because those labels were unknown. `plays.
audit_play_types` now fails when a season's unclassified share exceeds .004; the five
seasons sit between .00067 and .00091 after the fix.

## Construction

A league-wide logistic model predicts P(pass) on first through third down and
P(go for it) on fourth, from game state alone: down, log distance, distance×down,
yards to goal, goal-to-go, score margin, margin×urgency, a trailing flag, seconds
remaining, half and timeouts. A team's profile is its mean residual against that
expectation, so what survives is the part of the call the situation does not explain.

Fourth-down choices are counted only in the normal course — within 21 points and
outside the last five minutes — because down three scores with a minute left everyone
goes and up four scores everyone punts. Those rows measure the scoreboard. Fourth-down
rates are shrunk toward the league mean with a 25-play prior; early-down rates run to
hundreds of plays per team-season and are left alone.

## Question 1 — do the metrics repeat?

Lag-1 within-team correlation across 569 consecutive team-season pairs:

| metric | lag-1 r |
|---|---:|
| pass rate over expected | **.607** |
| raw early-down pass rate (the shipping style feature) | .597 |
| PROE, neutral script | .544 |
| PROE, trailing | .479 |
| PROE, late and close | .460 |
| PROE, leading | .454 |
| **fourth-down go rate over expected** | **.419** |
| fourth-down kick choice over expected | .188 |
| PROE game-to-game SD (consistency) | .143 |

Passing tendency and fourth-down aggression are real, repeatable traits. Decision
*consistency* is not — game-to-game variance in PROE does not carry to the next
season at all, so "this coach is predictable" is not a measurable property here.

## Question 2 — are they new information?

`corr(PROE, raw early-down pass rate) = .922`. The situation adjustment moves a team
by an SD of .032 against a PROE SD of .081, so it is doing real work — about 40% as
much spread as the metric itself — but PROE is mostly a rename of a rate the model
already had access to and never adopted.

Fourth-down aggression is the genuinely independent axis:
`corr(fourth_go_oe, PROE) = +.006` and `corr(fourth_go_oe, raw pass rate) = -.051`.

**There is no single "aggressiveness" dimension.** Passing tendency and fourth-down
aggression are orthogonal traits, so the unified-profile bullet is answered in the
negative: any construct averaging them destroys information rather than summarising it.

## Question 3 — do decisions mediate the coach effect?

Lagged decision tendencies added to the two-way HDFE decomposition, 522 team-seasons
over 2022–25:

| outcome | coach share without | with | relative change |
|---|---:|---:|---:|
| overall | .3311 | .3286 | −0.75% |
| offense | .3306 | .3337 | +0.93% |
| defense | .3321 | .3328 | +0.21% |

Decision tendencies explain **essentially none** of the apparent coach variance. The
coach effect estimated in phase 2 is not decision-shaped. Whatever it is, play-calling
tendency is not the channel.

## Question 4 — do they forecast games?

Same expanding selection, tuning, weekly replay and paired season-week bootstrap as
phases 3–4, against the same clean core and the same predeclared +.001 bar. Both
halves are refit inside every fold: the league expectation models see only seasons
≤ N−1, and a team's profile sees only its own seasons ≤ N−1. Decision coverage is
99.2% from 2022 on; 2021 has no prior plays, so all candidates are identical in the
2022 fold.

Pooled 2022–25. Negative is better.

| candidate | static Δ | static 95% CI | online Δ | online 95% CI |
|---|---:|---|---:|---|
| decision tendency | +.00035 | [−.00000, +.00068] | +.00016 | [−.00010, +.00044] |
| decision full | +.00115 | [+.00027, +.00208] | +.00049 | [−.00023, +.00125] |
| decision coach-carried | −.00048 | [−.00146, +.00052] | −.00037 | [−.00105, +.00028] |
| decision interactions | −.00126 | [−.00279, +.00029] | −.00112 | [−.00217, −.00011] |
| decision everything | −.00179 | [−.00370, +.00014] | −.00109 | [−.00223, +.00008] |
| **curvature core (no decision features)** | **−.00449** | **[−.00850, −.00086]** | −.00153 | [−.00407, +.00091] |
| curvature + decisions | −.00359 | [−.00788, +.00029] | −.00109 | [−.00384, +.00161] |

The plain decision features make the model **worse**, significantly so for the full
family. The interaction families looked like they improved it — until the confound
was controlled.

### The confound

`corr(PROE, talent) = +.338`. Talented teams pass more over expected. A `proe×talent`
term is therefore part quadratic talent, and the interaction candidates were taking
credit for curvature the core could not express.

`curvature_core` adds three terms — talent², returning², talent×returning — and **no
decision features at all**. It beats every decision candidate by a wide margin, and
adding decisions on top of it makes it worse (−.00359 against −.00449). The expanding
selection picks `curvature_core` in three of four folds; it never picks a decision
candidate once curvature is on the menu.

**The apparent decision effect was talent curvature.**

## Adoption verdict

**Reject every play-call decision feature.** They repeat, they are partly new, and
one axis is genuinely orthogonal to anything the model has — but they do not mediate
the coach effect, they do not forecast games, and their only apparent predictive value
was a nonlinear talent term wearing a decision costume.

## Spun-off candidate — nonlinear talent

`curvature_core` beats the shipping clean core by **−.00449 static** with a 95% CI
excluding zero, which is 4.5× the +.001 adoption bar, and is selected in three of four
expanding folds. That is the strongest result any coach-phase experiment has produced,
and it has nothing to do with coaches.

It was tested properly in `CURVATURE_EXPERIMENTS.md` (phase 7), which answered the
objection that it had only beaten a stunted baseline: curvature helps **all eight**
parent families on the static metric, including the granular and matchup families,
with every interval excluding zero. It still fails the online metric on both
production-selected families, so it is **not adopted**.

The split is mechanistic rather than noise. Curvature corrects the preseason prior;
the dynamic in-season update overwrites that prior once games are played. It is
therefore a candidate for the preseason products — futures, projected wins, playoff
odds — and phase 7 records what a phase 8 would have to test.

## Reproduction

```powershell
python -m scripts.decision_profile
python -m scripts.decision_predictive_backtest
```

Artifacts: `artifacts/decision_profile.json`, `artifacts/decision_profiles.csv`,
`artifacts/decision_predictive_backtest.json` and `.csv`.
