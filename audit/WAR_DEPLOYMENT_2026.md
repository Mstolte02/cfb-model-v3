# Deployment-aware player WAR, October 2026

Written 5 October 2026 for Mark's deployment brief: observed role over assumed role,
performance within a deployment over alignment alone, scarcity learned not declared,
versatility only if the data says so, availability kept apart from performance.

## Role assumptions found and what happened to them

| where | assumption | now |
|---|---|---|
| `war_model/candidates.py` | depth-chart tier mean removed from every facet (`partial`) | removed earlier this month (`pooled`) |
| in-season player list | the preseason two-deep decided who was in the model | removed: any PFF player with 20+ 2026 snaps is in (Jesse Legree, Oregon State, was third string and missing) |
| in-season playing time | share measured against the two-deep only | measured against every player who took a snap in the room |
| `src/inseason_war.STARTERS` | starters per position (QB 1, WR 3, ...) | kept: a count of who is on the field, used to turn a room fraction into a share; no value attached |
| preseason `depth_correction.reweight` | chart template splits each room's WAR | kept for the frozen week-0 projection only; it is now just a one-game prior on snap share |
| position value | block weights | already learned from wins (`two_level_weights.py`); no hand multipliers anywhere |
| replacement level | one league replacement win rate, credit per snap | unchanged; not conditioned on deployment (see below) |

## The test

`scripts/deployment_war_backtest.py`. Every alignment measure available in the PFF
season files for 2014-25 (receiving wide/slot/inline snaps and route and block rates;
defensive box, slot, deep, outside-corner and A-gap/B-gap/over-tackle/outside-tackle
snaps; offensive-line LT/LG/C/RG/RT snaps), plus alignment entropy as versatility.

For group g and measure x the team term is sum of WAR_i * (x_i - mean x): a team is
credited only for WAR earned in a deployment, never for the deployment itself, so a
mediocre outside receiver adds nothing. Models with and without the term explain that
side's season PPA (CFBD; 1,052 offence or defence team-seasons) and schedule-adjusted
win% (1,438), leave one season out. Both models control for the team's pass rate,
because pass-block share and route rate are mostly play-calling: without the control
OT pass-block share looked like a -1.1% effect, with it +0.06%.

Acceptance rule, fixed before the results: better out of sample on both outcomes,
|t| of at least 2 on both, at least 0.2% on the side outcome, and the effect must
replicate in odd and even seasons.

| group | measure | side efficiency RMSE | win% RMSE |
|---|---|---|---|
| WR | wide | +0.02% (t +1.3) | +0.05% (t -0.6) |
| WR | slot | +0.03% (t -1.2) | +0.04% (t +0.7) |
| WR | rcv_versatility | +0.06% (t +0.8) | +0.04% (t +0.5) |
| TE | inline | +0.00% (t +1.3) | +0.03% (t +1.2) |
| TE | slot | -0.14% (t -1.9) | +0.05% (t -1.1) |
| TE | wide | -0.02% (t +1.0) | +0.04% (t -0.6) |
| TE | route_rate | +0.13% (t -0.5) | +0.01% (t -0.9) |
| TE | rcv_block_rate | +0.10% (t -0.0) | -0.01% (t +1.2) |
| TE | rcv_versatility | +0.03% (t +0.8) | +0.08% (t +1.0) |
| RB | route_rate | +0.20% (t -0.5) | +0.09% (t -0.1) |
| RB | rcv_block_rate | +0.21% (t +0.2) | +0.07% (t +0.4) |
| RB | slot | -0.02% (t -1.6) | +0.10% (t +0.6) |
| RB | wide | -0.42% (t +3.0) | +0.04% (t +1.1) |
| OT | left_tackle | +0.05% (t +0.8) | +0.04% (t +0.1) |
| OT | pass_block_share | +0.06% (t +0.1) | -0.17% (t +2.5) |
| OT | ol_versatility | -0.03% (t -1.8) | -0.01% (t -1.2) |
| IOL | center | +0.17% (t -0.7) | +0.11% (t +0.8) |
| IOL | pass_block_share | +0.06% (t +0.8) | -0.33% (t +3.2) |
| IOL | ol_versatility | -0.20% (t -2.5) | -0.12% (t -2.2) |
| DT | a_gap | -0.47% (t -3.5) | +0.03% (t -0.2) |
| DT | b_gap | +0.01% (t +1.9) | +0.01% (t -0.6) |
| DT | over_t | +0.16% (t -0.2) | -0.01% (t +1.4) |
| DT | rush_share | -0.11% (t -1.7) | +0.04% (t -1.2) |
| EDGE | outside_t | -0.39% (t +3.1) | +0.03% (t +2.1) |
| EDGE | over_t | -0.34% (t -2.9) | +0.23% (t -1.7) |
| EDGE | rush_share | +0.23% (t +0.6) | -0.04% (t -1.8) |
| EDGE | on_line | +0.12% (t -0.3) | +0.10% (t -0.1) |
| EDGE | def_versatility | +0.15% (t -0.1) | +0.05% (t +0.7) |
| LB | box | +0.14% (t +0.9) | +0.01% (t -1.3) |
| LB | dslot | -0.03% (t +1.1) | +0.05% (t +0.2) |
| LB | on_line | -0.24% (t -2.8) | -0.03% (t +1.6) |
| LB | rush_share | -0.06% (t -1.7) | -0.01% (t +1.1) |
| LB | def_versatility | +0.14% (t -0.4) | -0.16% (t +2.7) |
| CB | cb_slot | +0.09% (t +0.7) | +0.01% (t +0.6) |
| CB | box | +0.12% (t +0.1) | -0.14% (t -2.6) |
| CB | def_versatility | -0.03% (t +1.2) | +0.02% (t -1.1) |
| SAF | deep | -0.05% (t +1.9) | -0.02% (t +1.9) |
| SAF | box | +0.39% (t +0.6) | -0.14% (t -2.3) |
| SAF | dslot | -0.12% (t -2.3) | +0.04% (t +1.3) |
| SAF | def_versatility | -0.67% (t -3.7) | -0.16% (t -2.7) |

QB has no deployment measure in the staged files (no play-action, RPO, clean/pressure
splits); special teams are not part of this WAR. Both keep their existing model.

## Superseded: the first test was not evidence that deployment is irrelevant

Mark rejected this test's conclusions, correctly. It asked only whether team-season
PPA moved with WAR earned in a deployment, which has little power for one position's
alignment mix, used an unadjusted whole-side outcome, never measured production by
alignment, never measured replacement, and used pass/fail gates. Its one adopted
result (a safety-versatility factor of up to x0.55-x1.5) was removed. The work below
replaces it.

## Deployment-specific production and replacement (5 October 2026)

No manual multipliers, no significance gates: every effect is estimated with ridge
regression, penalty chosen by leave-one-season-out cross-validation, and enters at
its shrunk size.

**Data.** PFF single-week reports for every week of 2022-25 (alignment snaps and
production per game), every game scored on the production facet path and adjusted
for that game's opponent. CFBD per-game PPA, opponent-adjusted by what each
opponent allowed in its other games (`src/unit_outcomes.py`).

**Difficulty** (`scripts/deployment_production_backtest.py` on PFF job metrics,
`scripts/deployment_rate_backtest.py` in WAR units). Within the same player, how
production moves with his alignment mix. Graded by predicting each player's later
games from his earlier ones.

| job / group | within-player finding | later-game prediction error |
|---|---|---|
| SAF coverage | deep and box games allow ~1.0 and 0.5 fewer yds/cov snap than slot | −3.35% (job metric), −1.03% (WAR rate) |
| TE routes | inline-heavy games +0.65 YPRR, +6.6 route grade vs split out | −1.1% / −1.3% |
| LB pass rush | off-ball snaps win ~5.6 pts more often | −0.77% |
| CB coverage grade | outside games ~2.2 grade points higher than slot | −0.43% |
| OL pass pro | same tackle allows +0.6 pts pressure rate at RT; centres −0.8 vs guards | −0.33% |
| EDGE pass rush | outside-T alignment wins ~2.75 pts more often | −0.30% |
| WR wide vs slot | none (YPRR +0.03 ± 0.07, grade +0.27 ± 0.42) | +0.05% |
| DT gaps | small, noisy | ~0 |

**Replacement** (`scripts/deployment_rate_backtest.py`). In games a regular missed,
his opponent-adjusted rate minus that of the players at his position who played
instead, on his alignment shares; graded on held-out seasons.

| group | absences | effect | held-out error |
|---|---:|---|---:|
| TE | 1,515 | split-wide TEs much easier to replace | −0.50% |
| EDGE | 703 | off-ball and outside-T edges harder to replace | −0.41% |
| CB | 1,120 | slot corners slightly easier to replace | −0.25% |
| SAF | 943 | deep and box safeties harder to replace than slot | −0.16% |
| WR | 1,580 | ~none | −0.10% |
| OT / IOL / DT | 1,124 / 1,280 / 737 | none; ridge shrinks to ~0 | +0.10-0.18% |

**Unit value** (`scripts/deployment_value_backtest.py`, team-seasons 2021-25, opponent-
adjusted position-appropriate outcome). Shrunk estimates: EDGE WAR earned outside the
tackle worth more (−0.64% RMSE); CB WAR earned in the slot worth somewhat less
(−0.17%); WR, TE, OT, IOL, SAF, LB, DT shrink to ~0 (penalty at the top of the grid).
The game-level absence version of this test lacked power (one game's unit PPA
swings ±0.3-0.5 per play against ~0.02 WAR per missing player) and is not used.

**QB environment.** PFF passing-pressure reports 2014-25: a pressure-neutral grade
(clean and pressured grades at the league pressure mix) predicts next season worse
than the raw grade (r .390 vs .403). Pressure faced is partly the quarterback's own
doing, so no environment adjustment is applied.

## What ships

`src/inseason_war.deployment_shift`: per-snap shift = sum over alignments of
(replacement slope − difficulty slope) × (his 2026 share − league share), times
k × snap share × full-time snaps. Applied to opening and current WAR alike, so it is
never shown as change since opening. At week 5 it moves EDGE −0.017 to +0.030 WAR,
LB up to +0.018, SAF −0.004 to +0.009, TE ±0.006, everyone else within ±0.003.
WR, OT and IOL move essentially nothing because the data gives them nothing.

## Availability

`base` (healthy WAR) and `war` (counted now) per player; injury report and the
movers lists keep injury out of performance.

## Downstream

The team rating signal is built from per-snap rate changes and is unchanged
(`inseason_war_team_2026.json`, ratings and model identical). Player tables, Top 10,
unit rankings and team pages carry the shifted player WAR.
