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

## Accepted

**Safety versatility.** −0.67% side RMSE (t −3.7), −0.16% win% (t −2.7); odd seasons
slope −3.40 (t −3.2), even −3.89 (t −2.4). WAR a safety earns while moving between
deep, box and slot is worth less than WAR earned in one job. Applied in
`src/inseason_war.deployment_factor`: versatility clamped to the historical 5th-95th
percentile (.468-.800), slope shrunk by (1 - 1/t²) to −3.33, normalised so league safety
WAR is unchanged, applied to opening and current WAR alike so it never shows as a
change. 664 safeties move; middle 90% between ×0.55 and ×1.5.

## Rejected

Everything else, including wide vs slot WR, WR versatility, LT vs RT, centre vs guard,
slot CB, A-gap DT, outside-tackle edge, box/slot/on-line LB and box/deep/slot safety.
**IOL versatility** passed the first rule (−0.20% / −0.12%) but did not replicate
(odd seasons −3.9, even −67.4), so it is out. Several measures improved one outcome
and worsened the other (DT A-gap, EDGE outside-T, LB on-line).

## Availability

Unchanged in principle and now in the output: `base` (healthy WAR, if he plays) beside
`war` (counted now). The injury report shows both; Trending Down and the Top 10
fallers leave injured players out.

## Outputs

`viz/data/players_inseason.json`, per player: `war` (current expected), `base`
(healthy), `d` (change since opening), `sh` (2026 snap share), `al` (alignment shares,
position-appropriate: wide/slot/inline for WR and TE, line spot for OL, gap and
box/slot/deep/outside-CB for defenders; omitted for RB and QB, where PFF has no
backfield or pocket alignment), `new` and `g` for players off the preseason chart.
The player table has an Alignment column.

## Downstream

The team rating signal is built from per-snap rate changes and is not touched by
the safety factor, so power ratings and game probabilities are unchanged (checked:
`inseason_war_team_2026.json` identical, replay identical bar last-digit float noise).
Player tables, Top 10, unit rankings and team pages reflect the safety change and the
added players.
