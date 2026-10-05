# Player WAR upgrades, October 2026

Written 4 October 2026, after Mark asked for five things: actual 2026 snaps, no
role (tier) adjustment, grades adjusted for opponent and competition level, the
non-grade PFF measurements at every position, and sample-weighted in-season updating.
A sixth, game-by-game grades, is in its own section.

## Summary

| change | evidence | status |
|---|---|---|
| Count stats scored as rates, plus new PFF measurements | player WAR/snap year-to-year r .584 → .626; team rating vs next-season wins r .486 → .493 | shipped (production WAR rebuilt) |
| IOL pass protection kept as its own composite | it was being merged into run blocking by a name collision | shipped |
| No tier adjustment (`pooled`) | year-to-year r .626 → .649, better in 9 of 11 groups; team level unchanged | shipped |
| Opponent / competition adjustment | rest-of-season error −1.3% pooled, lower in all 11 groups (QB −3.2%, OT −2.6%) | shipped |
| Sample-weighted in-season update | −0.76% vs the fixed-weight baseline, interval [−1.25, −0.27]; better in 10 of 11 groups | shipped |
| 2026 snaps for playing time | observed snaps cut rest-of-season share error by two thirds | shipped in player WAR; not in the team rating signal |

Graded weeks are untouched: all 271 completed 2026 games keep the probabilities they
were graded on, and the Preseason-Week 4 rating history is identical. The new method
applies from the Week 6 slate (`stack_war_from_slate = 6`; in-season cuts before
`FROZEN_BEFORE_CUT = 5` keep their published values).

## 1. The facet catalogue (`war_model/candidates.py`, `concepts.py`)

An audit of the production build found:

- **Sixteen count statistics were scored as rates.** `facet_values` z-scored the raw
  column and multiplied by the denominator, so pressures allowed, sacks, stops,
  tackles, coverage yards, PBUs, first downs, touchdowns, avoided tackles and YAC
  credited volume twice where more is better and charged a player for playing at all
  where less is better. A `COUNT` kind now divides by the denominator first.
- **IOL had no pass-protection block.** `consolidate._name_for` gave the IOL pass-block
  cluster and the IOL run-block cluster the same name (`IOL_core`), and `apply()` summed
  them. Names are now unique.
- **The offensive line was ~75% grades.** Added: pass-block efficiency, all-snap PBWR,
  positive and negative graded-play rates (OL); PRP, all-snap win rate, TFL, batted
  passes, missed-tackle rate (DL); TFL and coverage efficiency (LB); targets per
  coverage snap, passer rating allowed and stop rate (S); drop and contested-catch rate
  and avoided tackles (TE); missed tackles forced, explosive runs, first downs, fumbles
  and YPRR (RB). `ED_qb_rating_against` (coverage passer rating divided by pass-rush
  snaps) was removed. All 144 candidates are now assigned a football concept.
- PBWR and the graded-play rates exist from 2019 only, like the true-pass-set PBWR the
  build already used.

| | before | after |
|---|---:|---:|
| player WAR per snap, year-to-year r (300+ snaps) | .584 | .626 |
| team Massey rating vs next-season wins | .4859 | .4926 |
| weighted facet total vs next-season wins | .5535 | .5510 |

## 2. Tier adjustment removed (`WAR_ROLE_NORM=pooled`)

The full production build was run both ways on the corrected catalogue.

| | team Massey r next season | player WAR/snap year-to-year r |
|---|---:|---:|
| partial (tier mean removed) | .4926 | .626 |
| pooled (no role adjustment) | .4938 | **.649** |

Pooled is more repeatable in CB, DT, EDGE, LB, OT, QB, RB, SAF and WR. A WR2 on a strong
team already outscores one on a weak team without any role term: 2025 WR2s average
.245 WAR on top-fifth teams and .043 on bottom-fifth teams.

## 3. Opponent and competition adjustment (`src/opponent_strength.py`)

Difficulty is the opponent's CFBD PPA: its defence for offensive players, its offence
for defensive players, z-scored and shrunk toward the mean by two games. FCS opponents
are in the same feed and come out easy, which is the competition-level part. The
effect on a player's rate is estimated within players (his early window against his
late window), so team quality cannot stand in for schedule strength.
`scripts/opponent_adjust_backtest.py`:

| group | slope | rest-of-season error |
|---|---:|---:|
| QB | −.060 | −3.21% |
| OT | −.013 | −2.58% |
| IOL | −.007 | −1.59% |
| LB | −.021 | −1.54% |
| EDGE | −.022 | −1.15% |
| RB | −.034 | −1.11% |
| DT | −.019 | −1.09% |
| TE | −.025 | −0.74% |
| CB | −.009 | −0.47% |
| WR | −.013 | −0.33% |
| SAF | −.005 | −0.33% |
| pooled | | **−1.32%** |

## 4. Sample-weighted in-season update (`src/inseason_war.updated_rate`)

Each player's gain is `K = P / (P + s2 / snaps)`: P is the variance of his prior,
small when years of snaps sit behind it, and `s2 / snaps` is the noise in this
season's rate. A player with a thin record moves quickly, a veteran slowly, and
everyone moves faster as 2026 snaps accumulate. Rerun on the rebuilt WAR with
opponent-adjusted rates (`scripts/war_inseason_backtest.py`):

| vs the fixed-weight baseline | pooled | 95% interval |
|---|---:|---|
| one-weight blend from the Kalman prior | −0.43% | |
| **sample-weighted** | **−0.76%** | [−1.25, −0.27] |

The first run showed the sample-weighted rule 5-8% worse on the offensive line. That
was the grid, not the idea: the nine-point `s2` grid could not reach the gain linemen
need. On a 49-point log grid it is better in 10 of 11 groups (WR +0.4%, interval
crossing zero). Plays and snaps are the same unit here, as Mark said.

## 5. Playing time from 2026 snaps (`src/inseason_war.playing_time`)

`scripts/playing_time_backtest.py` on PFF weekly snaps, 2022-25: last season's share
alone has weighted error 666; this season's snaps alone 225 (−66%); the Bayesian blend
215 (−68%). The prior is worth 0.5-1 game at every position; the live prior is the
projection, so `n0 = 1`.

A player's 2026 share is his fraction of his two-deep room's snaps times the room's
starter slots, the same definition the projection uses. Dividing by every player who
took a snap understated every starter by 25-40%; that was found and fixed before
shipping. Value of a share change is `proj_war / share` when the projected share is
at least .15, otherwise the player's prior per-snap rate calibrated to the projection.

**Not in the team rating signal.** Summed to a team, the playing-time change at week 5
runs positive on average and moves with blowouts and depth-chart rotation rather than
with anything a close game will show, and there is no history to validate it on (no
weekly-grain preseason projection exists before 2026). The team signal already counts
each player's actual 2026 snaps through the performance term. It is published as
`team_playing_time` in `viz/data/players_inseason.json` for reference.

## 6. Team model

The live ensemble's WAR stack was refitted on the rebuilt 2022-25 history. Nothing else
in the ensemble changed (members, base and PFF stacks, the team frame are identical).
The WAR coefficient roughly doubled (.075 → .16 per scaled unit): the model leans
on the new signal about twice as hard.

Forward test on 2,189 games of 2023-25 (`scripts/inseason_war_team_backtest.py`, rule
fitted on the other seasons): the WAR column against the same model without it is
**+.00018 Brier** (interval −.00046 to +.00079), against +.00004 for the previous
method. Both are neutral at the game level; the measured gains are in the player
numbers (sections 1-4). Shipped on Mark's standing instruction, with the number
recorded here.

## 7. Game-by-game grades

`scripts/game_level_war_backtest.py` pulled every PFF report one week at a time for
2022-26 (2023 week 14, championship week, returns empty from the v1 reports), scored
each game on the production facet path, adjusted each game for its own opponent, and
combined a player's games two ways. All arms use the shipped sample-weighted rule.

| vs cumulative weeks 1..N | pooled | by group |
|---|---:|---|
| game_equal (snap-weighted games) | +3.39% | worse in all 11 (EDGE +5.4, DT +4.9, QB +4.4) |
| game_recency (recent games weighted up) | +3.42% | no better than equal weighting |

A single game is too small a sample to score on its own: many measurements fall under
their opportunity floor, and a z-score among one week's players is noisy. Summing those
noisy games loses what the season-to-date report keeps. Production stays cumulative.
The weekly files remain staged for later work.
