# Accuracy recovery review: soft evidence, expanded WAR, and in-season form

## Executive decision

The fixed `.001` Brier threshold and a 95% interval excluding zero are retired as
universal feature vetoes. They remain useful descriptions of effect size and
uncertainty, but they should not force a weak, complementary signal to zero.

The replacement policy is:

1. reject leakage, target contamination, construction identities, and consistent
   out-of-sample degradation;
2. remove degenerate collinearity, but keep correlated inputs when their residual
   information improves forward loss and their coefficients are regularized;
3. compare fixed candidate families in outer forward folds;
4. shrink weak effects and average near-tied models instead of selecting a single
   winner by a hard threshold; and
5. report clustered uncertainty after the point estimate, using it to size confidence
   and rollout risk rather than to redefine an estimated improvement as zero.

This is not permission to add every plausible variable. It replaces binary
significance filtering with forward scoring, regularization, stability, and model
averaging.

## Repository-wide decision review

The earlier threshold did prevent several bad additions, but it also held back useful
signals. The distinction becomes clear when the experiments are grouped by failure
mode rather than by p-value.

| Prior experiment | Evidence | Revised treatment |
|---|---|---|
| Positional recruiting + rated portal | Large, stable forward gain | Already shipped |
| Reduced seven-feature preseason model | Better point estimate, far lower VIF and train/test gap | Already shipped; correct stability decision despite nonsignificant accuracy difference |
| Robust margin updater | Material forward gain | Already shipped |
| PFF API player facets | Better team signal; initial uncertainty crossed zero | Retain through expanded WAR and regularization |
| Production-informed WAR | −.00103 static, −.00030 online | Keep as a complementary ensemble member |
| Talent curvature | Strong preseason gain, weak after the old updater overwrote the prior | Retest in an architecture that retains the prior; it helps the final ensemble |
| Moving K | −.00048, uncertain | Do not zero it by rule; superseded by the more targeted team-specific gain work |
| Team-specific Kalman gain | −.00050; its team-specific component beat shared decay | Next update candidate; the useful part is team heterogeneity, not generic time decay |
| CFBD postgame expectancy | −.00034 over robust raw margin | Candidate observation channel, not a required dependency yet |
| Reversibility | Favorable in all outer folds before serving integration | Preserve as a guarded rolling-context candidate |
| Scripted-window form | Tiny and unstable after the live rating | Eligible only inside a regularized form ensemble |
| Full-season PFF team tables | Worse in every selection window | Reject as redundant aggregate predictors |
| Pure in-season statistical refit | Materially worse than score updates | Reject as a replacement; use stats only beside score evidence |
| Coach mean/change features | Consistently worse | Reject |
| Plain decision features | Worse or explained by talent curvature | Reject |
| Quick-pass/pressure matchup | Worse in all full-model folds | Reject current construction |
| Rank-normal/winsorized transforms | Consistently worse | Keep ordinary within-season z-scores |

This review confirms the user's concern: rated portal, the reduced feature basis,
player-production WAR, curvature, Kalman heterogeneity, and PFF facets all had useful
information that a universal materiality/significance gate could have discarded.

## Expanded WAR rebuild

The expanded WAR build adds 24 PFF API facets covering true-pass-set protection and
rush, gap/zone run blocking, run stops and tackle depth, forced incompletions, yards
per coverage snap, and coverage snaps per target. The full historical projection was
rebuilt before game testing; changing only the player file would have left the game
model on the old historical WAR column.

| Season | Current projected vs realized team WAR r | Expanded WAR r |
|---:|---:|---:|
| 2021 | .581 | **.607** |
| 2022 | .687 | **.749** |
| 2023 | .803 | **.824** |
| 2024 | .741 | **.801** |
| 2025 | .762 | **.783** |

Although the 2025 player-level correlation fell from `.592` to `.581`, player MAE
improved from `.0411` to `.0401`, and the aggregation target that the game model
actually consumes improved in all five seasons. The earlier rejection over-weighted
one diagnostic and under-weighted the downstream estimand.

## In-season design

`scripts/prior_decay_backtest.py` compares three information channels:

- the preseason level, using expanded WAR alone or beside the reduced preseason
  inputs;
- robust score innovation, separated from the preseason level so it is not counted
  twice; and
- current offense/defense form, opponent-adjusted and aggregated with an EWMA.

Every current-season transform—including its mean and standard deviation—is computed
only from games before the prediction week. This fixes a subtle problem in the older
refit experiment, which cut rows at the correct week but standardized them against
the completed season's distribution.

The prior-decay arm uses `0.5 ** (games_played / prior_halflife)`. Current form tests
2-, 4-, and 8-game half-lives plus a straight average. All half-lives, penalties, and
score learning rates are selected using earlier seasons only.

## Correction: bowl games were leaking into early-season form

The first version of this replay (the numbers first reported on 2026-09-22) had a
future-information leak. `raw_game_stats` kept every row of CFBD's
`/stats/game/advanced`, and CFBD numbers every bowl and playoff game **week 1 of the
postseason**. The week cutoff `week < W` therefore handed each team's bowl game, played
four months later, to every prediction from week 2 on. At week 2 a bowl team had two
"games" of form, one of them its bowl, which is also exactly when the two-game form
threshold first switches on.

The schedule itself was regular-season only, so the leak reached the EWMA form input
and nothing else: the preseason models, the score walk and the current-updater baseline
are unaffected. `raw_game_stats` now keeps `season_type == "regular"` only, and the live
form table (`data/live/game_advanced_2026.json`) is regular-season by construction. Every
number below is from the corrected run. The leak was worth about .0004 of the reported
gain: the ensemble went from .177448 to .177832, while the baseline did not move.

## Results

Pooled strict outer replay, 2023–25, 2,189 games. Lower Brier is better.

| Preseason source / in-season method | Brier |
|---|---:|
| Current full model, score update | .178986 |
| Expanded WAR alone, score update | .181856 |
| Full preseason model + expanded WAR, score update | .178902 |
| Full expanded model, EWMA replacing score evidence | .185254 |
| Full expanded model, decaying prior + EWMA + score innovation | .179479 |
| Full expanded model, retained prior + EWMA + score innovation | .178260 |
| + production-informed WAR | .178266 |
| + talent curvature | .178295 |
| + both production WAR and curvature | .178058 |
| **Equal average of the four expanded full-model variants** | **.177832** |
| CFBD pregame Elo, same games | .184959 |

The fixed four-member average beats the current score-updated model by **−.001154
Brier**. It improves all three outer seasons:

| Outer season | Games | Current score updater | Expanded equal ensemble |
|---:|---:|---:|---:|
| 2023 | 729 | .174299 | **.172646** |
| 2024 | 741 | .183941 | **.182796** |
| 2025 | 719 | .178633 | **.177975** |

The season-week block bootstrap puts the pooled difference at **[−.002409, +.000099]**,
with a 96.4% probability of improvement. Before the correction the interval excluded
zero; now it touches it. Under the policy at the top of this review that is a statement
of confidence, not a veto: the point estimate is negative in every outer season, and the
alternative is to keep a model the replay says is worse in all three.

Against aligned CFBD pregame Elo, the ensemble improves Brier by **−.007126**, with 95%
interval **[−.011529, −.003070]**.

Expanded WAR alone is not the better preseason model. In the retained-prior EWMA
architecture, the complete preseason model beats WAR alone by **.004728** Brier (95%
interval [.001133, .008145]). WAR is a strong summary variable; recruiting, returning
production, prior O/D, portal movement, and their nonlinearities retain complementary
information.

## What the decay test says

The user's EWMA intuition is partly right:

- recent efficiency helps when it is added beside score evidence;
- the selected current-form half-life lengthens as the training pool grows: four games
  for the 2023 fold, eight for 2024, a full-season mean for 2025 and for the production
  fit on 2021–25;
- pure EWMA/refitting is worse than score updates (.185254 against .178986); and
- deliberately decaying the whole preseason prior does **not** help (.179479).

The winning structure keeps the preseason level, adds the change generated by scores,
and adds opponent-adjusted offense/defense form. Evidence accumulates without an
arbitrary cutoff, but the prior remains an anchor rather than being forced toward zero.
The score innovations already move a team away from that anchor.

## Multicollinearity

The first experimental stack mistakenly included both the complete Elo level and the
preseason level. That duplicated the prior and produced VIF near 9. The final design
uses `elo_change = current_rating − week_0_rating` instead.

For the richest preseason candidate, the outer-fold condition number is `4.4–5.3` and
maximum VIF is `3.7–4.4`. The final in-season stack has maximum VIF `1.9–2.1`. Its
standardized coefficients for preseason level, score innovation, current offense, and
current defense are positive in every outer fold and in the production fit.

## Production decision: live from 2026-09-22

The expanded WAR replaced the legacy WAR in the production build, and the four-member
ensemble is the live in-season model.

**One runtime, one state.** `scripts/ensemble_replay.py` is a standard-library port of
the backtest's arithmetic, because the scheduled market capture runs on a bare GitHub
runner. `tests/test_ensemble_replay.py` holds it to the backtest's own functions: the
form builder matches `todate_od` and the replayed probabilities match `season_design`
plus the stack to 1e-12. `python -m scripts.train_live_ensemble --check 2025` goes
further, refitting the members on 2021–24 exactly as training does and replaying 2025
through the published block. With `--backtest-scaling` it reproduces the backtest's
2025 ensemble column to 1.1e-13 on all 719 games. Without it the one remaining
difference shows: the backtest swapped its expanded WAR column in standardised over
the FBS frame, while `build_frames` has always standardised `war_projected` over every
team with a CFBD talent score before narrowing to the frame. Production keeps its own
convention; on 2025 it scores .177918 against the backtest's .177975. The
live state is the `ensemble` block of `viz/data/model_v4.json`; every six hours the
capture pulls finals and `/stats/game/advanced`, replays the season from week 0, and
rewrites it. The browser (`viz/app.js`) and the CFP simulation
(`scripts/simulate_playoff.R`) read that block.

**Nothing already graded moves.** The v4 `logistic`/`margin`/`teams` blocks and
`dynamic.preseason_ratings` are left byte-for-byte alone: Week 1 is graded on the v4
season-fixed margin, and the futures board is held to the week-0 ratings. Every Market
board row already carries its own point-in-time `modelSnapshot`, so weeks 1–3 and the
week-4 board locked on 21 September keep the v4 numbers they were published with. The
ensemble first prices the board at the next Monday lock (week 5). The Heisman index,
the one futures panel that read `players.json`, now reads the preseason WAR of its 20
candidates from `viz/data/futures_preseason.json`, because `players.json` follows the
rebuilt WAR. Checked on the rendered site: every market-board week in both markets,
every Pick'Em export, all six Tracking views and all six futures panels are identical
before and after the migration.

What does change on the site is forward-looking by design: power ratings and their
week-by-week history (replayed under the ensemble), the matchup lab, team cards, the
playoff projection, and the player tables.

Reproduce with:

```powershell
python -m scripts.prior_decay_backtest          # the forward test
python -m scripts.train_live_ensemble           # fit the production members
python -m scripts.train_live_ensemble --check 2025
python -m scripts.publish_live_ensemble         # attach, pull finals and form, replay
```

Artifacts are `artifacts/prior_decay_backtest.json`,
`artifacts/prior_decay_backtest_predictions.csv`,
`artifacts/prior_decay_ensemble_predictions.csv` and `artifacts/live_ensemble.json`.

## Next PFF phase: personnel, alignment, and decisions

The API does not expose a general play-by-play or formation table. It does expose
enough adjacent information for a disciplined next phase:

1. **Personnel and role change:** weekly player position pivots give snaps by actual
   alignment, while team rosters provide depth, grades, ranks, and snap counts. Build
   weekly room-share changes and cross-position usage, not a guessed formation label.
2. **Scheme and opponent tendencies:** team tables provide play-action, screen,
   average depth of target, time to throw, zone/gap run shares, pass/run rates, and
   the corresponding opponent tendencies.
3. **Run direction and blocking fit:** the team rushing-direction report supplies
   rusher-by-gap results and totals. Cross those with line continuity and gap/zone
   blocking facets.
4. **Coaching decisions:** treat the tendencies above as observable decisions and
   residualize them against roster talent, game state, and opponent. This avoids the
   earlier PROE mistake where a talent nonlinearity looked like coaching skill.
5. **Form:** request weekly or bounded game ranges, cache every response, and create
   EWMA deltas. Do not use completed-season aggregates in a historical pregame row.

API call volume is the constraint. Start with FBS rosters and material rotation
players, pull deltas only for newly completed weeks, and validate one family at a
time before expanding to every player.
