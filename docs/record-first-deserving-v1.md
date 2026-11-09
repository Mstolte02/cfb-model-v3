# Record-first Most Deserving v1 (experimental)

This implements the earned-results ranking in `viz/ranking-history.js`.
It is separate from the predictive v5.1 model and the committee-projection simulator.

## Rule

For each FBS team (FBS regular-season opponents only):

`deserving = -FBS_losses + quality_adjustment`

`quality_adjustment = 4.0 * (opponent_Colley_mean - 0.5) + 0.30 * mean(tanh(clamp(game_margin,-28,28)/21)) + 0.16 * FBS_wins/12`

Opponents' Colley values are fitted jointly from published completed FBS games, with two virtual 0.5 games per team. This is **schedule quality**, not a second loss/win bonus.
The margin term is deliberately small and saturating; FCS games and postseason
games are excluded. Conference membership and predictive power are not inputs.

The score is a scalar, not a pairwise comparison function. When team A has one more
loss than team B, A can pass B **if and only if** A's quality adjustment exceeds B's
by more than 1.0. Two additional losses require more than 2.0, etc.
The ranking is transitive and insensitive to iteration order.

## Important caveat

**These coefficients are provisional, not simulation-calibrated.** They encode the
user's record-first philosophy, but should not be represented as learned or
historically validated. A 0.25 opponent-Colley difference cancels one loss before
margin/extra-win adjustments. The empirical calibration step remains to be done.

## Proposed calibration

1. Freeze each historical week as of its actual date; compute opponent ratings
   using games completed by then, never end-of-season opponents' ratings.
2. Generate schedule-swap and outcome counterfactuals using a *separate*,
   chronological, calibrated game simulation model. Vary strength, schedule,
   and volatility; never optimize on current-season future results.
3. Test monotonicity: improving the result of a game must not hurt a team's
   rank under fixed other results, without creating a circular dependency.
4. Evaluate parameter grids with season-blocked validation, rankings stability,
   and expert-reviewed anchor cases. Because 'deserving' is normative, predictive
   log loss **cannot** determine the uniquely correct weights.
5. Publish sensitivity bands (rank ranges across plausible coefficients) and
   examples of 0-loss/1-loss/2-loss overrides with exact decompositions.

## Integration

`RankingHistory.deserving(schedule,metadata,teamNames,{method:'colley'})`
still works for existing callers, but now returns loss-equivalent scores.
`{method:'record-first'}` is also accepted. Consumers may continue to read
`score`, `rank`, `recordBase`, and `scheduleAdjustment`, with added
`marginAdjustment`, `winsAdjustment`, and `qualityAdjustment`.

**Do not assume the existing committee-anchored projected scores have been
replaced.** Those come from the separate R simulation/committee pipeline.
Historical committee comparisons need explicit version labeling once this
ranking has been reviewed and released.

Run `node tests/record_first_deserving.test.cjs` for core invariants.
