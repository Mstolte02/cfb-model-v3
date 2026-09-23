# Sample size in player WAR updates

Written 23 September 2026. The question came from Mark: a player we know a lot about
should be harder to move off his prior than a player we know little about. The
alternative on the table was a recency weight that treats every player the same.

## Summary

| question | answer |
|---|---|
| Across seasons, should a season's weight grow with its snaps? | **Yes.** −1.45% error against fixed year weights, 95% interval excludes zero, better in all five test seasons and in 10 of 11 position groups. |
| Is the gain from the snap term itself? | **Yes.** The same filter with snaps ignored is +0.20%, no better than fixed weights. |
| Where is the gain largest? | For players with the most history: **−3.8%** at 1,600+ prior snaps. |
| In season, should the update also weight this season's weeks by snaps? | **No measurable gain.** Once the prior is built with sample size, a single blend weight per position and week does as well (+0.01%, interval −0.28% to +0.30%). |
| What should an in-season player WAR use? | The sample-size prior, blended with season-to-date WAR by one weight per position and week. |
| Does the recency weight help? | Not on either data set. On CFBD per-game data it was no better than the sample-size prior. |

All numbers below are on PFF WAR for every position, including the offensive line and
all defenders, except the one section that says it uses CFBD.

## 1. Across seasons

`scripts/war_credibility_backtest.py`. It predicts a player's season N WAA per 1,000
snaps from his earlier seasons in `war_model/hybrid_player_war.csv`. Parameters are fit
per position group on seasons before N only. It grades players with 100+ snaps in N and
at least one earlier season, and weights the error by season-N snaps.

- **fixed**: one weight each for N−1, N−2 and N−3. Every player gets the same weights.
- **kalman**: a per-player filter. The true value drifts between seasons. Each season
  is a noisy look at it, with noise `sigma2 / snaps + omega2`. A season with more snaps
  counts for more. A player with a long record has a tight prior.
- **kalman_flat**: the same filter with `sigma2 = 0`, so snaps do not matter.

| error vs fixed | kalman_flat | kalman |
|---|---:|---:|
| pooled 2021-25, n = 17,807 | +0.20% | **−1.45%** |
| 2021 / 2022 / 2023 / 2024 / 2025 | | −1.48 / −1.64 / −1.61 / −1.29 / −1.23% |
| prior snaps < 300 | +0.61% | −1.33% |
| 300-800 | −0.15% | −1.16% |
| 800-1600 | +0.18% | −1.82% |
| 1600+ | −0.80% | **−3.80%** |

95% intervals on snap-weighted MSE, bootstrap over players: kalman − fixed
[−.00111, −.00046]; kalman − kalman_flat [−.00124, −.00058]. Both exclude zero.
Correlation with the actual season: .303 fixed, .318 kalman.

By group, kalman against fixed: OT −2.21%, IOL −2.32%, CB −2.04%, LB −1.75%, TE −1.75%,
QB −1.62%, DT −1.61%, EDGE −1.36%, SAF −0.98%, RB −0.92%, **WR +0.14%**.

The fitted parameters say why. For QB, CB and safety, the noise that does not depend on
snaps (`omega2`) fits to zero, so snaps explain nearly all of how much a season should
count. For receivers it is the other way round: `sigma2` fits to zero. Snaps are a poor
measure of how involved a receiver was; targets or routes would be a better one.

**Limit.** This beats a linear fixed-weight baseline. It has not been tested against
the shipped preseason projection (`war_model/project_2026_v2.py`), which is a boosted
tree that already sees `snaps_lag1` and `snaps_lag2`. The next step is to give that
model the filter's mean and variance as features and compare holdout error. Only that
test can decide whether the preseason WAR changes.

## 2. In season, on PFF WAR at every position

`scripts/war_inseason_backtest.py`. PFF WAR is a full-season number, so it had to be
rebuilt for windows of weeks.

**Data.** `scripts/sync_pff_war_windows.py` pulled the ten reports the WAR build reads
(five v1 facet reports, five v2 position-report families) for weeks 1-3, 4-16, 1-6 and
7-16 of 2022-25: 160 requests. The v1 endpoints take `week=1,2,3`; the range form
`1-3` returns HTTP 500. The v2 endpoints take `week` and `weekTo`. The upstream also
returns occasional HTTP 500s on heavy v2 reports; the script retries with backoff.

**Method.** `src/war_window.py` runs the production facet path on a window: the same
merge, the same facet catalogue and role-relative z, the consolidated composites and the
production weights and sigma, held fixed. It reproduces the production per-player
`f_contrib` on full seasons at r = .994-.999 (2016, 2019, 2022, 2024, 2025). Two gaps:
the 12 CFBD facets have no weekly source and are left out, and the Massey schedule
share is not applied. So the unit is "facet WAR" before schedule allocation, and the
prior, window and target all use it.

One trap found on the way: `candidates.facet_values` drops a facet with fewer than 200
qualifying player-seasons. A three-week window has about 110-220 qualifying QBs, so the
QB facets disappeared without an error and weeks 1-3 correlated .17 with the full season
for QBs. `war_window` now pads the window with a full season under another season label.
Z is taken per season, so the padding does not change the window. QB rose to .66.

**Arms.** The prior is the section-1 filter run over full seasons. Every arm gets the
same linear calibration. Every free parameter is fit per position and week on earlier
seasons only, so 2022 is training and 2023-25 are graded. The target is the
rest-of-season window, weighted by its snaps.

- **same_for_all**: `(1 − λ) · fixed prior + λ · season to date`, one λ per position and week.
- **same_for_all_kalman**: the same blend from the filter's prior mean.
- **bayes**: precision weights. Prior variance from the filter, window variance
  `s2 / snaps`. Both halves depend on sample size.
- **bayes_flat_prior**, **bayes_flat_window**: bayes with one half made the same for all.

| error vs same_for_all, n = 25,833 | week 3 | week 6 | pooled |
|---|---:|---:|---:|
| prior only | +2.49% | +4.89% | +3.49% |
| season to date only | +4.03% | +1.88% | +3.13% |
| same_for_all_kalman | −0.54% | −0.53% | **−0.53%** |
| bayes | −0.54% | −0.51% | **−0.53%** |
| bayes_flat_prior | −0.49% | −0.38% | −0.45% |
| bayes_flat_window | −0.57% | −0.21% | −0.42% |

95% intervals, as % of the second arm's error: bayes vs same_for_all [−0.98, −0.08];
**bayes vs same_for_all_kalman [−0.28, +0.30]**; bayes vs flat prior [−0.21, +0.05];
bayes vs flat window [−0.38, +0.15].

So the in-season gain is real, and it comes entirely from the better prior. Weighting the
season's weeks by snaps on top of that adds nothing that can be measured at week 3 or 6.
By prior depth, both filter-based arms gain most on the most-known players (−3.8% at
1,600+ snaps) and nothing on newcomers.

By group, bayes against same_for_all:

| group | n | bayes | same_for_all_kalman | bayes 95% |
|---|---:|---:|---:|---|
| QB | 926 | −1.74% | −0.97% | [−3.85, +0.34] |
| TE | 1,237 | −1.27% | −0.29% | [−3.10, +0.49] |
| LB | 2,672 | −1.22% | −1.04% | [−2.50, +0.10] |
| DT | 3,031 | −0.80% | −0.51% | [−1.94, +0.34] |
| CB | 2,899 | −0.65% | −0.62% | [−1.51, +0.18] |
| SAF | 2,856 | −0.34% | −0.36% | [−1.12, +0.44] |
| RB | 1,571 | −0.23% | −0.29% | [−1.86, +0.94] |
| OT | 1,914 | +0.15% | −0.06% | [−1.77, +1.91] |
| WR | 2,758 | +0.32% | +0.21% | [−0.43, +1.07] |
| EDGE | 3,087 | **+1.36%** | −0.04% | [+0.19, +2.55] |
| IOL | 2,882 | **+2.99%** | −0.81% | [+0.86, +5.16] |

The full Bayes update is significantly worse for EDGE and IOL. The one-λ blend from the
filter prior is never meaningfully worse than the baseline in any group. That makes the
blend the safer rule to ship.

## 3. The CFBD check, and why it is not the answer

`scripts/player_prior_sample_size.py` first tested the idea on the older in-season
harness, which uses CFBD per-game PPA. That feed has only QB, RB, WR and TE, and no play
counts, so it is not the WAR the site shows. Kept for the record:

- The old harness graded against a rest-of-season value that was itself shrunk toward
  zero, which rewards whichever arm shrinks hardest. The new script uses a nearly
  unshrunk target (alpha 0.1), weighted by games.
- Against the best single prior strength, chosen on earlier seasons, a prior whose
  strength grows with last season's games (or PFF snaps, joined by name at 90-94%)
  does not help WR (−0.34%, interval crosses zero), RB or TE. QB gains only in weeks
  1-2; after that the league-mean shrink wins.
- The recency weight did no better than the sample-size prior.
- `src/qbwar.fit_season_values` gained `strength` (per-player prior stiffness) and
  `sample_weight`. Both default to None, which leaves the fit exactly as before.

## What to do next

1. **Build the in-season player WAR on this rule.** Prior = the filter over full seasons;
   update = `(1 − λ) · prior + λ · season to date`, λ per position and week. The fitted
   λ values are in `artifacts/war_inseason_backtest.json` under `chosen`. This is a
   display change: the live team model does not read in-season player WAR
   (`dynamic_blend = 1`).
2. **Test the filter against the shipped preseason projection** before changing
   preseason WAR (section 1, Limit).
3. **For receivers, measure depth in routes or targets, not snaps.**
4. **Add the CFBD facets to the window build** if a weekly source is wired in. They are
   the one part of production WAR the windows do not cover.

## Reproduction

```powershell
python -m scripts.war_credibility_backtest
python -m scripts.sync_pff_war_windows --seasons 2022-2025 --windows 1-3,4-16,1-6,7-16
python -m scripts.war_inseason_backtest
python -m scripts.player_prior_sample_size
```

The sync needs `PFF_API_KEY`. The quota is about 100 requests per short rolling window.
Files land in `source-data/pff_api/war_windows/`.
