# Most Deserving v3: soft head-to-head, early-season prior, backtests

Status: production. Builds on [MOST_DESERVING_V2.md](MOST_DESERVING_V2.md); the
score itself (wins above a playoff-caliber reference team, in loss units) is
unchanged. This report covers six changes that were proposed, and what the data said
about each. Every number comes from `scripts/deserving_calibration.cjs` (raw output:
`audit/output/deserving_calibration.json`), which imports the production engine.

Data: CFBD regular-season FBS-v-FBS results 2014–2025, 2020 excluded. Outside
rankings and CFP games: `scripts/pull_deserving_benchmarks.py` (CFBD postseason games,
SP+, FPI with ESPN's résumé ranks). Selection-day committee rankings: the last
regular-season committee poll in CFBD's `rankings_YYYY.json`.

## Summary

| # | Proposal | Shipped? | Evidence |
|---|---|---|---|
| 1 | Head-to-head as a soft, capped adjustment | **Yes** | Same top-12 agreement with the committee; one more correct CFP team than with head-to-head off; no path dependence |
| 2 | Adaptive playoff-caliber band | **Built, tested, not adopted as rolling** | A 3- or 4-season window anticipates the band *worse* (mean miss 1.45 / 1.54 pts) than pooling every earlier season (1.27). The script re-selects the window at every recalibration; a live drift check is on the page |
| 3 | Backtest vs committee and outside rankings | **Yes** | Table in §3, also on the page |
| 4 | Schedule difficulty in the UI | **Yes** | Per-game reference expected losses, quality wins/losses, "Why is A ahead of B?" is the default view with every game's exact contribution plus head-to-head |
| 5 | Early-season stability | **Yes**, prior-season blend | Week 2–3 held-out log loss 0.573 → 0.505; zero weight from week 6 |
| 6 | Robustness checks | **Yes** | §6: window ±0.1, FCS, FBS definitions, vacated wins |

## 1. Head-to-head: a capped Bayesian update

v2 reordered teams after scoring: a head-to-head winner inside 0.39 jumped straight
above the team it beat. That was a binary leap at the band edge, and the result
depended on which pair the walk reached first. v3 adds a term to the score instead:

```
ranking score = résumé score + head-to-head adjustment
```

For each pair where w beat l (on balance), with résumé gap g = D_l − D_w and W = 0.39
(one SD of the measured score-gap noise, re-measured at 0.386):

```
prior on the true gap   Δ ~ N(D_w − D_l, W²)
evidence                P(w beat l | Δ) = Φ(Δ / W)
posterior mean shift    m(g) = (W/√2) · φ(z)/Φ(z),   z = −g / (W√2)
taper                   1 for g ≤ W/2, linear to 0 at g = W, 0 beyond
gap closes by           t = min(W, m(g) · taper(g));  winner +t/2, loser −t/2
team cap                net adjustment clipped to ±W/2
```

- A winner that trails by less than **0.24** moves ahead; further back, the gap only
  narrows. Outside the band, nothing. The function is continuous: no jumps.
- Confirming results (winner already ahead) move little: 0.16 at a 0.2 lead,
  0.02 at a 1.0 lead.
- **Why the cap matters.** v2 rejected a score bonus because, with nine head-to-head
  games inside its window, a team's bonus could shrink faster than a better result
  elsewhere raised its score. Here a team's head-to-head total is capped at ±0.195,
  so its swing is at most 0.39. A flipped loss is worth +1 and a flipped tie +0.5,
  so a better result always raises the ranking score (tested on live flips: at least
  1 − W). The résumé score keeps its strict guarantees (result sensitivity, margin
  restraint, leave-team-out) unchanged.
- **The cost, measured.** Because the adjustment is a score, a team can pass, or
  drop below, a team it never played. Across 2014–2025 final top 25s: 23 winners
  placed above the team they beat (v2: 32), 12 more gaps narrowed without reversing,
  and **72 top-25 pairs that never met reordered** (about 6–7 per season). The mean
  top-25 adjustment is 0.052. Only 1.8% of top-25 teams hit the cap. Live example
  (2026, week 10): Notre Dame (+0.143 for beating Miami) now also passes Alabama,
  which gave up 0.141 to Georgia.

Examples (résumé gap → ranking): 2017 Ohio State 11-2 over Wisconsin 12-1 (0.22 →
Ohio State +0.04); 2023 Texas 12-1 over Alabama 11-1 (0.20 → +0.06); 2025 Texas Tech
11-1 over BYU 10-2 (0.001 → +0.22).

Overrides (final top 25): 1-loss over unbeaten 19 of 83 (v2: 20), 2-loss over
unbeaten 4 (v2: 4), 3-loss over unbeaten 0.

## 2. The playoff-caliber band: adaptive, but measured

The proposal: pool only the last 3–4 seasons, so the band follows parity. The script
now scores each window by how well it anticipated the band each season ended with
(ranks 8–16, mean, 2018–2025):

| Window | Mean miss (points) |
|---|---|
| Last 3 seasons | 1.45 |
| Last 4 seasons | 1.54 |
| Last 6 seasons | 1.28 |
| **Every earlier season** | **1.27** |
| This season's own ratings after week 9 | 1.34 |

The band does not drift. It bounces (2019 16.4, 2021 11.6, 2023 15.4), so a short
window mostly adds noise. The script picks the smallest error at every recalibration
(today: every earlier season), so the band becomes rolling on its own if a trend ever
appears. The ranking barely depends on it: top 12 identical under every window,
top-25 Spearman ≥ 0.999.

The 20th–40th-percentile alternative was not built. That percentile range of the
full distribution is nowhere near playoff caliber, and the in-season estimate
above already covers the current-season idea.

**Drift check (on the page, never in a score).** The in-season fit is shrunk toward
average, so it reads low (about 1.3 points at week 9). The header therefore compares
this season's #8–#16 band with the same week's reading in past seasons. At 2026 week
10 it reads 0.4 points below, which is normal (past-season SD 1.5).

## 3. Backtest: committee, CFP games, outside rankings

Final regular-season rankings through conference championship weekend, put through
the format of the time: top 4, or from 2024 the five highest-ranked conference
champions plus seven at-large teams.

| Method | CFP field | Committee top 4 | Committee top 12 | Spearman, committee top 25 | CFP games: higher-ranked won |
|---|---|---|---|---|---|
| **Most Deserving v3** | **55 of 60** | 3.55 | 10.45 | 0.883 | 25 of 49 |
| Most Deserving v2 | 55 of 60 | 3.64 | 10.45 | 0.887 | 26 of 49 |
| v3 without head-to-head | 54 of 60 | 3.55 | 10.45 | 0.887 | 26 of 49 |
| Résumé rarity (Model B) | 54 of 60 | 3.45 | 10.27 | 0.883 | 25 of 49 |
| Colley (former production) | 54 of 60 | 3.45 | 10.55 | 0.892 | 32 of 49 |
| Power rating, end of regular season | 40 of 60 | 2.45 | 9.09 | 0.743 | 34 of 49 |
| ESPN Strength of Record (post-bowl) | 49 of 60 | 3.00 | 10.09 | 0.846 | 40 of 49* |
| SP+ (post-bowl) | 42 of 60 | 2.45 | 9.09 | 0.729 | 35 of 49* |
| ESPN FPI (post-bowl) | 41 of 60 | 2.27 | 9.00 | 0.759 | 37 of 49* |

\*CFBD only serves final, post-bowl values for SP+, FPI and SOR. They have seen the
playoff games, so their CFP-game column is contaminated. It is shown, not compared.

Reading it:

- **Selection.** Most Deserving picks 55 of 60 actual CFP teams, 6 more than ESPN's
  SOR and 15 more than a power rating. The misses are the committee's famous calls
  plus two bubble picks: 2017 Alabama (MD had Ohio State), 2018 Oklahoma (Georgia),
  2023 Alabama (Florida State 13-0), 2024 SMU (BYU), 2025 Alabama (BYU). v3 and v2
  tie on the field. v3 gains 2024 Tennessee and loses 2023 Alabama.
- **Later performance.** Conditional on making the playoff, the better résumé won
  25 of 49 CFP games, a coin flip, while the power rating won 34 of 49. That is
  expected and is the reason the systems are separate. Deservingness is not strength.
  Spearman between résumé rank and CFP wins is −0.07, against +0.20 for the power
  rating.
- **Committee agreement** is descriptive, not a target. v3 and v2 match the
  committee equally well.

Data fix found on the way: the cached `data/raw/committee_rankings.json` has 2023
wrong (Georgia #4, Alabama #5; Alabama was #4 and played Michigan in the Rose Bowl).
Every other season matches CFBD. The backtest now reads CFBD's final poll directly.
`scripts/fit_committee.py` still reads the cache and has not been refit.

## 4. The page

The public page is deliberately plain. The numbers, formulas and backtests live in this report, not on the site.

- **Most Deserving** opens on the Top 25. A **Résumé Lab** toggle swaps in the lab.
- **Why is A ahead of B?** gives a one-sentence verdict, then a step chart: start at
  team B's score, add each reason (losses, who they beat, who they lost to,
  head-to-head), end at team A's score. Below it, each team's games are listed with
  their values.
- **Résumé scores** shows rank, team, record, score and the likely range (80%, against a
  stronger or weaker playoff-caliber benchmark) on one shared axis.
- **Cutoff sensitivity** shows each team's best and worst rank across all 36 assumption
  combinations, with the official rank and the share of combinations in the top 12.
- **How it works** has five plain-language cards and links here.
- **Committee Watch** (separate tab) puts the committee's latest Top 25 next to the
  committee model's projection after the newest results. It is a browser port of the
  proxy and anchor in `scripts/simulate_playoff.R` / `committee_anchor.R`, tested
  against that run's stored scores (`tests/test_committee_watch.cjs`).

## 5. Early season: last season's ratings as a decaying anchor

Through the first weeks, the ridge pulls each opponent toward `regression × last
season's rating` instead of toward average, with weight `1 − (week − 1)/decay_weeks`.
Fitted on chronological out-of-sample log loss (train 2015–2019, 2021; 2021 uses
2019, since 2020 is excluded):

| Held out 2022–2025 | Games | No prior | Prior |
|---|---|---|---|
| Weeks 2–3 | 396 | 0.573 | **0.505** |
| Weeks 4–5 | 443 | 0.560 | **0.532** |
| Weeks 6–7 | 419 | 0.592 | 0.590 |
| Weeks 8+ | 1,548 | 0.559 | 0.559 |

Selected: regression 1.0, decay 5 weeks (zero from week 6). **Log loss alone wanted
14 weeks**, keeping last season in all year (train 0.5325 vs 0.5377). That was
refused on purpose: a résumé should be built on this season's results, so the decay
is capped at 5 weeks, matching the proposal. The ranking effect is modest. Top-12
overlap with the final top 12 at week 3 goes 4.75 → 5.0, and at week 5 7.75 → 8.0;
weekly top-25 movement at week 3 goes 13.0 → 12.2. Early rankings still move a lot.
That is honest, and the page says so.

## 6. Robustness

| Check | Top-12 overlap with production | Top-25 Spearman |
|---|---|---|
| Head-to-head band 0.29 / 0.34 / 0.44 / 0.49 | 12 / 12 / 12 / 11.91 | ≥ 0.997 |
| Head-to-head off | 11.73 | 0.990 |
| Opponents that left FBS the next season (UAB 2014, Idaho 2017) excluded | 12 | 1.000 |
| First- and second-year FBS teams treated as non-FBS | 12 | 0.968 |
| FCS games counted (all FCS opponents pooled into one team) | 11.27 | 0.970 |

- **The 0.39 band is not knife-edge.** ±0.1 changes one top-12 spot in 11 seasons.
- **FCS is the one that matters.** Production still ignores FCS games, so an FCS loss
  costs nothing. Counted, 2016 Iowa (lost to North Dakota State) falls #16 → #31 and
  2016 Washington State #19 → #32. Most other changes are #12 ↔ #13. Recommended
  next step, not done here.
- **Vacated wins.** Removing a top-12 team's best win from its own record (the loser
  keeps the loss, as the NCAA does) drops that team 3.2 places on average (max 10).
  Other top-25 teams barely notice: 0.18 per case move more than one place.

## Unchanged

Opponent model (λ 2, cap 60, home 3.0, σ 15.05), loss weight 1, reference centre #12,
convincingness 0, rarity (Model B), schedule fairness at the cutoff (0.079 loss
units), simulation override accuracy (76.5%), and no leakage from future results
(0 of 8 snapshots).

## Reproduce

```
python3 scripts/pull_deserving_benchmarks.py          # once; needs CFBD_API_KEY
node scripts/deserving_calibration.cjs                 # about 3 minutes
node --test tests/test_most_deserving.cjs
```

The v2 engine is frozen at `audit/legacy/ranking-history-v2.cjs` for the comparison
rows. The site does not load it.
