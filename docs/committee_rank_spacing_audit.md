# Prior-rank spacing experiment (2026-11-18)

Question: Does a nonlinear prior-rank feature improve next-release CFP predictions?
Each fold fits only earlier seasons. The 2020 pandemic season is excluded from fitting.
2018–23 are development years; 2024–25 are held out. The 2026 releases are
retrospective replays and have no role in selection. All variants use Ridge
`alpha=300`, the same as the shipped model, and a 0.25 rank-place head-to-head
adjustment in six-team pools.

| Prior-rank form | Development Top 25 MAE | Held-out Top 25 MAE | Held-out Top 12 overlap | Held-out top-12 H2H pair order |
| --- | ---: | ---: | ---: | ---: |
| Linear (shipped) | 1.326 | 1.284 | 11.4 / 12 | 74 / 81 |
| Piecewise hinges at 4 and 12 | 1.314 | 1.276 | 11.5 / 12 | 73 / 81 |
| Steps at 4 and 12 | 1.314 | 1.268 | 11.5 / 12 | not evaluated |
| Quadratic | 1.328 | 1.232 | 11.5 / 12 | not evaluated |
| Logarithmic | 1.317 | 1.256 | 11.4 / 12 | not evaluated |
| Hinges at 4, 12 and 25 | 1.414 | 1.452 | 11.5 / 12 | not evaluated |

The two-hinge form adds `max(0, 4 - prior_rank)` and
`max(0, prior_rank - 12)`. The final fit assigns a 0.693 predicted-rank-place
gap between prior Nos. 2 and 3, 0.903 between Nos. 11 and 12, and 0.705
between Nos. 12 and 13, holding every other feature fixed. Its 2026 replay
MAE is 1.78 versus 1.82 for linear; top-12 overlap is 12/12 in both.

**Decision:** Keep the linear model for now. The development and held-out
rank-error gains from the two-hinge form are small, while direct head-to-head
ordering deteriorates slightly (one of 177 development pairs and one of 81
held-out pairs). Selecting the quadratic based on its larger held-out gain
would use the holdout to choose the model despite worse development performance.
Head-to-head pairs repeat across weekly releases, so their counts are not
independent games. A future experiment should predefine a composite objective
that values both rank error and nearby head-to-head pair order and test it on
new releases.
