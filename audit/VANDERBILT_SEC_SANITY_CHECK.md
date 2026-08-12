# Vanderbilt / SEC 2026 sanity check

Checked 12-Aug-2026 after Vanderbilt's No. 25 rating and the SEC's concentration in
the top 25 were questioned.

## What drives Vanderbilt

Vanderbilt is not high because the model thinks the roster is continuous. The prior
season offense is No. 2 in FBS (`O = 2.089z`), while defense is No. 38, recruiting
talent No. 55, and projected all-roster WAR No. 30. Jared Curtis is a true freshman,
but he is also a five-star and the highest-rated Vanderbilt commit on record. The
five-source QB sheet has only two inputs for him and ranks him tied No. 54 of 124
named starters; the final 0.535 QB WAR is approximately a median-starter forecast,
not an elite-QB forecast.

The model's fitted Vanderbilt logit contributions are approximately:

| Component | Contribution |
|---|---:|
| prior offense | +0.569 |
| projected roster WAR | +0.255 |
| prior defense | +0.139 |
| recruiting talent | +0.042 |
| returning production | -0.023 |

That decomposition explains the result: the 2025 offense and a still-above-average
two-deep outweigh low continuity. The schedule simulation is less flattering than
the neutral power rank: No. 22 Vanderbilt averages 6.77 wins and 5.26 losses because
it plays an SEC schedule.

Official Vanderbilt sources confirm Curtis is a freshman and document his five-star
profile:

- https://vucommodores.com/sports/football/roster/season/2026
- https://vucommodores.com/sports/football/roster/player/jared-curtis-2

## Tests performed

### QB / passing continuity

CFBD reports Vanderbilt returning 35.4% of total PPA and only 8.7% of passing PPA.
Adding passing-PPA continuity to the selected player model did not validate:

| Strict expanding replay, 2022-25 | Pooled Brier |
|---|---:|
| selected core + projected WAR | 0.20695 |
| plus returning passing PPA | 0.20729 |
| plus returning passing PPA and offense interaction | 0.20723 |

The passing-continuity coefficient also changed sign across folds. It should not ship
merely because it gives one current team a more intuitive rank.

### Opponent adjustment / conference inflation

The v4 model was retested at several opponent-adjustment strengths while holding the
selected feature family fixed:

| Adjustment alpha | Strict pooled Brier | SEC top 25 | Vanderbilt rank |
|---:|---:|---:|---:|
| 0.00 | 0.20799 | 11 | 27 |
| 0.50 | 0.20764 | 11 | 27 |
| 0.75 | 0.20721 | 12 | 25 |
| **0.85** | **0.20695** | **12** | **25** |
| 1.00 | 0.20757 | 12 | 26 |

Removing schedule adjustment barely changes Vanderbilt or the conference count and
makes the historical replay worse. On interconference games in the strict replay,
the dynamic model underpredicted SEC teams by 6.7 percentage points on average; the
evidence does not support a conference-specific SEC haircut.

## Defect found and fixed

Historical `returning` rows use CFBD `percentPPA`, but the live 2026 frame had been
fed the separately defined Bill Connelly / ESPN number. The two sources correlate
only 0.30 across 135 matched teams. That was a train/serve feature-definition mismatch.
The model now uses the checked-in `data/returning_2026_cfbd.csv` snapshot; the Connelly
offense/defense splits remain descriptive data. This fix moves Vanderbilt to No. 22,
which is less intuitive but statistically coherent: a consistency bug should not be
retained because it happened to penalize one team.

## Conclusion

The Vanderbilt rank is aggressive, but the audit did not find evidence for a manual
freshman-QB or SEC penalty. The honest remaining limitation is uncertainty: the power
table reports a point estimate and does not communicate that a two-source true-
freshman QB projection should have a wider interval than a veteran's estimate. That
is a worthwhile future feature only after historical preseason QB-source coverage is
available to validate the interval calibration.
