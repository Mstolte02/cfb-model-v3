# Idempotence-shaped rating experiments

Date: 2026-09-02

## Decision

**Promote the recursive correction to a full-v4 candidate, but do not put it into
production Elo yet. Promote second-pass opponent lift to a Most Deserving candidate.**

The predictive result is real enough to warrant the model-native test: a recursive
correction beside a clean weekly Elo improved eight-season Brier by 0.00082, improved
seven of eight outer seasons, and had a paired season-week bootstrap interval wholly
below zero. A broader persistence-plus-correction family cleared the repository's
0.001 adoption threshold, but failed in two seasons. The present experiment starts
Elo at 1500 rather than from v4's preseason projection, so it cannot establish that
the signal survives the shipping prior and dynamic update.

For committee ranking, literal persistence and generic instability both made the
ranking worse. The useful version was much simpler: **how much the second opponent
pass lifts or lowers a team's first-pass result**. It improved leave-one-season-out
Spearman by 0.00601 beside the production-style base, including head-to-head; it
helped in eight of eleven seasons and remained +0.00223 after deleting whichever
single season was most favorable. Network-wide disruption looked good before
head-to-head entered, but did not survive the production-style control.

## Definition

The production opponent adjustment in `src/oppadj.py` already solves an exact fixed
point. Applying that completed function a second time would return the same result by
construction and give every team a zero residual. The test therefore measures the
*path* to the fixed point from completed game margins:

1. **Pass 1** is average capped, home-field-adjusted scoring margin.
2. **Pass 2** adds the average pass-1 rating of the opponents played.
3. **Fixed point** is the regularized Massey/SRS solution on those same games.

The predictive candidate is `fixed point - pass 1`, expressed as the home-away gap
before kickoff. The resume candidate is `pass 2 - pass 1`, standardized within the
season. “Stable strength” is the signed magnitude shared by pass 1 and the fixed
point, and is zero when the two disagree on direction. “Disruption” is the signed
global L1 movement of the rating vector under an exact leave-one-game-out ridge
influence calculation.

## Predictive test

For every test season from 2018 through 2025, the model trains on all earlier seasons.
Games within a week are predicted together before any result from that week updates
Elo or the recursive graph. The comparison recalibrates Elo in every fold, so the
extended model does not win merely by getting an intercept or probability slope that
the baseline was denied. There are 5,705 held-out FBS-vs-FBS games and 123
season-week bootstrap blocks.

| Candidate beside Elo | Brier | Change vs Elo | 95% block-bootstrap interval | Outer seasons improved |
|---|---:|---:|---:|---:|
| Recalibrated Elo | 0.20084 | — | — | — |
| Pass-1 strength | 0.20067 | -0.00017 | [-0.00079, +0.00042] | 4 / 8 |
| Pass-2 correction | 0.20037 | -0.00047 | [-0.00088, -0.00007] | 6 / 8 |
| **Recursive correction** | **0.20002** | **-0.00082** | **[-0.00148, -0.00019]** | **7 / 8** |
| Stable strength | 0.19975 | -0.00109 | [-0.00217, -0.00006] | 5 / 8 |
| Recursive correction + stable strength | **0.19898** | **-0.00186** | **[-0.00317, -0.00056]** | **6 / 8** |

The two-column family has the largest pooled gain, but 2020 and 2023 regress. The
recursive correction alone is smaller than the 0.001 threshold but more stable. The
next honest test is to join this pregame column to the existing
`v4_backtest_predictions` path and compare it against the shipping weekly v4, not to
replace v4's dynamic update based on this benchmark.

## Most Deserving test

The comparison mirrors `scripts/fit_committee.py`: final CFP committee rank is the
target; the base is record, opponent-adjusted capped-margin quality, SOS, power status,
and the circular near-neighbor head-to-head feature. Every candidate is evaluated by
leave-one-season-out Spearman on 275 ranked team-seasons from 2015 through 2025.

| Candidate added to production-style base | LOSO Spearman | Change | Seasons improved | Worst mean after deleting one season |
|---|---:|---:|---:|---:|
| Base | 0.90358 | — | — | — |
| Literal persistence | 0.90211 | -0.00147 | 2 / 11 | -0.00200 |
| Full recursive lift | 0.90624 | +0.00266 | 4 / 11 | -0.00054 |
| **Second-pass opponent lift** | **0.90960** | **+0.00601** | **8 / 11** | **+0.00223** |
| Instability | 0.90316 | -0.00042 | 4 / 11 | -0.00138 |
| Network disruption | 0.90806 | +0.00448 | 7 / 11 | -0.00131 |
| Second-pass lift + disruption | 0.90813 | +0.00455 | 6 / 11 | -0.00331 |

Second-pass lift is the only new resume feature here that clears the same jackknife
logic used to admit head-to-head: its gain does not depend on any one season. The
full fixed point is already represented by `rating_z` and SOS, which likely explains
why the shallow second pass adds more than the fully recursive lift. It captures the
specific question “did the first layer of opponent evidence validate the result?”
without asking another feature to restate the final quality rating.

## Reproduction and provenance

The experiment is `python -m scripts.idempotence_backtest`. It accepts one compressed
schedule CSV and one ranking CSV per year rather than checking third-party data into
the repository. This run used the public `cfb_schedules_YEAR.csv.gz` assets from the
[sportsdataverse schedule release](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/cfb_schedules)
and the CFBD-derived `rankings_YEAR.csv` cache in
[pdaly42/sports-edge](https://github.com/pdaly42/sports-edge/tree/main/data/raw/cfb).

```bash
python -m scripts.idempotence_backtest \
  --schedules-dir /path/to/schedules \
  --rankings-dir /path/to/rankings
```

Three invariants are checked in `tests/test_idempotence.py`: a balanced cycle centers
at zero, reversing every result negates the rating, and the second pass correctly
downgrades a schedule-aided result.

## Limitations

- The predictive baseline is a clean 1500-seeded Elo because the licensed player and
  roster inputs needed to rebuild historical v4 are intentionally outside git.
- The public schedule source and the repository's CFBD cache do not have identical
  team universes, so the committee base is 0.90358 here versus 0.9126 for the final
  head-to-head model documented in `scripts/fit_committee.py`. Candidate comparisons
  are paired within the public-data universe; the absolute scores are not interchangeable.
- Game disruption is mathematically sensitive to bridge games in a sparse schedule
  graph. That is part of the proposed meaning, but it also makes the feature less
  stable and is why its positive mean is not enough to promote it.

