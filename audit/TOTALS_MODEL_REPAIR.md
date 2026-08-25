# Repairing the totals model

## Status

**Adopted and shipped.** The published game total now comes from a points model that
knows a team's scoring level and pace. It is meaningfully better than what it
replaced and still well short of the market, and the site says so.

## The defect

`MODEL_VS_MARKET_DIAGNOSIS.md` found the published model total correlated **.099**
with actual points over 2022–25 and **.002 in 2025** — no measured information in the
most recent season — against the market's .379. It was also under-dispersed, with a
prediction SD of 5.2 against the market's 7.4 and the outcome's 16.9.

The cause was structural, not a bug. `src/spread.py` predicts a side's points from
three numbers:

```
points_scored ~ O_scorer + D_opponent + home
```

That is enough for a **margin**, which is a difference of two standardised composites
and correlates .507 with actual margin. It cannot produce a **total**, for two
reasons:

* **No possession term.** Points are roughly efficiency times possessions. A 60-play
  team and an 85-play team with identical efficiency ratings received identical
  points predictions. Offensive plays per game range from 59.9 to 89.0 across the
  664 team-seasons in the play cache.
* **No scoring level.** `O` and `D` are standardised opponent-adjusted *rate*
  composites. They encode that a team is 1.4 SD above average, never that it scores
  34 a game. The ridge had to recover a points level from a z-score, which it can only
  do at league average — hence the compressed spread.

## The test

`scripts/totals_backtest.py`, expanding folds, level and pace lagged to season N−1.
Candidates add one omission at a time so the answer names which one mattered.

| candidate | RMSE | corr | prediction SD | bias |
|---|---:|---:|---:|---:|
| current (three feature) | 17.148 | .104 | 5.23 | +0.95 |
| + scoring level | 17.027 | .150 | 6.01 | +0.42 |
| + pace | 17.070 | .121 | 5.32 | +0.80 |
| + both | 17.013 | .154 | 6.06 | +0.39 |
| **+ both, with interaction** | **16.997** | **.156** | 6.03 | +0.35 |
| *market, same games* | *15.587* | *.379* | *7.41* | *−0.39* |

`current` reproduces the published number exactly (.104 against the published .104),
so the comparison is against the real thing rather than an approximation.

Every candidate beats it on RMSE with a 95% blocked bootstrap interval excluding
zero; the selected specification is −5.12 mean squared error, interval
[−8.79, −1.54].

**Scoring level was the bigger omission** — .150 alone against pace's .121. The
interaction earns its place because points are closer to a product of level and
possessions than a sum.

By season, correlation with actual points:

| | 2022 | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|
| current | +.172 | +.122 | +.101 | **+.005** |
| selected | +.172 | +.162 | +.187 | **+.094** |
| market | +.478 | +.381 | +.281 | +.345 |

## Adoption

The predeclared rule was an RMSE improvement with a 95% interval excluding zero and
correlation materially above .099. Both hold: correlation rises about 50% in relative
terms, and 2025 goes from no information to some.

**What this does not do.** At .156 against the market's .379, the totals model remains
far weaker than the price. The repair removes a published number that carried almost
nothing; it does not make totals bettable, and
`BET_THRESHOLD_CALIBRATION.md` still finds no supportable totals threshold. The board
now states this next to the column.

## Shipping notes

`src/totals.py` owns the shared design. `scripts/totals_backtest.py` and
`scripts/export_viz.fit_points_model_v2` both build rows through it, so the measured
model and the exported model cannot drift apart.

The client recomputes the same arithmetic in `viz/app.js` (`sidePoints`). That hand
port is the fragile part: a mismatch would be silent, publishing a different total
from the one measured here. `tests/test_totals.py` pins the JS design vector against
`totals.POINTS_FEATURES` term by term, and Python and JS were checked to agree to the
cent on sample matchups.

The payload keeps the old three-coefficient `points` block alongside the new
`points_v2`. A browser holding a cached older `app.js` still indexes the old block,
and a client that does not understand `points_v2` falls back rather than breaking.

The margin model is untouched. It was not broken.

## Reproduction

```powershell
python -m scripts.totals_backtest
python -m scripts.export_viz
```

Artifacts: `artifacts/totals_backtest.json` and `.csv`.
