# CFB player-production forecast experiment

## Decision

The NFL player-prop framework contains useful player-level structure, but it does not
clear the bar as a new v4 game-prediction feature.

Forecasted passing, rushing, receiving, and sack production carries player-level
signal, and the expanded interaction/hurdle models improve player-level WAR
correlation in every modern fold. Team aggregation is still not stable. The best downstream
construction adds that production-informed WAR estimate beside the existing
projected-WAR feature. It saves 0.00103 static Brier versus the current player
baseline, but only 0.00030 after the weekly update, with a season-week bootstrap
interval of [-0.00090, +0.00026]. Its incremental forward-selection gains are 0.00063
and 0.00020 in the two mature
selection windows, both below the predeclared 0.001 bar. Nothing ships.

This is a football-forecast experiment, not a prop-betting backtest. No historical
CFB player-prop lines, prices, or timestamped quotes are present, so ROI and CLV are
unmeasured.

## What was transferred from the NFL framework

The reference is `Mstolte02/nfl-prop-models` at commit `812e987`. The parts retained
are the point-in-time contract, separate market targets, naive/regularized/nonlinear
candidate models, calibration-season selection, full predictive distributions, and
strict forward holdouts. The NFL implementation is weekly; the licensed CFB PFF
exports available here are season totals, so this adaptation forecasts the next
season rather than the next game.

Six targets are built from the PFF exports:

| target | eligible rooms | candidate family |
|---|---|---|
| passing yards | QB | carry-forward, ridge, HGB, interaction HGB, participation hurdle |
| passing touchdowns | QB | same |
| rushing yards | QB, RB, WR | same |
| receiving yards | RB, WR, TE | same |
| receptions | RB, WR, TE | same |
| defensive sacks | DT, EDGE, LB, CB, SAF | same |

For target season N, every player predictor is from N-1 or earlier: three production
lags, three WAR lags, prior snaps/share/room rank, prior experience, recruiting,
class, transfer status, prior team rating, and position. The season-N CFBD roster
defines the forecast population through the same loader used by the existing WAR
projection. Players without a target-season PFF row remain in the population with a
zero outcome. The historical CFBD roster cache does not have retrieval timestamps;
this is the same documented preseason-archive limitation as the current projected-WAR
feature, not evidence that the roster was frozen on a particular preseason date.

For each holdout, the immediately prior usable season selects the candidate with the
lowest predictive negative log likelihood. The selected candidate is then refit on
all permitted earlier seasons. Passing/rushing/receiving yard residuals compare
Gaussian, Student-t, Laplace, logistic, and skew-normal families. Count targets compare
Poisson and negative binomial; the hurdle candidate separately models a nonzero
season and conditional production. The fixed threshold-grid Brier is a distribution
diagnostic only.

## Player-production results

Weighted across the common 2022-25 holdouts:

| target | player-seasons | selected models | Pearson r | MAE | threshold Brier |
|---|---:|---|---:|---:|---:|
| passing yards | 2,980 | ridge / interaction / carry / hurdle, 1 each | .690 | 379.88 yd | .0734 |
| passing touchdowns | 2,980 | interaction 2/4, hurdle 1/4, HGB 1/4 | .688 | 2.83 | .0563 |
| rushing yards | 16,032 | carry-forward 3/4, hurdle 1/4 | .655 | 52.93 yd | .0461 |
| receiving yards | 17,186 | hurdle 2/4, HGB 1/4, carry 1/4 | .660 | 67.21 yd | .0402 |
| receptions | 17,186 | hurdle 3/4, HGB 1/4 | .676 | 5.51 | .0376 |
| defensive sacks | 29,157 | hurdle 4/4 | .518 | .402 | .0272 |

The expanded family changes the market winners, but does not create a clean universal
model. Hurdles dominate sparse sacks and receptions; interactions win two touchdown
folds and one passing-yard fold; simple carry-forward still wins three rushing folds.

## WAR channel

The production forecasts were added to the existing ex-ante WAR projection. Both
models use the same all-roster population and the same forward folds; the extension
receives only cross-fitted production predictions.

| holdout | player r, baseline | player r, + production | team WAR r, baseline | team WAR r, + production |
|---:|---:|---:|---:|---:|
| 2022 | .612 | **.625** | **.740** | .736 |
| 2023 | .626 | **.637** | **.832** | .814 |
| 2024 | .608 | **.618** | .778 | **.779** |
| 2025 | .595 | **.601** | .762 | **.769** |

The selector compares player and aggregated-team correlation on the immediately prior
calibration season. It chooses a hurdle in 2022, ordinary production HGB in 2023, and
a 50/50 baseline–interaction ensemble in 2024–25. Player MAE improves in 2022–24 and
worsens in 2025: .04072→.04065, .04029→.04009, .04001→.03993, and
.04047→.04058. The Laplace conditional-median candidate improves raw MAE more, but
collapses correlation in the zero-heavy target and is correctly rejected by the
stability selector. Team aggregation is mixed: down in 2022–23, nearly flat in 2024,
and up in 2025.

## Standalone and game-prediction channels

Direct offense is the equal-weight mean of within-season standardized passing yards,
passing touchdowns, rushing yards, receiving yards, and receptions. Direct defense is
standardized projected sacks. The WAR channel is a separately fitted player WAR model
with the six cross-fitted forecasts added. All team features enter games as
antisymmetric home-minus-away differences.

Pooled 2022-25 results from the existing v4 fitting/tuning and weekly replay:

| feature set | static Brier | online Brier | online log loss | online accuracy |
|---|---:|---:|---:|---:|
| clean core | .20958 | .18824 | .55522 | 70.55% |
| current projected WAR | .20695 | .18669 | .55111 | 70.79% |
| direct production, no projected WAR | .20791 | .18754 | .55283 | 70.44% |
| production-informed WAR, no current WAR | .20681 | .18669 | .55075 | 70.85% |
| projected WAR + direct production | .20644 | .18681 | .55107 | 70.48% |
| **projected WAR + production-informed WAR** | **.20592** | **.18639** | **.54990** | **70.58%** |
| production only | .22659 | .19725 | .57726 | 69.21% |

The direct component is not a replacement for WAR: versus current projected WAR it
worsens static Brier by 0.00096 and online Brier by 0.00085. The best combined channel
does improve static Brier by 0.00103, bootstrap 95% interval [-0.00152, -0.00056].
After weekly updating, the incremental difference contracts to -0.00030 with interval
[-0.00090, +0.00026].

### Fold and selection stability

| outer holdout | current WAR static | combined static | current WAR online | combined online |
|---:|---:|---:|---:|---:|
| 2022 | .20384 | .20384 | .19345 | .19345 |
| 2023 | .19559 | **.19471** | .17951 | **.17878** |
| 2024 | .21855 | **.21658** | .19208 | **.19191** |
| 2025 | .20964 | **.20839** | .18161 | **.18132** |

The first fold has no clean 2021 player-production frame because 2020 is deliberately
excluded from the WAR history; missing components are neutral zero. More importantly,
the selection-window comparison against the current projected-WAR baseline is only:

| selection window ending | current projected WAR | best combined | gain |
|---:|---:|---:|---:|
| 2023 | .19960 | .19897 | +.00063 |
| 2024 | .20533 | .20513 | +.00020 |

Both are positive and both fail +.001. The expanded family removes the earlier 2024
online reversal, but the gain remains too small and its bootstrap interval crosses zero.
The extension is retained as a reproducible experiment but is not added to production
model payloads or the JS port.

## Reproduction

```powershell
python -m war_model.player_production_forecast
python -m scripts.player_production_backtest
python -m unittest tests.test_v4 -v
```

Generated evidence lives in `war_model/player_production_metrics.json`,
`war_model/preseason_player_components.csv`, and the ignored research artifacts
`artifacts/player_production_backtest.json` / `.csv`.
