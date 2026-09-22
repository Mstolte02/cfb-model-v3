# PFF API integration and promotion decision

## Decision

**The original conservative decision below has been superseded by the complete
downstream test in `ACCURACY_RECOVERY_REVIEW.md`.** The API ingestion layer ships,
and since 2026-09-22 the expanded WAR is the production WAR: `war_model` reads all
five API report families by default, and the live in-season model is the four-member
expanded-WAR ensemble described there. Its player-ranking result was mixed, but its
team projection improved in every season and the game model improved in every outer
season.

The newer player reports improved forward team-level WAR signal, but the uncertainty
still includes no improvement and the downstream player projection produced a mixed
result. The full-season team-stat tables were clearly worse than the features already
in the model. This is enough evidence to keep collecting and testing the data, not
enough to silently change the published model.

## What the API adds

The public API exposes season and game-filtered player aggregates, team statistical
tables, team directories, and position-specific reports. It does **not** expose a
play-by-play feed, so it cannot currently supply one row per play for a play-level WAR
model. The best incremental fields for the present architecture are:

- true-pass-set pass blocking, pressure rate allowed, and pass-block win rate;
- zone/gap run-block grades;
- true-pass-set pass rush and pass-rush win rate;
- run-stop rate and average tackle depth;
- forced-incompletion rate, yards per coverage snap, and coverage snaps per target.

Game filters may still support opponent- and context-adjusted player splits later,
but constructing those would require many more calls and a separate validation plan.

## Data validation

The five 2025 API facet reports were compared with the legacy exports. Every legacy
column was retained, all legacy player IDs were present, and the API response was a
strict row superset:

| Report | API rows | Legacy rows |
|---|---:|---:|
| Blocking | 6,052 | 5,894 |
| Defense | 5,767 | 5,652 |
| Passing | 569 | 541 |
| Receiving | 2,398 | 2,344 |
| Rushing | 1,727 | 1,687 |

All five position-report families needed by the WAR experiment were staged for model
seasons 2014–19 and 2021–25. Seven team-stat categories were staged for 2020–25.
Licensed responses remain under `source-data/pff_api` and are not committed.

## Experiment 1: full-season team tables

`scripts/pff_api_backtest.py` lags every PFF season by one year and runs the same
expanding selection contract used by the model. It compares clean PFF feature groups,
the current feature set, and their combination.

The selector continued to choose the existing player layer. The combined current +
PFF process arm had worse validation Brier in every selection window:

| Test season | Existing player layer | Current + PFF process | All PFF |
|---|---:|---:|---:|
| 2023 | .200568 | .204314 | .217983 |
| 2024 | .195386 | .197512 | .207953 |
| 2025 | .199051 | .200161 | .215495 |

**Decision:** reject the full-season PFF team tables for the preseason and weekly
predictors. They are descriptive and largely redundant with existing information.

## Experiment 2: API-only player facets

`scripts/pff_war_facets_backtest.py` constructs player values using the existing WAR
candidate grammar, aggregates them to team-season features, and predicts next-season
adjusted win percentage in expanding folds. It never overwrites production artifacts.

| Feature set | RMSE | MAE | Correlation |
|---|---:|---:|---:|
| Existing facets | .163874 | .131139 | .5193 |
| Existing + all API facets | **.162879** | **.130222** | **.5280** |

Five of six folds improved. The paired MSE difference was −.000325, but the
team-cluster bootstrap 95% interval was [−.000748, +.000110], with a 92.8% probability
of improvement. Run blocking, run defense, and coverage helped individually; pass
blocking and pass rush did not.

**Decision:** promising, but below the bar for automatic promotion because the
clustered interval crosses zero.

## Experiment 3: complete WAR and projection rebuilds

Two isolated builds tested whether the apparent team-level gain survives the entire
valuation and projection chain. Neither wrote into `war_model/` production artifacts.

| 2025 player projection holdout | Production | All API groups | Positive groups only |
|---|---:|---:|---:|
| Correlation | **.592** | .581 | .578 |
| MAE | .0411 | **.0401** | .0406 |

The all-API build raised the weighted facet total's correlation with next-season wins
from .527 to .553, but lowered the same-season Massey fit from .702 to .683 and reduced
player-projection correlation by .011. The conservative run-blocking, run-defense,
and coverage build did not resolve that tradeoff.

**Decision:** keep production artifacts unchanged. Lower MAE is useful, but a new WAR
feature set should not ship while it degrades ranking correlation and its aggregate
improvement is statistically uncertain.

## Promotion rule

Revisit after another completed season or a longer historical entitlement. Promote
only if the API facets:

1. improve clustered forward team-level error with a confidence interval excluding
   zero;
2. do not reduce held-out player-projection correlation;
3. retain or improve MAE; and
4. remain stable across position groups and seasons rather than deriving the gain from
   one cohort.

Until then, API WAR features require `PFF_API_WAR_REPORTS`; the default is empty and
therefore identical to the current production build.

## Follow-up: the missing downstream test

The initial decision put too much weight on one player-level holdout and the
clustered interval. Rebuilding the historical preseason team projections changed the
answer. Expanded-WAR projected-versus-realized team correlation improved in every
available season: `.581→.607`, `.687→.749`, `.803→.824`, `.741→.801`, and
`.762→.783` for 2021–25.

In the complete 2023–25 game replay, replacing current WAR with expanded WAR improved
the preseason, score-update, EWMA, and combined arms. The point gains are small, but
they are all in the same direction and the replacement adds no game-model parameter.
The correct conclusion is therefore to retain the expanded measurement and let
regularization/model averaging price its contribution—not to discard it because one
interval crosses zero.

See [`ACCURACY_RECOVERY_REVIEW.md`](ACCURACY_RECOVERY_REVIEW.md) for the full result.
