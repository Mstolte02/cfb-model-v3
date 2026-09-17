# Changelog

The release history of the model, newest first. These notes were one running section
of the root README until September 2026. They are kept as written, so a note
describes the model at the time it was written, not the model today. File paths in
them are relative to the repository root.

For what the model does now, read the root [README](../../README.md).

## [v4](v4.md) — the model that ships today

| Release | What changed |
|---|---|
| v4.1 | Projected rosters, clean time, reciprocal matchups, live evidence |

V4 is smaller than v3 on purpose. The audit found that part of the v3 player signal
knew which players would play, and how many snaps they would take, in the season it
was supposed to predict. V4 states a temporal contract, selects every feature inside
expanding forward folds, and adds a weekly update so the model learns from the season
in progress.

## [v3](v3.md) — player WAR and the web app

| Release | What changed |
|---|---|
| v3.10 | One build, a split offensive line, and EA back in its lane |
| v3.9 | An audit, worked through |
| v3.8 | A denominator, an ordering, and a calculator |
| v3.7 | Head-to-head, realistic scorelines, and a résumé for the field |
| v3.6 | The ratings stop believing last season |
| v3.4 | WAR rebuilt on 11 seasons |
| v3.3 | Talent weights jointly fitted, and a Method view |
| v3.2 | Live depth charts, feature audit, formation layout |
| v3.1 | Committee model, bracket fixes, wins ledger, lineup view |
| v3.0 | Player WAR joins the talent signal, and the web app is rebuilt |

V4 replaced the v3 team model. The player WAR model and the web app came from v3 and
still ship.

## [v2](v2.md) — the margin ensemble and the playoff simulation

The ridge margin model joins the logistic as an ensemble, and
`scripts/simulate_playoff.R` simulates the 12-team bracket with `cfbseedR`.

## [v1](v1.md) — the original README

The leakage-free rebuild of the original notebooks: a calibrated mean, an
offense-against-defense matchup structure, and the first LOSO results.
