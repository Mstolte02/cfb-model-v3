# CFB Predictive Model

A college football forecasting system for the 2026 season. It rates all 138 FBS
teams, predicts each game, builds a wins-above-replacement figure for every player,
and publishes the result as a browser app.

**Live site: [2026 Forecast Lab](https://mstolte02.github.io/cfb-model-v3/)**

Three rules hold across the whole build:

- **A forecast uses only what was known before the game.** Every feature and every
  number in the model is selected inside expanding forward folds.
- **A feature ships only if it earns its place in forward loss.** Candidates are
  compared in outer season folds, weak signals are regularized or averaged rather than
  zeroed by a significance threshold, and uncertainty is reported beside the point
  estimate. Leakage and consistent out-of-sample losses still reject a feature. The
  evidence for each one stays in [`audit/`](audit/); the policy is in
  [ACCURACY_RECOVERY_REVIEW.md](audit/ACCURACY_RECOVERY_REVIEW.md).
- **Team A against Team B gives the same answer as Team B against Team A.** The
  reciprocal result comes from the structure, not from a correction.

## Results

The model replays each season from the start, week by week, and predicts every game
from the state before that week.

**The live model since 2026-09-22 is v5**, the equal average of four expanded-WAR
preseason models, each updated by score innovation and opponent-adjusted form. It was
selected on a strict outer replay of 2023–25, where every current-season transform is
recomputed from earlier weeks only:

| Strict outer replay, 2023–25 | Games | Brier |
|---|---:|---:|
| v4: preseason model + score update | 2,189 | .178986 |
| **v5: four-member expanded-WAR ensemble** | **2,189** | **.177832** |
| CFBD pregame Elo, same games | 2,189 | .184959 |

v5 is better in each of the three seasons. The pooled difference is −.001154 Brier with
a season-week bootstrap interval of [−.002409, +.000099]: a 96.4% probability of
improvement, not a certainty. See
[ACCURACY_RECOVERY_REVIEW.md](audit/ACCURACY_RECOVERY_REVIEW.md).

The v4 replay the site launched with:

| Strict expanding replay, 2022–25 | Games | Brier | Log loss | Accuracy |
|---|---:|---:|---:|---:|
| Preseason model, no weekly update | 2,913 | .2025 | .5864 | 67.52% |
| **Production model, weekly update** | **2,913** | **.1845** | **.5452** | **71.47%** |
| CFBD pregame Elo, same games | 2,913 | .1898 | .5613 | 70.99% |

The model beats Elo by more than its own uncertainty: the paired season-week bootstrap
difference is −0.00533 Brier, with a 95% interval of [−0.00976, −0.00118]. Elo wins in
0.6% of draws. Elo is a low bar, and there is no closing-line file in the repository,
so the model still has no market benchmark. Read
[MODEL_VS_MARKET_DIAGNOSIS.md](audit/MODEL_VS_MARKET_DIAGNOSIS.md) before you bet
anything on it.

## Quickstart

```powershell
pip install -r requirements.txt

# The CFBD key goes in .env at the repository root (gitignored):
#     CFBD_API_KEY=your_key_here
# Keep the PFF key in the operating-system environment, never in the repo:
#     setx PFF_API_KEY "your_key_here"

python -m scripts.train_v4              # the v4 preseason model and site data
python -m scripts.export_viz            # write the data the web app reads
python -m scripts.train_live_ensemble   # fit the four v5 members
python -m scripts.publish_live_ensemble # attach v5 to the site data and replay 2026
python -m http.server 8642 -d viz
```

Open <http://localhost:8642>. The PFF, TruMedia and depth-chart files are private, so
they live outside the repository, in a `source-data/` folder beside it or in your
Downloads folder. Set `CFB_EXTERNAL` to point somewhere else. `config.py` lists every
path and every environment variable that overrides one.

## Weekly workflow

Nothing is required. The **Capture market snapshot** workflow runs every six hours:
it pulls prices, final scores and CFBD's regular-season `/stats/game/advanced`
(committed to `data/live/`), replays the whole season from week 0 through the v5
ensemble with `scripts/ensemble_replay.py`, and commits `viz/data/model_v4.json` and
`ratings.json`. Its Monday 12:30 PM ET run locks the week's betting board, freezing a
point-in-time model snapshot into every row. A push to `main` rebuilds the CFP
projection with `scripts/simulate_playoff.R` and deploys the site.

To do the same thing by hand:

```powershell
python -m scripts.update_v4                      # finals + form -> replay (v5 path)
python -m scripts.rank                           # print current ratings
Rscript scripts/simulate_playoff.R 500 current   # CFP projection
python -m unittest discover -s tests -v          # invariants
```

## How it works

**The temporal contract.** A season-N forecast may use completed season N-1 team
performance, earlier player history, the published season-N roster, and preseason
information such as recruiting and returning production. It may not use season-N
participation, roles, performance, or realized snaps.

**The team rating.** Each team collapses to an offensive rating and a defensive
rating, adjusted for the opponents it played. The game model differences the two team
vectors, fits a logistic with no intercept, and calibrates with temperature only. That
is why reversing the teams gives the exact complement.

**The features.** Forward selection chose five: opponent-adjusted offense,
opponent-adjusted defense, recruiting talent, returning production, and the roster's
projected player WAR. Recruiting enters once, as a single principal component, and
talent is orthogonalised against it.

**The in-season model (v5).** Four preseason models share the reduced feature set and
the expanded, PFF-API-enriched WAR; two add talent curvature, two add
production-informed WAR. Each keeps its own rating walk: the observed margin minus the
margin its pregame probability implied, in units of its margin sigma, capped at ±2.5
and multiplied by K = .25. A logistic stack then combines the week-0 rating
difference, the change the walk has made since week 0, opponent-adjusted season-to-date
offense and defense form, and home field. The published probability is the mean of the
four. The spread is that probability read through the ensemble's margin sigma.

**v5.1 (from 22 September 2026)** adds PFF's season-to-date offence and defence
composites as two more stack columns in every member, used whenever PFF's table
through the previous week exists and falling back to the v5 stack when it does not.
On the 2023–25 replay it is −.001225 Brier against v5, better in all three seasons.
See [V5_EXTENSION_EXPERIMENTS.md](audit/V5_EXTENSION_EXPERIMENTS.md), which also records
what did not help: pace, game control, scheme clashes and last-week availability.

**What v5 did not change.** Week 1 is graded on the v4 season-fixed margin, the
futures board (now archived) was held to the v4 week-0 ratings, and every Market board
row keeps the model snapshot it was locked with; those v4 blocks ship unchanged beside
the ensemble.
v5 prices the board from the week 5 lock on.

**Player WAR.** `war_model/` builds a wins figure for every FBS player in five
stages: 106<!--live:n_facets--> facets measure the jobs a player does, a regression
against the following season's wins prices each facet, a Massey rating turns the team
total into wins, replacement credit turns wins above average into wins above
replacement, and a projection carries it forward to 2026. See
[war_model/README.md](war_model/README.md).

**The web app.** `viz/` is a static site with four hubs — Power Rankings, Weekly,
Ratings, and Simulation. It reads the JSON that `export_viz` writes. There is one build
and one set of numbers. The Futures hub was taken off the site on 23 September 2026;
its code and the steps to put it back are in
[archive/futures-2026/](archive/futures-2026/README.md).

## Repository layout

| Path | What is in it |
|---|---|
| [`src/`](src/) | The model: features, opponent adjustment, rating, prediction, spreads, totals |
| [`src/data/`](src/data/) | Loaders for CFBD, PFF, TruMedia, plays, coaches and WAR |
| [`scripts/`](scripts/) | Entry points. `train_v4`, `train_live_ensemble`, `publish_live_ensemble`, `update_v4`, `rank`, `export_viz`, the stdlib live runtime `ensemble_replay`, and one backtest per experiment |
| [`war_model/`](war_model/) | The player WAR build, its rosters, and its own README |
| [`viz/`](viz/) | The published web app |
| [`audit/`](audit/) | One write-up per experiment, with the decision it produced |
| [`docs/`](docs/) | Documentation index and the release history |
| [`tests/`](tests/) | Temporal, reciprocity, selection and weekly-order invariants |
| [`data/`](data/) | Small checked-in inputs, such as returning production and market snapshots |
| `config.py` | Every path, year and constant the build uses |

## Data sources

| Source | Used for |
|---|---|
| CollegeFootballData.com | Games, advanced season stats, recruiting, returning production, drives |
| PFF exports/API | Player grades and snap counts, the base of every WAR facet; API additions are staged and tested before promotion |
| TruMedia | Supporting team and player measures |
| thetwodeep.com | 2026 depth charts for all 138 FBS teams. Ourlads is the fallback |
| EA CFB 27 ratings | Ordering players with under 300 prior snaps, who have no record to rank them by |
| A five-source QB composite | The starting quarterback in each room |

Injuries are manual, in `war_model/availability_2026.csv`. The one depth-chart source
with an injury feed prohibits automated access.

### PFF API staging

PFF API downloads go to `source-data/pff_api`, outside Git and separate from the
hand-verified legacy exports. A sync never changes the production model merely
because a credential or downloaded file is present.

```powershell
python -m scripts.sync_pff_api --seasons 2025 --validate-legacy
python -m scripts.sync_pff_api --seasons 2014-2019,2021-2025 --position-reports pass-blocking,run-blocking,pass-rush,run-defense,coverage
python -m scripts.sync_pff_api --seasons 2020-2025 --team-stats
```

The richer WAR reports are opt-in for experiments with
`PFF_API_WAR_REPORTS=all` or a comma-separated subset such as
`PFF_API_WAR_REPORTS=rblk,rdef,cov`. See
[`audit/PFF_API_EXPERIMENTS.md`](audit/PFF_API_EXPERIMENTS.md) for the promotion
decision and forward-test results.

The combined expanded-WAR/EWMA experiment is reproduced with
`python -m scripts.prior_decay_backtest`. Its selection-policy review and results are
in [`audit/ACCURACY_RECOVERY_REVIEW.md`](audit/ACCURACY_RECOVERY_REVIEW.md).

## Documentation

- [docs/](docs/) — the documentation index.
- [docs/changelog/](docs/changelog/) — the release history, v1 to v4. Every version
  note that used to sit in this README is there, as written.
- [audit/](audit/) — the experiments, including the ones that failed.
- [war_model/README.md](war_model/README.md) — how player WAR is built.
