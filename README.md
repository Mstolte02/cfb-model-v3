# CFB Predictive Model

A college football forecasting system for the 2026 season. It rates all 138 FBS
teams, predicts each game, builds a wins-above-replacement figure for every player,
and publishes the result as a browser app.

**Live site: [2026 Forecast Lab](https://mstolte02.github.io/cfb-model-v3/)**

Three rules hold across the whole build:

- **A forecast uses only what was known before the game.** Every feature and every
  number in the model is selected inside expanding forward folds.
- **A feature ships only if it earns its place.** It must clear a predeclared
  accuracy bar in consecutive selection windows. Most tested features do not, and the
  evidence for each one stays in [`audit/`](audit/).
- **Team A against Team B gives the same answer as Team B against Team A.** The
  reciprocal result comes from the structure, not from a correction.

## Results

The model replays each season from the start, week by week, and predicts every game
from the state before that week.

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

python -m scripts.train_v4      # select, fit, and publish the model
python -m scripts.rank          # 2026 power ratings
python -m scripts.export_viz    # write the data the web app reads
python -m http.server 8642 -d viz
```

Open <http://localhost:8642>. The PFF, TruMedia and depth-chart files are private, so
they live outside the repository, in a `source-data/` folder beside it or in your
Downloads folder. Set `CFB_EXTERNAL` to point somewhere else. `config.py` lists every
path and every environment variable that overrides one.

## Weekly workflow

Run this after a week of games is complete. `update_v4` records its predictions
before it applies a result, and it ignores a game it has already processed.

```powershell
python -m scripts.update_v4                      # apply completed games
python -m scripts.rank                           # publish current ratings
python -m scripts.update_player_values           # current player estimates
python -m scripts.export_viz                     # rebuild the app data
Rscript scripts/simulate_playoff.R 500 current   # CFP projection
python -m unittest discover -s tests -v          # invariants
```

A push to `main` deploys the site with the Pages workflow.

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

**The weekly update.** The updater scores the difference between the observed margin
and the margin its own pregame probability implied, in units of the fitted margin
sigma. The score is capped at ±2.5 and multiplied by K = .20.

**Player WAR.** `war_model/` builds a wins figure for every FBS player in five
stages: 87<!--live:n_facets--> facets measure the jobs a player does, a regression
against the following season's wins prices each facet, a Massey rating turns the team
total into wins, replacement credit turns wins above average into wins above
replacement, and a projection carries it forward to 2026. See
[war_model/README.md](war_model/README.md).

**The web app.** `viz/` is a static site with five hubs — Futures, Power Rankings,
Weekly, Ratings, and Simulation. It reads the JSON that `export_viz` writes. There is
one build and one set of numbers.

## Repository layout

| Path | What is in it |
|---|---|
| [`src/`](src/) | The model: features, opponent adjustment, rating, prediction, spreads, totals |
| [`src/data/`](src/data/) | Loaders for CFBD, PFF, TruMedia, plays, coaches and WAR |
| [`scripts/`](scripts/) | Entry points. `train_v4`, `update_v4`, `rank`, `export_viz`, and one backtest per experiment |
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
| PFF exports | Player grades and snap counts, the base of every WAR facet |
| TruMedia | Supporting team and player measures |
| thetwodeep.com | 2026 depth charts for all 138 FBS teams. Ourlads is the fallback |
| EA CFB 27 ratings | Ordering players with under 300 prior snaps, who have no record to rank them by |
| A five-source QB composite | The starting quarterback in each room |

Injuries are manual, in `war_model/availability_2026.csv`. The one depth-chart source
with an injury feed prohibits automated access.

## Documentation

- [docs/](docs/) — the documentation index.
- [docs/changelog/](docs/changelog/) — the release history, v1 to v4. Every version
  note that used to sit in this README is there, as written.
- [audit/](audit/) — the experiments, including the ones that failed.
- [war_model/README.md](war_model/README.md) — how player WAR is built.
