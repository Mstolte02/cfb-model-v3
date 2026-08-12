# V4.1 context and player-projection experiments

Date: 2026-08-12

## Decision

Ship the all-roster projected-WAR team prior. Do not ship rest, travel, schedule-load,
realized weather, or qualitative Four-Pass overrides. A quantitative rolling
reversibility family earns continued development as a jointly fitted initial-model
candidate; it is not interchangeable with a post-hoc override.

The player feature cleared the repository's frozen 0.001 Brier adoption threshold in
two consecutive expanding selection windows. Context did not. Plausible football
stories are not promoted without forward evidence.

## What the Russo framework supplied

[Andrew Russo's Four-Pass article](https://andrewrusso.ai/article-03-nfl_framework)
names four passes: base context; mismatch weighting; favorite fragility; and underdog
reversibility. It also says the passes produce structured judgment rather than a
formula. The page publishes neither coefficients nor game-level predictions, and its
86% claim has no disclosed estimand, sample, split, or replication data. It therefore
defined candidate questions, not a numerical component.

- Pass 1 became the schedule-context experiment below.
- Pass 2 maps to v4's already-tested granular offense/defense mismatch candidates;
  those still fail the 0.001 adoption threshold.
- The article's versions of Passes 3 and 4 require timestamped injury/depth, coaching
  tendency, and scheme data absent from this repository. Encoding them as analyst
  flags would make the backtest irreproducible. The experiment below instead defines
  reproducible scoreboard-based versions that are the model's own indices, not a
  reconstruction of an unpublished Russo formula.

## Fragility, reversibility, and structural weighting

Two distinct entry points were tested on the same 2,189 games from 2023-25. The
overlay test fits a penalized correction after the locked weekly v4 probability. The
initial test jointly estimates the core team and context coefficients inside the
reciprocal logistic/margin model. For every outer season N, all fitting and numeric
tuning use seasons before N only.

The rolling last-12-game measurements are frozen before kickoff. Fragility uses
giveback after halftime leads, lead-loss rate, second-half decline, and final-margin
volatility. Reversibility uses gain after halftime deficits, comeback-win rate, and
second-half gain. Four pseudo-opportunities shrink small samples. Structural evidence
uses five existing v4 offense-versus-defense matchup edges, their breadth and
nonlinearity, and antisymmetric interactions with the clean-core strength difference.
All features negate when the teams are swapped.

Post-hoc overlay results (negative is better):

| Candidate | Brier change | Margin MAE change | Margin RMSE change |
|---|---:|---:|---:|
| Fragility components | +0.00085 | -0.041 | -0.083 |
| Reversibility components | +0.00017 | -0.038 | -0.046 |
| Compact fragility/reversibility indices | +0.00031 | -0.046 | -0.060 |
| Structural edges | +0.00024 | -0.026 | +0.009 |
| Structural context weighting | +0.00036 | -0.122 | -0.162 |
| All Four-Pass proxies | +0.00072 | -0.294 | -0.287 |

The overlay can improve margins while worsening probability calibration. It is not a
probability-model candidate.

Joint initial-fit results against the same core-only architecture (negative is
better):

| Candidate | Tuned Brier change | Same-knobs Brier change | Margin MAE change | Decision |
|---|---:|---:|---:|---|
| Fragility components | -0.00074 | +0.00027 | +0.055 | reject: below bar and control regresses |
| **Reversibility components** | **-0.00155** | **-0.00101** | **-0.021** | promote as candidate |
| Compact fragility/reversibility indices | -0.00030 | +0.00034 | +0.028 | reject |
| Structural initial weighting | +0.00001 | +0.00007 | +0.033 | reject |
| All Four-Pass proxies | +0.00080 | +0.00039 | +0.070 | reject |

The tuned reversibility changes are -0.00250, -0.00013, and -0.00205 in 2023, 2024,
and 2025: improvement in all three outer folds. Its paired season-week bootstrap
interval for `Brier(joint)-Brier(core)` is [-0.00272, -0.00039], with 99.6% of draws
favoring the joint model. The identical-knobs control clears the pooled 0.001 bar but
is less stable (-0.00292, +0.00175, -0.00190 by season), and its interval includes
zero. The comeback-win coefficient is positive in every outer fold; component
direction is not being rescued by one anomalous season.

This promotes only the granular reversibility family, not the compact index and not
the full framework. Production v4 remains unchanged in this commit because rolling
pregame features need a guarded game-context path; forcing them into the season-fixed
team frame would silently change power-rating semantics. The reproducible research
paths are `python -m scripts.four_pass_backtest` and
`python -m scripts.four_pass_initial_backtest`.

## Pace, scripted windows, and style control

The pace hypothesis was tested with the [official CFBD API](https://api.collegefootballdata.com/)
`/drives` endpoint, which publishes drive clock, play count, score state, drive
number, and starting/ending period. One request per season cached all regular-season
drives for 2021-25. The
quick-pass hypothesis uses PFF's prior-year, dropback-weighted quarterback average
time to throw (TTT), depth of target, positive-EPA rate, and pressure-to-sack rate,
matched against prior-year TruMedia pressure generated/allowed, blitz volume, and
early-down pass tendency.

All drive traits are start-of-week rolling last-12-game summaries; same-week games
cannot see one another. Pace is valid-drive offensive game-clock seconds per play.
Provider rows with impossible elapsed time are discarded, and sparse values shrink
to a neutral 26.5 seconds/play. The candidate families are:

- pace identity: offensive/defensive seconds per play and plays per drive;
- scripted windows: approximately the first 15 offensive plays, final four minutes
  of Q2 plus first four of Q3 (Middle Eight), and Q4 net performance;
- state/control: shrunk win rates when entering Q4 trailing/leading and whether a
  game's realized pace moved closer to one team's entering preference;
- matchup control: expected fast/slow conditional success and pace-clash interactions
  with prior control and script strength;
- quick-pass/pressure: TTT and early pass tendency versus pressure, blitz, protection
  exposure, depth of target, and pressure-to-sack conversion.

Each quantity is reciprocal: swapping the teams negates the complete matchup vector.
Three entry points were evaluated on 2,189 games from 2023-25 (negative Brier change
is better):

| Candidate | Post-hoc overlay vs locked weekly v4 | Joint static initial fit | Context-adjusted weekly v4 |
|---|---:|---:|---:|
| Pace identity | +0.00045 | -0.00021 | +0.00019 |
| Scripted windows | +0.00034 | -0.00571 | **-0.00019** |
| State and control | +0.00108 | -0.00596 | +0.00322 |
| Pace matchup/control | +0.00007 | -0.00653 | +0.00111 |
| Quick pass versus pressure | +0.00183 | +0.00204 | +0.00115 |
| Pace + script + control | +0.00222 | -0.00824 | +0.00254 |
| All tempo/style | +0.00390 | -0.00576 | +0.00419 |

The static gains are large but do not survive the complete production architecture.
They are primarily another measurement of current team form, which the weekly rating
update already captures more efficiently. The best full-model result, scripted
windows, saves only 0.00019 Brier, changes sign by season (-0.00172 / +0.00215 /
-0.00105), and has a paired season-week 95% interval of [-0.00146, +0.00116]. It
fails both the 0.001 adoption bar and fold stability. Quick-pass/pressure regresses in
all three full-model folds; the available season-level TTT aggregation does not
validate that matchup theory.

The complete bundle worsens Brier by 0.00419. No pace/style feature ships. This does
not establish that tempo is irrelevant; it establishes that these pregame summaries
do not add independent probability information after the live rating. A future test
should use archived play-level win probability joined to clocked play-by-play for
state-conditioned WPA. CFBD's `/metrics/wp` requires a game ID and omits period/clock,
so reconstructing five seasons requires a per-game WPA pull plus the play feed—beyond
the current free API request budget. Selectively sampling games would be weaker than
the complete drive replay and was not used. Reproduce the present test with
`python -m scripts.tempo_style_backtest`.

## Rest, travel, and load

Inputs are exact CFBD kickoff timestamps, venue IDs, team home coordinates/time zones,
and the official CFBD venue catalog. The current [CFBD API](https://api.collegefootballdata.com/)
documents `/games`, `/venues`, and `/games/weather`. The configured account can access
the first two; `/games/weather` returns HTTP 401.

For each team before each kickoff, `src/context.py` computes rest days, short rest,
great-circle home-to-venue miles, absolute/eastward time-zone shift, prior 21-day
travel, and road streak. The correction is ridge logistic regression with the locked
v4 probability as a fixed logit offset. For test season N, it trains only on locked
predictions from earlier outer seasons.

Research supports testing these quantities but not assuming large universal effects.
The [2024 state-space rest study](https://doi.org/10.3389/frbhe.2024.1479832) estimates
rest while controlling latent team strength and finds modern NFL bye effects weak.
The NCAA home-advantage study [DOI 10.1080/24733938.2018.1524581](https://doi.org/10.1080/24733938.2018.1524581)
explicitly examines distance and time-zone direction. The college-football travel
market study [DOI 10.1177/1527002515574514](https://doi.org/10.1177/1527002515574514)
examines distance, direction, time zones, temperature, elevation, and aridity.

Online results on 2023-25 (2,189 games):

| Candidate | Pooled Brier change vs locked v4 | Fold changes (2023 / 2024 / 2025) | Decision |
|---|---:|---:|---|
| Rest | +0.00032 | -0.00026 / +0.00078 / +0.00043 | reject |
| Travel/time zones | **-0.00010** | +0.00017 / +0.00017 / -0.00065 | reject: too small, unstable |
| Prior load/road streak | +0.00056 | +0.00059 / +0.00075 / +0.00034 | reject |
| All context | +0.00087 | +0.00065 / +0.00183 / +0.00009 | reject |

For the full bundle, the paired season-week bootstrap 95% interval for
`Brier(context)-Brier(v4)` is [-0.00041, +0.00218].

### Why no Dijkstra feature

Dijkstra needs a graph whose edges represent real feasible routes and costs. The data
contains origins and venues, but no flight/road legs, hotel stay-overs, departure
times, or transport modes. On a complete location graph with great-circle distance,
the shortest path is the direct edge by the triangle inequality; Dijkstra returns the
same distance with extra machinery. Inventing route edges would add false precision.
Great-circle distance plus observed schedule load is the reproducible estimand until
actual itinerary data exists.

### Why no historical weather feature

[Open-Meteo's historical API](https://open-meteo.com/en/docs/historical-weather-api)
provides high-quality hourly reanalysis, but reanalysis is realized/postgame weather,
not the forecast known before kickoff. Adding it to a nominal pregame replay would
violate the temporal contract. Production should ingest timestamped forecasts when
games enter the forecast horizon; historical validation needs archived forecast runs
at the same lead time, not observations.

## Player projections and WAR aggregation

The previous projection model removed a leaked starter flag but still built its
historical population by retaining the K highest target-season snap players in each
room. Thus the feature values were lagged while population membership knew who later
won playing time.

V4.1 changes the population and aggregation:

1. start with every player in the CFBD season roster;
2. attach target WAR only as the training label, with zero for no target PFF row;
3. use only N-1 and earlier WAR/snaps/share, prior role rank, class, recruiting,
   transfer status, and prior team strength;
4. predict every roster member;
5. select the top K in each position room by the prediction, not realized snaps;
6. sum and standardize the projected team WAR.

The 2025 player holdout is now 12,073 roster players rather than 5,927
outcome-selected slots. Correlation is 0.592 versus 0.523 for carry-forward, and MAE
is 0.0411 versus 0.0413. These values are not directly comparable with the old, easier
population.

Team-level projection validity:

| Season | Teams | Projected vs realized team-WAR r | z-MAE |
|---|---:|---:|---:|
| 2021 | 130 | .581 | .708 |
| 2022 | 131 | .687 | .622 |
| 2023 | 133 | .803 | .506 |
| 2024 | 134 | .741 | .583 |
| 2025 | 136 | .762 | .530 |

Game-model selection and outer results:

| Outer season | Feature selected using seasons before test | Prior-window validation gain vs clean core | Outer dynamic Brier |
|---|---|---:|---:|
| 2022 | clean core | insufficient history | .1938 |
| 2023 | clean core | .00080 (below bar) | .1809 |
| 2024 | core + projected WAR | **.00213** | .1921 |
| 2025 | core + projected WAR | **.00232** | .1816 |

Pooled 2022-25 dynamic Brier is .18713, versus .18824 before the projected player
feature and .18983 for aligned CFBD pregame Elo. The paired difference against Elo is
-0.00270 with a 95% season-week bootstrap interval of [-0.00738, +0.00183]. The model
is competitive with Elo; the interval still does not prove superiority.

## Remaining limitations

- Historical roster cache files are season-specific but not retrieval-timestamped.
  They remove realized participation and snap selection, but a frozen preseason roster
  archive would provide stronger provenance.
- WAR remains a predictive attribution index, not identified causal wins added.
- The top-K room aggregation is nonlinear and role-aware, but it still lacks actual
  historical two-deep order, injury timing, weak-link OL effects, and player-player
  interactions.
- Weather forecast ingestion and itinerary data should be evaluated only when their
  historical information sets can be reconstructed honestly.
