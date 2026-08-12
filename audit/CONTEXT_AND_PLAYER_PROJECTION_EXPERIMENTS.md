# V4.1 context and player-projection experiments

Date: 2026-08-12

## Decision

Ship the all-roster projected-WAR team prior. Do not ship rest, travel, schedule-load,
realized weather, or qualitative Four-Pass overrides.

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
- Passes 3 and 4 require timestamped injury/depth, coaching tendency, and scheme data
  absent from this repository. Encoding them as analyst flags would make the backtest
  irreproducible, so they remain hypotheses.

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
