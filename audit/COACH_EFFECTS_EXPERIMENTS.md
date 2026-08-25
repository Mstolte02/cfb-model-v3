# Coach effects experiments

## Prior and phase status

The prior is that head-coach mean effects will fail the model's predictive adoption
bar because much of the signal is mediated by recruiting talent and returning
production. A coaching-change contribution to uncertainty is more plausible. The
numbers below are descriptive identification diagnostics only; they are not evidence
that a coach feature predicts games.

**Status: Phases 0–5 complete. No coach feature clears the predictive adoption bar; nothing ships.**

## Phase 0 — mover graph

Scope is 2014-2025, the completed FBS seasons shared
with the available player/model history. CFBD stable coach ids and team ids define the
bipartite graph. For the 132 team-seasons with more
than one returned head coach, the row is attributed to the coach with the most games;
ties break on coach id. The selected coach's game share is retained (median
1.000, p10 1.000, minimum
0.000).

| diagnostic | result |
|---|---:|
| expected FBS team-seasons | 1,568 |
| matched coach team-seasons | 1,568 (100.0%) |
| unmatched / outside the graph | 0 |
| movers with >=2 seasons at each of >=2 schools | **72** |
| movers with >=3 seasons at each of >=2 schools | **37** |
| connected components | 39 |
| largest component | **88 schools, 212 coaches** |
| largest-component team-seasons | **1,021** (65.1% of matched; 65.1% of all FBS) |
| team-seasons outside largest component, including unmatched | 547 |
| team-seasons outside every mover-connected component, including unmatched | 332 |

Unmatched examples: none.

"Outside any connected component" is reported as the literal data-join quantity:
team-seasons with no coach node are outside the graph. Because every matched edge is
mathematically in some component, the table also reports the more useful identification
quantities outside the largest component and outside every component containing a
coach who moved schools.

### Seasons per coach-school spell

There are 432 consecutive spells. Mean length is
3.63 seasons; the median is 3, p75 is
5, p90 is 7, and the
maximum is 12.

Histogram: 1y: 87, 2y: 89, 3y: 77, 4y: 63, 5y: 32, 6y: 32, 7y: 17, 8y: 8, 9y: 7, 10y: 7, 11y: 4, 12y: 9.

### Gate verdict: PASS

The predeclared gate requires at least 40 two-season-at-each-school movers and about
60% of all team-seasons in the largest connected component. This graph has
72 qualifying movers and
65.1% coverage, so it is **eligible for Phase 2 after review**.

This is a stop gate, not authorization to interpret coach effects. Even with a pass,
limited-mobility bias can inflate naive coach variance and push coach-program
covariance downward. Phase 2, if approved, must report both naive and bias-corrected
decompositions and must keep existing preseason observables in the same fit.

## Reproduction

```powershell
python -m scripts.coach_effects
```

Machine-readable results are in `audit/coach_mover_graph.json`. CFBD coordinator data
does not exist; this experiment can test head coaches only.

## Phase 2 — descriptive variance decomposition

The reviewed graph gate was followed by a 2021–2025 model-aligned frame: **653 team-seasons, 134 programs, and 222 head coaches**. It contains 147 between-season head-coach changes and 87 dominant-coach rows with a midseason change. The shorter period is imposed by availability of all four existing preseason covariates in the same fit.

### Model comparison and bias handling

The naive column is the two-way HDFE plug-in estimate from `pyfixest`. The corrected candidates are crossed program/coach Bayesian models in PyMC. Gaussian and Student-t observation models were fit; the interaction candidate adds talent×returning, head-coach-change×returning, midseason-change×returning, and a partially pooled coach-specific returning-production slope. The corrected headline model is selected by the largest PSIS-LOO expected log predictive density separately for each outcome. This is descriptive model comparison, not authorization to ship a feature.

| outcome | naive coach share | naive program share | naive 2cov(C,P) | singletons dropped | corrected candidate | ELPD-LOO | corrected coach share (90% interval) | corrected corr(C,P) |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| overall | 36.3% | 35.9% | 14.6% | 58 | student_t | -710.7 | 0.9% (0.0%, 4.8%) | 0.765 |
| offense | 35.8% | 32.5% | 9.8% | 58 | student_t | -786.9 | 5.3% (0.3%, 12.2%) | 0.760 |
| defense | 33.7% | 33.8% | 4.8% | 58 | student_t | -799.6 | 2.5% (0.0%, 8.0%) | 0.751 |

The displayed shares are observation-weighted. The machine-readable artifact reports every variance and twice-covariance term, including X–FE cross terms that are omitted from the abbreviated identity in the brief, and verifies exact reconstruction of outcome variance. Bayesian intervals are posterior 5th–95th percentiles. HDFE singleton removal is recursive; partial pooling retains all rows.

### Distribution and interaction candidates

| outcome | candidate | ELPD-LOO | SE | pLOO | warning | max R-hat | min bulk ESS | divergences |
|---|---|---:|---:|---:|---|---:|---:|---:|
| overall | gaussian | -714.3 | 21.0 | 38.6 | False | 1.009 | 324 | 0 |
| overall | student_t | -710.7 | 20.6 | 32.7 | False | 1.012 | 385 | 0 |
| overall | student_t + interactions | -713.4 | 20.5 | 42.5 | True | 1.009 | 283 | 0 |
| offense | gaussian | -789.0 | 20.2 | 56.2 | False | 1.008 | 322 | 0 |
| offense | student_t | -786.9 | 20.0 | 62.9 | False | 1.025 | 177 | 0 |
| offense | student_t + interactions | -790.0 | 19.7 | 69.4 | False | 1.010 | 353 | 0 |
| defense | gaussian | -800.9 | 19.3 | 34.8 | False | 1.011 | 221 | 1 |
| defense | student_t | -799.6 | 19.3 | 42.7 | False | 1.027 | 278 | 0 |
| defense | student_t + interactions | -802.6 | 19.0 | 54.8 | True | 1.013 | 524 | 0 |

### Highest and lowest shrunken coach intercepts

These are posterior means from the selected corrected model, in the original rating scale. `posterior SD` is the uncertainty analogue of a standard error. For an interaction-selected model, the list ranks the intercept at average returning production; the decomposition itself uses each row's combined intercept-plus-slope coach component.

#### Overall rating
| side | coach | effect | posterior SD | seasons |
|---|---|---:|---:|---:|
| high | Curt Cignetti | 0.163 | 0.268 | 3 |
| high | Mario Cristobal | 0.141 | 0.228 | 5 |
| high | Jim Harbaugh | 0.125 | 0.227 | 3 |
| high | Josh Heupel | 0.121 | 0.215 | 5 |
| high | Ryan Day | 0.114 | 0.211 | 5 |
| high | Dan Lanning | 0.114 | 0.215 | 4 |
| high | James Franklin | 0.114 | 0.216 | 4 |
| high | Eli Drinkwitz | 0.113 | 0.209 | 5 |
| high | Mike Elko | 0.103 | 0.208 | 4 |
| high | Troy Calhoun | 0.101 | 0.205 | 4 |
| low | Will Hall | -0.116 | 0.216 | 4 |
| low | Ken Wilson | -0.116 | 0.230 | 2 |
| low | Kenni Burns | -0.113 | 0.229 | 2 |
| low | Stan Drayton | -0.094 | 0.211 | 3 |
| low | Ryan Walters | -0.084 | 0.211 | 2 |
| low | Butch Jones | -0.082 | 0.194 | 5 |
| low | Kevin Wilson | -0.079 | 0.210 | 2 |
| low | Jay Norvell | -0.079 | 0.191 | 5 |
| low | Will Healy | -0.076 | 0.205 | 2 |
| low | David Shaw | -0.074 | 0.202 | 2 |

#### Offense rating
| side | coach | effect | posterior SD | seasons |
|---|---|---:|---:|---:|
| high | Dan Lanning | 0.313 | 0.272 | 4 |
| high | Lance Leipold | 0.294 | 0.247 | 5 |
| high | Josh Heupel | 0.255 | 0.242 | 5 |
| high | Eric Morris | 0.247 | 0.264 | 3 |
| high | Jeff Monken | 0.227 | 0.231 | 5 |
| high | Mario Cristobal | 0.220 | 0.225 | 5 |
| high | Lincoln Riley | 0.209 | 0.215 | 5 |
| high | Kirby Smart | 0.207 | 0.225 | 5 |
| high | Clay Helton | 0.191 | 0.223 | 4 |
| high | Sonny Dykes | 0.187 | 0.214 | 5 |
| low | Will Hall | -0.320 | 0.276 | 4 |
| low | Ken Wilson | -0.241 | 0.274 | 2 |
| low | Kenni Burns | -0.241 | 0.275 | 2 |
| low | Stan Drayton | -0.214 | 0.246 | 3 |
| low | Jimbo Fisher | -0.203 | 0.237 | 3 |
| low | Kirk Ferentz | -0.171 | 0.240 | 5 |
| low | Joe Harasymiak | -0.156 | 0.252 | 1 |
| low | Butch Davis | -0.154 | 0.264 | 1 |
| low | Scotty Walden | -0.153 | 0.236 | 2 |
| low | Troy Taylor | -0.152 | 0.239 | 2 |

#### Defense rating
| side | coach | effect | posterior SD | seasons |
|---|---|---:|---:|---:|
| high | Curt Cignetti | 0.204 | 0.229 | 3 |
| high | Hugh Freeze | 0.167 | 0.192 | 5 |
| high | James Franklin | 0.163 | 0.200 | 4 |
| high | Pat Narduzzi | 0.151 | 0.192 | 5 |
| high | Kyle Whittingham | 0.123 | 0.175 | 5 |
| high | Brent Venables | 0.117 | 0.186 | 4 |
| high | Jim Harbaugh | 0.115 | 0.186 | 3 |
| high | Eli Drinkwitz | 0.111 | 0.174 | 5 |
| high | Kirk Ferentz | 0.108 | 0.170 | 5 |
| high | Jon Sumrall | 0.101 | 0.171 | 4 |
| low | Bronco Mendenhall | -0.124 | 0.190 | 3 |
| low | Jamey Chadwell | -0.120 | 0.174 | 5 |
| low | Mack Brown | -0.117 | 0.181 | 4 |
| low | Chris Creighton | -0.116 | 0.182 | 5 |
| low | Will Healy | -0.113 | 0.192 | 2 |
| low | Trent Dilfer | -0.109 | 0.179 | 3 |
| low | Michael Desormeaux | -0.105 | 0.173 | 4 |
| low | Thomas Hammock | -0.102 | 0.179 | 5 |
| low | Clay Helton | -0.100 | 0.180 | 4 |
| low | Mike Neu | -0.097 | 0.169 | 4 |

### Interpretation boundary

This phase estimates association after observed roster covariates and partial pooling. It does not establish a causal coach effect and does not satisfy the temporal prediction contract. The next phase must refit every coach estimate inside expanding folds using seasons no later than N−1. CFBD has no coordinator history, so unit-level OC/DC changes remain untested.

Machine-readable results are in `audit/coach_variance_decomposition.json`; all posterior coach estimates are in `audit/coach_effect_estimates.csv`.

## Phases 3–4 — leakage-safe mean features

Predictive assignments do **not** reuse the descriptive max-games attribution. The
preseason reconstruction retains a returning incumbent when an in-season interim is
also listed, excludes post-August hires, and zeroes unresolved multi-coach rows. It
never chooses a coach using season-N games. Coverage is 100% in 2021, 2022, and 2024,
99.25% in 2023, and 99.26% in 2025.

For each target season N, coach effects are rebuilt from completed outcomes through
N−1. The empirical-Bayes approximation residualizes the same prior offense, prior
defense, talent, and returning-production observables, then alternates partially
pooled program and coach intercepts. Features enter as antisymmetric team differences.
The mixed family adds coach×talent, coach×returning, first-year×talent, and
first-year×returning terms. The requested mediator diagnostic is small:
`corr(hc_first_year, returning) = −.020` and
`corr(hc_first_year, talent) = +.021`.

Pooled 2022–25 results through the existing v4 tuning and weekly replay:

| candidate | static Brier | online Brier | static Δ vs core | online Δ vs core |
|---|---:|---:|---:|---:|
| clean core | .20958 | .18824 | — | — |
| coach mean / tenure / change | .21166 | .18933 | +.00209 | +.00109 |
| separate O/D coach effects | .21159 | .18938 | +.00201 | +.00113 |
| mixed coach interactions | .21248 | .18961 | +.00290 | +.00137 |

Positive deltas are worse. The coach-mean degradation has a season-week bootstrap
95% interval of `[+.00092, +.00332]` static and `[+.00015, +.00204]` online.
The clean core is selected in every outer fold; no coach family approaches the
predeclared +.001 improvement in either mature selection window. The interaction
family is the worst candidate, so the data specifically reject the idea that these
simple mixed effects rescue the mean channel.

## Phase 5 — staff change in the uncertainty channel

The rating experiment holds mean shrinkage at the shipping `λ=.70`. A first-year
staff receives more regression toward the talent baseline and established staffs
receive correspondingly less, so this tests the *shape* of uncertainty rather than
quietly increasing average shrinkage.

| uncertainty shape | mean O/D year-to-year r | mean O/D RMSE | static Brier | online Brier |
|---|---:|---:|---:|---:|
| no rating shrink (v4 historical core) | .50522 | .99099 | .20958 | .18824 |
| matched flat `λ=.70` | **.55243** | **.84908** | .21052 | .18924 |
| first-year shock .10 | .55050 | .84930 | .21095 | .18946 |
| first-year shock .20 | .54184 | .85157 | .21205 | .19006 |
| first-year shock .30 | .52754 | .85587 | .21366 | .19087 |

Flat shrinkage materially improves the rating target, but the coach-specific shock
does not: even .10 reduces rating correlation and adds +.00043 static / +.00023
online Brier relative to the matched flat control. Larger shocks get monotonically
worse. The .30 shock degradation versus flat has bootstrap intervals
`[+.00096, +.00528]` static and `[+.00016, +.00312]` online.

## Adoption verdict

**Reject all coach-derived production features.** The partially pooled descriptive
coach share is small and weakly separated from program effects; every leakage-safe
mean candidate worsens games; and a first-year uncertainty shock worsens both the
rating target and games relative to matched flat shrinkage. Code and evidence remain
for reproducibility, but the production payload and JS port are unchanged.

```powershell
python -m scripts.coach_effects --phase2
python -m scripts.coach_predictive_backtest
```

Predictive artifacts are `artifacts/coach_predictive_backtest.json` and `.csv`.
