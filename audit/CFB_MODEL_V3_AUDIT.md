# CFB Model V3 — adversarial team-prediction audit

Audit date: 2026-08-12
Scope: the active team-prediction path, its player/WAR inputs, supporting experiments, saved artifacts, and available historical data.
Primary empirical sample: 2,913 matched FBS-v-FBS games in expanding-window tests for 2022–2025.

## Executive verdict

**Adjusted overall score: 32/100 — Not statistically defensible in its current validated form.**

The repository contains several good statistical ideas and an unusually candid record of prior failures. The final team model also has real predictive signal: in an expanding-window test it improves materially on an aligned home-only baseline (Brier 0.2020 versus 0.2424), produces a margin correlation of 0.535, and is competitive with pregame Elo in Weeks 1–4 (Brier 0.1803 versus 0.1808).

Those facts do not rescue the current validation. The strongest incremental input—the player-derived roster signal—is historically constructed using the season being predicted to decide which players are on the roster and how many snaps weight them. The PFF path takes season-N graded participants and season-N snaps, attaches season-(N−1) grades, and calls the result an entering-N roster. The WAR path likewise uses season-N WAR rows to define season-N roster membership before carrying forward prior WAR. This is post-treatment information for a preseason forecast. It is not a subtle theoretical objection; removing PFF and WAR worsens the reported expanding-window Brier by 0.0072, so the contaminated layer is responsible for a large share of the measured advantage.

The second major problem is that player/facet/position weights and many architecture choices were learned once using outcomes across the same seasons later called held out. The repository’s own nested-CV artifact admits 0.00528 Brier of selection optimism and lists six important choices that remain outside the nested loop. That nested procedure also uses future seasons to train predictions of earlier seasons and does not rebuild the historical player layer inside each fold.

Finally, the primary model loses decisively to the aligned CFBD pregame Elo benchmark on the same 2,913 games: Brier 0.2020 versus 0.1898, log loss 0.5870 versus 0.5613, and accuracy 67.9% versus 71.0%. The advertised “matchup” structure has almost no incremental value: replacing it with one scalar team-strength difference changes Brier by only +0.00048. There are no learned football-style interactions such as pass offense × pass defense, OL × pass rush, QB × coverage, or run offense × run defense.

The model is therefore best described as a **promising static preseason rating model whose player-input validation is contaminated and whose matchup/prediction claims exceed the evidence**.

## Audit method and evidence boundaries

Observed findings below come from:

- direct code tracing of the active `scripts/train.py` path and all functions it calls;
- reproduction of the shipped 2025 holdout: 719 games, Brier 0.2023, log loss 0.5880, accuracy 68.4%;
- independent expanding-window tests: train only on seasons earlier than the test season, for 2022, 2023, 2024, and 2025;
- aligned ablations and baselines on the same 2,913 games;
- direct workbook inspection of the supplied two-deep/PFSN workbook;
- repository artifacts for WAR projection, nested CV, collinearity, external validation, and interval coverage.

The independent harness is `audit/run_audit.py`; its principal outputs are `audit/output/audit_metrics.json`, `benchmark_ablation.csv`, `fold_metrics.csv`, `detailed_predictions.csv`, `team_residuals.csv`, and `conference_residuals.csv`.

No betting-line file is present. A closing-spread benchmark, ATS error, and market probability comparison could not be performed. The available game payload has CFBD pregame Elo but not market spreads. The repository also lacks historical preseason roster snapshots/depth charts, so the magnitude of performance after fully removing target-season participation leakage cannot be measured without new data.

## 1. Architecture map and active objective

```text
CFBD prior-season advanced team statistics ──┐
TruMedia prior-season team statistics ───────┼─> within-season z-scores
                                             │        ├─> mean(5 offense stats) = O
                                             │        └─> mean(5 defense stats) = D
                                             │                     │
prior-season schedule/results ───────────────┘                     └─> centered SRS opponent adjustment

PFF grades + season-N participants/snaps ─> position-room weighted grades ─┐
PFF/CFBD facets ─> player WAR ─> season-N participant roster × prior WAR ───┼─> scalar talent z
CFBD recruiting talent ──────────────────────────────────────────────────────┘
                                                                            │
                                          O,D = 30% prior adjusted O/D + 70% talent-implied O/D
                                                                            │
                                             game features (home perspective)
                                             ├─ O_home − D_away
                                             ├─ D_home − O_away
                                             ├─ returning_home − returning_away
                                             └─ home-field indicator
                                                                            │
                                      L2 logistic win model + ridge margin model
                                      p = 0.4 logistic + 0.6 Φ(margin / sigma)
                                                                            │
                                      win probability, margin, power ranking
                                      optional in-season Elo / playoff simulation
```

The active route is `scripts/train.py:221-265`. `load_bundle()` obtains standardized team stats, recruiting talent, returning production, games, and Pythagorean records. `blended_talent()` combines PFF, CFBD, and WAR (`scripts/train.py:26-66`). Opponent-adjusted O/D is built at `scripts/train.py:237`; talent slopes and game rows are created at `scripts/train.py:239-247`; and the final logistic/ridge model is fitted at `scripts/train.py:251-252`.

Several mathematically interesting modules are not in the shipped team training path. `src/projection.py` is an alternative per-stat projection system; `src/spread.py` is a separate points-scored model; many `scripts/loso*` files are experiments; the EA layer affects 2026 player projections but not historical team-stat construction. Their existence should not be interpreted as functionality of the primary model.

## 2. Foundational assumptions — 45/100

### Player-level assumptions

| Assumption | Where it enters | Assessment | Consequence if false | Better alternative |
|---|---|---|---|---|
| Prior PFF grade predicts future contribution | `src/data/pff.py:216-260` | Directionally defensible, but grade is context/scheme dependent and only 61.6% of season-N roster rows match a prior grade. | Talent becomes a mixture of ability, role, teammate quality, and replacement imputation. | Hierarchical player model with position, age/class, team/scheme, opponent, and measurement error. |
| Player WAR is on a transferable wins scale | `war_model/build_hybrid.py:538-776` | Partly defensible as a predictive index, not established as causal wins added. Facets include teammate-dependent measures; repository external validation reports partial correlation 0.369 historically. | Transfer and lineup effects are overstated; attribution is mistaken for causal value. | Treat WAR as a regularized latent rating with explicit uncertainty, not literal causal wins. |
| Players can be compared across positions using learned weights | PFF weights at `src/data/pff.py:24-30`; WAR weights at `war_model/build_hybrid.py:272-468` | Weakly identified. PFF weights were optimized on win percentage without fold-specific refitting. WAR’s 86 facets have 23 sign flips in repeated fits and 86-feature VIFs as high as 54.8. | Position value and player rankings change with sample or redundant metric choice. | Fit position blocks inside temporal outer folds; report posterior/bootstrapped bands and avoid causal labels. |
| Unknown players are replacement level | `src/data/pff.py:235-257`, `406-425` | More defensible than dropping them, but the 10th percentile is arbitrary and not roster-specific. | Freshmen/JUCO/FCS transfers are systematically biased low; elite recruits are misrepresented. | Position/class/recruiting prior with partial pooling and explicit uncertainty. |

### Aggregation assumptions

The PFF path is linear throughout. Within a position group it uses a snap-weighted mean of prior grades (`src/data/pff.py:245-255`); across positions it uses a fixed weighted mean (`src/data/pff.py:264-277`). The 2026 path replaces unavailable snaps with empirically averaged depth shares (`src/data/pff.py:286-301`, `396-404`). WAR is summed by team (`src/data/war.py:87-107`, `111-117`). PFF, recruiting talent, and WAR are then linearly blended (`scripts/train.py:26-66`).

This assumes:

- contributions are additive or averageable;
- marginal value is constant regardless of teammates or scheme;
- a position room is represented by its weighted mean, not its weakest link, distribution, or actual starting lineup;
- the entire player system can be compressed to one scalar before it reaches offense and defense;
- the same scalar talent axis can anchor both offense and defense through two fitted slopes;
- losing a player is valued against a generic replacement or zero prior WAR, not the actual backup.

These assumptions are too strong for football. There is no OL weakest-link term, no QB × OL or QB × receiver interaction, no pass-rush × coverage complementarity, no diminishing return for a third elite receiver, and no explicit actual-backup replacement. A quarterback and guard differ only through upstream weight shares inside the scalar talent index. The final team model never sees position identity.

### Team-level assumptions

The model represents prior offense, prior defense, recruiting/player talent, returning production, opponent strength, and a global home-field effect. Important absences are special teams (the only proxy, `fp_margin`, is zeroed), coaching, scheme, explicit quarterback status, injuries in historical validation, pace, game state, garbage time, conference effects, weather, travel, continuity by unit, and in-season information in the primary static predictor.

The model assumes last season’s full-season aggregates remain a useful baseline for every game in the following season. That is reasonably true early; the evidence shows degradation from Brier 0.1803 in Weeks 1–4 to 0.2124 in Weeks 10+, while in-season Elo scores 0.1932 from Week 5 onward.

## 3. Player → team aggregation — 38/100

### Exact implemented functions

1. **Player → position group (historical PFF):**

   \[
   G_{t,p,N} = \frac{\sum_{i\in(t,p,N)} \text{snaps}_{i,N}\,\text{grade}_{i,N-1}}
                         {\sum_{i\in(t,p,N)} \text{snaps}_{i,N}}
   \]

   Unknown prior grades are credited at the season-position 10th percentile. This is implemented at `src/data/pff.py:229-260`.

2. **Position groups → PFF roster talent:**

   \[
   T^{PFF}_{t,N}=\frac{\sum_p w_pG_{t,p,N}}{\sum_{p\ available}w_p}
   \]

   followed by a within-season z-score (`src/data/pff.py:264-277`). PFF code weights are QB .19, DT/LB/OL .12 each, WR .10, EDGE/SAF .08, CB/TE .07, RB .03 (`src/data/pff.py:27-30`).

3. **Player WAR → team WAR:** prior player WAR is carried one season forward and summed over season-N participant rows (`src/data/war.py:94-107`), then standardized within season (`src/data/war.py:120-140`).

4. **Talent blend:**

   \[
   T^{base}=0.5T^{PFF}+0.5T^{CFBD},\qquad
   T=0.6T^{base}+0.4T^{WAR}.
   \]

   The implementation is `scripts/train.py:37-65` with weights from `config.py:193` and `config.py:226`.

5. **Talent → O/D baseline:** for training years, slopes through the origin regress prior adjusted O and D on entering talent (`src/matchup.py:72-97`). With flat `u=1`, shipped ratings are:

   \[
   O'=0.3O+0.7b_OT,\qquad D'=0.3D+0.7b_DT
   \]

   (`src/matchup.py:131-136`, `config.py:125-131`).

### Depth and replacement

There is real effort to model depth. The 2026 two-deep path uses historical snap-share curves rather than a guessed starter/backup multiplier. It credits unresolved players at replacement and keeps every listed player. The supplied workbook itself is much weaker—only 1,493 of 5,662 rows match a PFSN score, only four teams have all ten group scores, and its primary position weights are based on 31 complete teams in 2025—but that workbook’s direct team score is not the active final roster-talent path.

Historical validation is the fatal exception: actual season-N snaps determine weights, and actual season-N PFF participation determines the roster. That is neither a preseason depth forecast nor a roster snapshot.

### Empirical player ablations

| Component removed | Configured Brier | New Brier | Change | Accuracy change | Conclusion |
|---|---:|---:|---:|---:|---|
| PFF + WAR player evaluations | 0.2020 | 0.2092 | +0.0072 | −1.79 pp | Large apparent value, but contaminated by target-season roster/snaps and global outcome-fitted weights. |
| WAR only | 0.2020 | 0.2040 | +0.0020 | −0.52 pp | Modest apparent value; not cleanly causal or leakage-free. |
| Dedicated QB player signal | 0.2020 | 0.2014 | **−0.0007** | +0.17 pp | Removing QB-specific PFF/WAR improves pooled results; it improves 2023–2025 and hurts only 2022. |
| All talent shrink | 0.2020 | 0.2119 | +0.0099 | −1.82 pp | Talent is important, but this combines recruiting and contaminated player inputs. |
| Prior team-stat component (talent-only O/D) | 0.2020 | 0.2067 | +0.0047 | −0.62 pp | Prior team production retains incremental value. |

The correct conclusion is not “player evaluation adds 0.0072 Brier.” The correct conclusion is: **the current player-derived column adds 0.0072 in a backtest that uses target-season participation and snaps, so its clean incremental value is unknown.**

## 4. Mathematical grounding — 58/100

### Sound elements

- Within-season z-scores are dimensionally appropriate for combining heterogeneous team statistics (`src/features.py:21-34`).
- Defensive signs are consistently flipped so higher is better.
- Active O/D/game features are regularized and well conditioned: off-edge/def-edge correlation 0.613, VIF 1.61/1.62, returning VIF 1.03, home-field VIF 1.00, standardized condition number 2.09.
- The SRS solver correctly projects out unidentified constants and solves the centered linear system directly (`src/oppadj.py:47-85`). This is a material improvement over the former nonconvergent iteration at alpha≈1.
- Logistic Brier selection and ridge margin fitting are conventional (`src/model.py:101-142`).

### Weak or invalid elements

1. **O/D composites are arbitrary equal means.** Five offensive and five defensive z-scores are averaged at `src/matchup.py:40-44`. Equal weight is not empirically justified, does not account for measurement reliability, and ignores covariance.

2. **The “matchup” algebra is nearly a scalar rating.** Features are `O_h-D_a` and `D_h-O_a` (`src/matchup.py:159-179`). Their sum is exactly `(O_h+D_h)-(O_a+D_a)`. A one-column scalar model is only 0.00048 Brier worse. No nonlinear interaction exists.

3. **Neutral-site probabilities violate reciprocity.** At a neutral site, a coherent binary model should satisfy `P(A beats B)+P(B beats A)=1`. The implemented feature order swaps offense and defense when teams reverse, while coefficients differ and both submodels have intercepts. Across sampled 2025 team pairs, mean absolute violation is 0.0079 and maximum is 0.0384. Predicted neutral margins sum to 0.54 points on average and as much as 1.64, rather than zero.

4. **Normal residual conversion is treated as probability uncertainty.** `Phi(pred_margin/sigma)` uses the training residual SD (`src/model.py:121-132`, `57-63`). Residuals are not demonstrated normal, homoskedastic, or independent; a single sigma cannot represent team/week-specific uncertainty.

5. **Missing team stats become exactly average.** `src/features.py:31-33` fills missing standardized values with zero without a missingness indicator. This is benign for genuinely unavailable external sources only if missingness is unrelated to quality; that is not established.

6. **Through-origin talent slopes impose zero intercept.** With standardized variables an intercept should be near zero, but pooling seasons/conferences without random effects still makes this a strong restriction (`src/matchup.py:92-97`).

There is no PCA in the active team model. PCA exists in WAR experiments, where the repository does standardize before PCA (`war_model/build_hybrid.py:420-428`), but it is not shipped as the team predictor and cannot be credited for team performance.

## 5. Statistical validity — 42/100

Games are not independent. The same team appears roughly weekly, opponents connect observations, team quality persists across seasons, and conferences/schedules create clusters. The active model uses ordinary logistic/ridge fits over game rows. Hyperparameter CV is `cv=5` without season or team groups (`src/model.py:107-125`). This understates selection uncertainty and allows adjacent games involving the same teams into different inner folds.

The data-generating process is hierarchical—player → unit → team → conference → season → game—but the final estimator is not. There are no team random effects, conference effects, coach/scheme regimes, season intercepts, or clustered standard errors. Opponent adjustment partly addresses schedule, but not dependence.

Selection and playing-time bias are material:

- only players who appear in season-N PFF/WAR data define the historical roster;
- season-N snaps upweight players who stayed healthy and won roles;
- players without prior grades are assigned replacement even when they are elite recruits;
- FBS teams missing prerequisite data and games involving unmatched teams are dropped. The aligned evaluation uses 2,913 games, while raw completed game tables contain many more games, largely FBS–FCS or teams without full frames.

The model separates some ability from schedule through the SRS layer. Empirically, removing opponent adjustment worsens Brier from 0.2020 to 0.2053 and margin RMSE from 17.22 to 17.48, so it contributes real signal. It does not separate game state/garbage time or estimate latent ability jointly with schedule and observation noise.

## 6. Data leakage and temporal integrity — 25/100

### Severe leakage: historical PFF roster and playing time

For season N, `build_group_scores()` creates the roster from `g[g.season.isin(usable)]`, where `g` is the season-N PFF grade/participation table, merges the prior grade, and then weights by the `snaps` already attached to those season-N rows (`src/data/pff.py:216-255`). Therefore a preseason prediction of N knows:

- who eventually recorded a PFF row in N;
- which team a transfer actually represented in N;
- how many snaps each player actually played in N;
- indirectly, injuries, benchings, breakout roles, and late roster changes.

That information is not known at preseason prediction time.

### Severe leakage: historical WAR roster membership

`team_war_by_year()` builds a season-N roster from the season-N player WAR table, then joins season-(N−1) WAR (`src/data/war.py:87-107`). Even though the value is lagged, inclusion and destination come from realized season-N participation. A player who never plays is absent; a player who emerges is included; transfer destination is inferred from the completed season dataset.

### Global target-informed weights

- PFF position weights were optimized via NNLS against season win percentage across the available sample (`scripts/compare_signals.py:52-82`) and then hard-coded.
- WAR concept/facet weights are fitted against next-season adjusted wins using the full artifact build (`war_model/build_hybrid.py:386-468`). They are not rebuilt inside each team-game outer fold.
- `config.py:75` explicitly says an older shrinkage value was selected on held-out 2025.
- `config.py:169-171` acknowledges opponent-alpha selection on the same LOSO used for reporting.

### Validation reuse

The main 2025 split trains only on earlier team-game seasons, which is good. But model architecture, feature retirement, blends, alpha, uncertainty shrink, ensemble weight, score shape, player weights, and WAR construction were repeatedly chosen after examining 2021–2025 performance. `scripts/nested_cv.py:1-24` explicitly documents this. Its saved result is:

- configured LOSO Brier: 0.20258;
- nested outer Brier: 0.20786;
- selection optimism: +0.00528 Brier (+2.6%);
- untuned inside: feature set, Pythagorean exponent, shrinkage lambda, ensemble weight, score shape, EA threshold.

The nested procedure is an improvement, not a certificate. Early outer seasons are trained using later seasons, and player artifacts/position weights are fixed globally rather than rebuilt from each outer training set.

### Leakage-free elements

Prior-season CFBD/TruMedia team stats, prior-season schedules, entering-season recruiting talent, and entering-season returning production are conceptually available before the season. Their per-season standardization is also safe if performed after the full preseason input population is frozen. These safe elements should become the basis of a clean rebuild.

## 7. Multicollinearity and feature independence — 67/100

### Active team model

The current active design is healthy after retiring field position, Pythagorean, and the duplicate talent column:

| Active term | VIF |
|---|---:|
| off_edge | 1.607 |
| def_edge | 1.620 |
| returning | 1.031 |
| home field | 1.003 |

Condition number is 2.09. The off/def correlation of 0.613 is meaningful but not severe. Regularization is appropriate. The zeroed retired columns remain in the serialized six-wide shape (`src/matchup.py:138-155`); this is operationally convenient but can confuse consumers into thinking six independent signals remain.

### Player/WAR layer

The upstream layer is heavily redundant. The saved 86-feature audit has VIF 54.8 for `WR_pass_route` and 43.9 for `WR_offense`; 80% of variance requires far fewer than 86 components; within-position facets reach correlations near 0.93. The repository recognizes that NNLS arbitrarily zeroes members of correlated clusters (`war_model/two_level_weights.py:13-25`). Two-level weighting improves attribution stability but does not make redundant football concepts independent.

The final scalar blend also combines correlated signals: saved correlations are PFF–WAR 0.668, PFF–CFBD 0.645, and WAR–CFBD 0.625. A linear blend can be predictive, but interpreting 40% WAR as 40% distinct information is wrong.

## 8. Game prediction performance — 50/100

### Reproduced shipped split

Train 2021–2024, test 2025:

| Split | Games | Brier | Log loss | Accuracy |
|---|---:|---:|---:|---:|
| Train | 2,894 | 0.2016 | 0.5856 | 67.62% |
| Test 2025 | 719 | 0.2023 | 0.5880 | 68.43% |

The small train/test gap is encouraging, but the test is not independent of architecture selection or player-weight construction.

### Expanding-window benchmark

| Model | N | Brier | Log loss | Accuracy | Margin MAE | Margin RMSE | Margin R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| Configured model | 2,913 | 0.2020 | 0.5870 | 67.87% | 13.67 | 17.22 | 0.280 |
| One scalar team-strength difference | 2,913 | 0.2025 | 0.5887 | 67.49% | 13.68 | 17.22 | 0.280 |
| Prior Pythagorean only | 2,913 | 0.2174 | 0.6246 | 64.64% | 14.61 | 18.45 | 0.173 |
| Home only, aligned | 2,913 | 0.2424 | 0.6778 | 58.81% | — | — | — |
| CFBD pregame Elo, aligned | 2,913 | **0.1898** | **0.5613** | **70.99%** | — | — | — |

The model has legitimate preseason signal and is far better than trivial baselines. It does not beat a straightforward in-season Elo-style benchmark. On the identical sample, Elo improves Brier by 0.01219 (6.0% relative) and accuracy by 3.12 percentage points.

No closing-market benchmark can be reported because no spread/odds dataset is present.

## 9. Calibration — 58/100

Pooled expanding-window calibration intercept is +0.072 and slope is 0.860. The positive intercept means home-perspective probabilities are slightly low overall; slope below one indicates overconfidence.

| Probability bucket | N | Mean predicted | Actual win rate | Error |
|---|---:|---:|---:|---:|
| 0–20% | 158 | 13.6% | 18.4% | −4.8 pp |
| 20–35% | 384 | 28.3% | 32.0% | −3.7 pp |
| 35–50% | 574 | 42.9% | 47.0% | −4.2 pp |
| 50–65% | 586 | 57.6% | 56.7% | +0.9 pp |
| 65–80% | 648 | 72.5% | 70.8% | +1.6 pp |
| 80–100% | 563 | 88.7% | 87.4% | +1.3 pp |

The model is systematically too pessimistic about away/nominal underdogs and mildly too confident about favorites. Away-favorite games have Brier 0.2237 versus 0.1886 for home favorites. Calibration slopes vary markedly: 0.708 in 2022, 1.012 in 2023, 0.846 in 2024, and 0.968 in 2025. A single global mapping is not stable enough to call fully calibrated.

## 10. Residual and error diagnostics — 45/100

Margin residuals are reasonably centered overall (predicted-minus-actual bias −0.38 points), but they are not structure-free.

- Probability Brier deteriorates from 0.1803 in Weeks 1–4 to 0.2079 in Weeks 5–9 and 0.2124 in Weeks 10+.
- Team identity explains 5.75% of probability-residual variance versus 2.30% under label permutation; permutation p=0.002. Persistent team misrating remains.
- Away favorites are underpredicted: home-perspective probability residual +0.041 and margin residual +2.49 points.
- Conference margin biases range from −1.42 points for the MAC to +1.54 for FBS Independents. Conference RMSE ranges from 15.62 (ACC) to 19.33 (Big 12).
- The largest persistent margin biases include Kennesaw State +16.1 (11 games), James Madison +13.0 (34), Massachusetts −11.7 (43), Navy +11.1 (33), and Georgia State −9.9 (44).

The late-season degradation and significant team clustering point to omitted dynamic team information—quarterback changes, injuries, coaching/scheme shifts, and newly observed team quality—not merely random football variance.

## 11. Team rating validity — 52/100

The opponent-adjusted O/D ratings have useful signal. The saved rating-vs-realized artifact reports offense correlation 0.617 and defense correlation 0.597 over 653 team-seasons, and removing opponent adjustment degrades game prediction consistently in every expanding fold (+0.0021 to +0.0050 Brier).

However:

- O/D are equal-weight averages of selected stats, not latent ratings jointly estimated from games;
- 70% shrink toward one talent axis makes offense and defense strongly dependent on the same roster scalar;
- the ratings are static for the entire following season;
- current-season player participation leaks into historical talent;
- power rating is the mean of first-listed matchup probabilities against all opponents (`src/matchup.py:222-231`), but neutral pair probabilities are not complements, so the power metric inherits a mathematical ordering asymmetry;
- circular schedule reinforcement is controlled by the centered SRS solve, but the chosen alpha is still outcome-tuned and close to a sensitive region.

The ratings distinguish broad elite/average/poor quality but should not be interpreted as clean, neutral, causal offense/defense estimates.

## 12. Matchup logic — 25/100

The model’s only matchup terms are offense-vs-defense differences. They contain no style variables and no interaction products. Algebraically the scalar `(off_edge + def_edge)` is ordinary total team strength difference. Empirically:

- configured Brier: 0.2020;
- scalar strength Brier: 0.2025 (+0.00048);
- O/D edges with returning removed: 0.2024 (+0.00034).

The small difference does not justify calling the architecture meaningfully matchup-specific. A genuine test would include predeclared interactions such as pass efficiency × coverage, pressure × pass protection, rush efficiency × run-front efficiency, QB mobility × containment, and explosiveness × explosive-play prevention, each evaluated out of sample against the scalar baseline.

## 13. Robustness and sensitivity — 42/100

Positive evidence:

- active-design VIF and condition number are low;
- expanding-fold Brier is fairly bounded: 0.1949–0.2061;
- opponent adjustment improves all four expanding folds;
- model fitting is deterministic, so ordinary random seed sensitivity is negligible at the final linear stage.

Negative evidence:

- nested selection raises Brier from 0.20258 to 0.20786;
- inner folds choose talent blend 0.3, 0.5, or 0.7, showing weak identification;
- several important settings remain outside the nested loop;
- removing QB-specific player information improves three of four recent seasons and pooled performance;
- scalar strength is essentially tied with the claimed matchup model;
- WAR has 23 coefficient sign flips in repeated fits and highly redundant facets;
- a true leave-one-season-out rebuild of player weights, historical roster forecasts, and team model does not exist.

The model is robust to small changes in its final three active columns, but not demonstrated robust to how the player/talent column was created or how the architecture was selected.

## 14. Generalization — 40/100

The team-game layer generalizes moderately across 2022–2025, and each test fold uses earlier game seasons in this audit. The architecture has not been demonstrated to generalize across fully unseen eras because:

- player and position weights are global artifacts informed by later outcomes;
- COVID 2020 is excluded, leaving no stress test for environment change;
- only four expanding test seasons are available;
- coaching regime, transfer-rule, conference-realignment, and style changes are not modeled;
- no external league/market benchmark is used for preseason ratings;
- performance deteriorates as each season progresses.

The cleanest defensible statement is: **prior team/talent information predicts future games, but the incremental generalization of the repository’s player architecture is unproven.**

## 15. Model logic and architecture — 50/100

Every active stage has a plausible purpose, but several discard the football structure they claim to preserve:

| Transition | Information introduced | Information discarded / duplicated | Audit judgment |
|---|---|---|---|
| Raw player facets → WAR | play quality, volume, position | scheme/context separation, causal identification | Useful index; overinterpreted as wins. |
| Players → room mean/sum | depth/volume | weak link, actual replacement, interaction, distribution | Too linear. |
| Rooms/WAR/recruiting → one talent z | multiple talent sources | offense/defense/position identity | Major bottleneck. |
| Team stats → O/D equal means | interpretable unit axes | feature reliability, pace/game state | Simple but arbitrary. |
| O/D → SRS | schedule strength | game-level uncertainty | Empirically useful. |
| O/D + talent shrink | continuity/regression | unit-specific continuity, team-specific uncertainty | Overaggressive fixed 70% scalar shrink. |
| O/D edges → prediction | opponent unit strength | style, interaction, dynamics | Nearly scalar; label overstates logic. |
| Logistic + normal-margin blend | outcomes and margin information | distribution shape, heteroskedasticity | Reasonable ensemble; incompletely calibrated. |

The repository has substantial parallel/dead experimental code. This is valuable research history, but it obscures the one production graph and increases the chance that the app, scripts, and artifacts describe different models. The code comments document several prior cases where exactly that happened.

## 16. Interpretability — 55/100

The final linear model can be decomposed, but the upstream player contribution cannot be traced cleanly into a game because all positions collapse to talent and talent is then embedded inside O/D while its explicit feature is zeroed.

Representative held-out 2025 game: Michigan home vs Ohio State. Entering ratings yield:

- off edge: `0.290 − 1.121 = −0.831`; logistic contribution `0.954 × −0.831 = −0.793`;
- def edge: `0.680 − 1.229 = −0.549`; contribution `0.787 × −0.549 = −0.432`;
- returning edge: `−0.447 − (−0.768) = +0.321`; contribution `0.127 × 0.321 = +0.041`;
- home field: +0.342 logit;
- intercept: −0.058.

The raw logistic probability is approximately 28.9%. The ridge model predicts Michigan −9.79 points; converting through `Phi(margin/17.23)` and blending 40/60 gives 28.66%. Ohio State won by 18.

That explains the numerical prediction. It does **not** explain how much came from quarterback, OL, receivers, pass rush, secondary, special teams, coaching, or a specific matchup. Those contributions no longer exist as stable coordinates in the final model.

## 17. Uncertainty modeling — 28/100

The primary game model is deterministic. It has:

- one global margin residual sigma (17.23 points);
- no team-specific prediction interval;
- no early/late-season uncertainty difference;
- no propagation of player-rating, depth-chart, injury, transfer, or missing-data uncertainty into team ratings and game probabilities;
- a variable named “uncertainty” that is actually a shrinkage weight. It is flat for every team because returning production failed validation (`src/matchup.py:47-69`).

The WAR subsystem does estimate player/team projection uncertainty. Its 2025 validation reports 86.1% coverage for a nominal 68% interval and 97.8% for nominal 95%, so intervals are conservative. Its per-team uncertainty spread failed validation; the code intentionally exports only a scalar talent-noise SD of 0.359 (`src/data/war.py:168-180`). That scalar is used in playoff simulation, not propagated through ordinary game predictions.

This is a good research start but not uncertainty modeling for the deployed predictor.

## 18. Red-team conclusions

Alternative explanations for the apparent performance that do not require the model to understand team quality:

1. **Realized participation proxy:** season-N participants/snaps reveal health, starting roles, and who actually contributed.
2. **Global target-tuned weights:** later win outcomes influence upstream position/facet weights used in earlier “held-out” seasons.
3. **Recruiting/talent dominance:** at 70% shrink, a relatively simple talent prior drives O/D; the elaborate player attribution may add little clean information.
4. **Schedule structure:** team/conference identity and prior results persist; a static rating can look strong without representing specific matchups.
5. **Model-selection reuse:** repeated LOSO exploration over many knobs makes the best published configuration optimistic.

The red-team test succeeds: the repository’s own player ablation is strongest exactly where temporal leakage exists, and a simpler Elo benchmark is better.

## 19. Critical failure test — five deployment risks

| Rank | Issue | Evidence | Severity | Probability | Consequence | Fix |
|---:|---|---|---|---|---|---|
| 1 | Historical roster/snap leakage | `src/data/pff.py:229-255`; `src/data/war.py:94-107`; removing PFF+WAR costs 0.0072 Brier | Critical | High/certain in current backtest | Reported player value is inflated; forward performance disappoints | Acquire frozen preseason rosters/depth charts and forecast snap shares using only prior data. |
| 2 | Global outcome-fitted artifacts and incomplete nesting | nested Brier 0.20786 vs 0.20258; weights/features outside outer folds | Critical | High | Validation optimism and unstable architecture | Rebuild every learned artifact inside expanding outer folds; freeze a final design before last-season test. |
| 3 | Static model fails to ingest in-season evidence | model 0.2020 vs Elo 0.1898; Week 10+ Brier 0.2124 | High | Certain | Ratings become stale after injuries, QB changes, and team development | Make dynamic game-by-game rating the production layer, seeded by preseason model. |
| 4 | “Matchup” logic is mostly scalar and violates reciprocity | scalar +0.00048 Brier; neutral max probability complement error 3.84 pp | High | High | Incorrect matchup explanations and internally inconsistent neutral probabilities | Use antisymmetric team-difference features; add only predeclared interactions that improve temporal tests. |
| 5 | No game-level uncertainty propagation | global sigma only; per-team spread failed validation | High | High | Overconfident probabilities and misleading simulation tails | Hierarchical predictive distribution with player/roster/depth and residual uncertainty propagated by simulation. |

## 20. Five strongest components

1. **Centered exact opponent-adjustment solve.** `src/oppadj.py:47-85` identifies and removes the constant null direction and directly solves the fixed point. It is mathematically sound and empirically useful: +0.0033 Brier when removed.
2. **Prior-season temporal discipline for team statistics.** `src/features.py:47-70` and `src/matchup.py:100-130` use N−1 team performance for N games. This avoids the common “season average includes future weeks” leak.
3. **Simple, regularized final estimator.** The active design has low VIF/condition number, L2 logistic regression, and ridge margin regression. Complexity is restrained where it matters most.
4. **Empirical 2026 depth-share construction.** `src/data/pff.py:286-301`, `396-425` uses historical position-room snap shares and replacement credit rather than discarding unknowns or imposing one universal 1/.45 rule.
5. **Player projection has a genuine held-out signal.** The saved player projection artifact reports 5,927 holdout players, correlation 0.607 and MAE 0.0765, with ex-ante features after removing a leaked current-season starter flag. This is useful intermediate evidence, though it is not evidence of clean team-game lift.

## 21. Ten biggest weaknesses

| Rank | Current implementation | Problem | Consequence | Proposed solution | Expected effect |
|---:|---|---|---|---|---|
| 1 | Historical PFF uses season-N participants/snaps | Temporal leakage | Inflated player/team lift | Frozen preseason roster + ex-ante snap forecast | Trustworthiness; measured Brier may worsen before improving. |
| 2 | Historical WAR roster comes from season-N WAR rows | Temporal leakage/selection bias | Knows who played and transfer destination | Preseason roster transaction/depth snapshots | Same. |
| 3 | Player/position/facet weights fixed globally | Outer-test outcomes influence inputs | Optimistic validation | Fold-local artifact rebuild | Honest estimate; likely +0.003–0.008 Brier relative to reported. |
| 4 | Static preseason model is primary | Ignores revealed current-season ability | Loses to Elo by 0.0122 Brier | Dynamic Bayesian/Elo/Kalman update | High; benchmark shows attainable improvement. |
| 5 | Player talent compressed to one scalar | Position/unit information discarded | Cannot model lineup or matchup effects | Unit-specific latent talent and actual backup deltas | Medium/high if clean inputs exist. |
| 6 | Equal-mean O/D | Arbitrary weights and overlapping stats | Measurement noise/double counting | Reliability-weighted or regularized latent O/D | Medium; must beat equal mean OOS. |
| 7 | No real interactions | “Matchup” is scalar in disguise | No style-specific prediction | Predeclared, regularized interaction block | Unknown/small until proven; current ceiling <0.0005 Brier. |
| 8 | Neutral probabilities not reciprocal | Mathematical incoherence | Rankings and reversed matchups disagree | Antisymmetric logit/margin parameterization | Small metric gain, important correctness gain. |
| 9 | Persistent residual structure | Team identity R² 5.75%, p=.002 | Systematic over/underrating | Dynamic team effects and richer state variables | Medium. |
| 10 | Constant uncertainty and mild overconfidence | Calibration slope 0.860 | Bad risk decisions/simulation | Cross-fitted calibration + heteroskedastic predictive intervals | Medium for probability quality. |

## 22. Improvement roadmap

### Tier 1 — critical before trusting predictions

1. **Rebuild historical roster talent from information snapshots.**
   - Current problem: realized participants and snaps define preseason features.
   - Solution: store dated roster, transfer, injury, and depth-chart snapshots; predict snap shares from N−1 usage, class, recruiting, and announced depth only.
   - Justification: restores the required filtration `feature_time < kickoff_time`.
   - Difficulty: high.
   - Expected impact: unknown direction on accuracy, critical impact on validity.

2. **Use nested expanding-window validation for the entire graph.**
   - Current problem: weights and architecture are global; LOSO is reused.
   - Solution: for outer test year N, rebuild PFF weights, WAR facet weights, standardizers, roster model, alpha, shrinkage, feature selection, ensemble weight, and calibration using years <N only.
   - Justification: estimates the actual model-selection procedure, not one chosen model.
   - Difficulty: high/computationally heavy.
   - Expected impact: likely reveals several Brier points of current optimism; repository artifact already shows +0.00528.

3. **Make predictions algebraically reciprocal.**
   - Current problem: reversing a neutral game does not complement probability/margin.
   - Solution: model a single antisymmetric score `S(A,B)=-S(B,A)`; keep HFA as the only orientation term. If retaining O/D, enforce shared symmetric coefficients or construct explicit team rating differences.
   - Justification: required by binary probability coherence.
   - Difficulty: low.
   - Expected impact: modest accuracy, high correctness/interpretability.

4. **Adopt an in-season production update.**
   - Current problem: static model falls behind Elo after early weeks.
   - Solution: preseason model supplies priors; update offense/defense after each game with margin, opponent, recency, and uncertainty.
   - Justification: observed Brier gap 0.0122.
   - Difficulty: medium.
   - Expected impact: high.

### Tier 2 — high value

1. **Hierarchical latent team model.** Separate offense, defense, special teams, team, conference, and season effects; weight games by recency and garbage-time-adjusted possessions. Difficulty high; expected impact medium/high.
2. **Unit-specific player aggregation.** Carry QB, OL, receivers, rushing, front, LB, secondary, and special teams separately. Use actual backup-relative loss and nonlinear weak-link/diminishing-return functions learned only inside temporal folds. Difficulty high; expected impact unknown but essential to justify player architecture.
3. **Test genuine matchup interactions.** Add a small predeclared block and compare to the scalar baseline. Use hierarchical shrinkage/group lasso. Difficulty medium; expected impact likely small unless richer game/unit features are added.
4. **Cross-fitted calibration.** Estimate intercept/slope or beta calibration inside training years and apply once to the next year; evaluate by favorite, venue, week, conference, and probability bucket. Difficulty low; expected impact medium for log loss/Brier.
5. **Market benchmark dataset.** Add dated consensus open/close spreads and prices with timestamped joins. Difficulty medium; expected impact is diagnostic—essential for judging economic rather than academic value.
6. **Propagate uncertainty.** Sample player value, role/snap share, injuries, team latent strength, and residual outcome noise; produce team-specific intervals and calibrated win probabilities. Difficulty high; expected impact high for decisions/simulation.

### Tier 3 — refinements

1. Add missingness indicators and source/version hashes rather than neutral fill alone. Low difficulty; small/medium impact.
2. Consolidate active and experimental pipelines behind one explicit model manifest. Medium difficulty; high reproducibility impact.
3. Add invariance tests: neutral reciprocity, team-name mapping, feature-time audit, prediction parity between Python and JS, and artifact freshness. Low/medium difficulty; high reliability impact.
4. Report paired bootstrap/season-block uncertainty for every ablation; never report a third decimal without an interval. Low difficulty; interpretation impact.
5. Expand residual dashboards to coaching regime, QB starter, injuries, pace, garbage time, and roster turnover once those timestamped fields exist. Medium difficulty; medium diagnostic impact.

## 23. Final scorecard

| Category | Score /100 | Confidence | Biggest issue |
|---|---:|---|---|
| Foundational Assumptions | 45 | High | Linear scalar talent omits football interactions and actual replacement. |
| Player → Team Aggregation | 38 | High | Target-season participation/snaps and loss of position identity. |
| Mathematical Grounding | 58 | High | Equal means and neutral reciprocity failure. |
| Statistical Validity | 42 | High | Hierarchical/repeated data treated as ordinary game rows. |
| Data Leakage / Temporal Integrity | 25 | High | Historical roster and snap leakage; global target-informed artifacts. |
| Multicollinearity / Feature Independence | 67 | High | Team layer is clean; WAR layer remains highly redundant. |
| Game Prediction Performance | 50 | High | Real signal, but materially worse than aligned pregame Elo. |
| Calibration | 58 | High | Pooled slope 0.860; away-underdog/favorite asymmetry. |
| Residual Diagnostics | 45 | High | Significant persistent team residual structure and late-season decay. |
| Team Rating Validity | 52 | Medium | Useful opponent adjustment, but static and talent-dominated. |
| Matchup Logic | 25 | High | Scalar model is nearly identical; no genuine interactions. |
| Robustness / Sensitivity | 42 | Medium-high | Selection optimism and unstable player/architecture choices. |
| Generalization | 40 | Medium-high | Only four expanding tests; upstream artifacts are not fold-clean. |
| Model Logic / Architecture | 50 | High | Clear final model, but upstream complexity collapses into one scalar. |
| Interpretability | 55 | High | Final logit decomposes; player/unit contribution does not. |
| Uncertainty Modeling | 28 | High | Global sigma only; player uncertainty not propagated. |
| **RAW AVERAGE** | **45.0** | High | Arithmetic mean of 16 scores. |
| **ADJUSTED OVERALL SCORE** | **32.0** | High | Model-invalidating historical leakage and incomplete validation. |

### Adjusted-score penalties

- **−8 points: target-season roster/playing-time leakage.** This contaminates the strongest measured incremental component and prevents the backtest from reproducing a preseason information set.
- **−3 points: model selection and upstream weights not fully nested.** The repository itself measures +0.00528 Brier selection optimism; important stages remain outside that estimate.
- **−2 points: benchmark and coherence failures.** The primary model loses decisively to aligned pregame Elo and violates neutral probability reciprocity.

These penalties are not double punishment for ordinary weaknesses. They reflect defects that invalidate the interpretation of the main performance claim.

## 24. Final verdict questions

| Question | Answer | Evidence |
|---|---|---|
| 1. Is the foundational theory sound? | **PARTIALLY** | Prior performance, talent, opponent strength, and regression to mean are sound ideas; scalar/additive player and matchup assumptions are incomplete. |
| 2. Is player-to-team aggregation mathematically defensible? | **NO** | The formulas are computable but historically use future participation/snaps and omit nonlinear/backup/unit structure. |
| 3. Is the mathematics correct? | **PARTIALLY** | Z-scores, ridge/logit, and centered SRS are correct; neutral reciprocity and probability-uncertainty interpretation are not. |
| 4. Are statistical assumptions defensible? | **NO** | Team/game/player hierarchy, clustering, selection, and missingness are not adequately modeled. |
| 5. Is the model free from meaningful data leakage? | **NO** | Direct target-season roster/snap leakage and globally outcome-fitted artifacts. |
| 6. Is multicollinearity adequately controlled? | **PARTIALLY** | Yes in the active team design; no in the upstream 86-facet player layer. |
| 7. Do residuals appear appropriately random? | **NO** | Team identity R² 5.75%, permutation p=.002; late-season and favorite structure remain. |
| 8. Does the model demonstrate genuine out-of-sample predictive power? | **PARTIALLY** | It beats trivial/prior-Pythagorean baselines, but player lift is contaminated and artifact selection is not fold-clean. |
| 9. Does it outperform substantially simpler models? | **NO** | Scalar team strength is within 0.00048 Brier; pregame Elo is materially better. |
| 10. Does the player-evaluation layer materially improve team predictions? | **PARTIALLY** | Apparent +0.0072 Brier contribution, but the test is not temporally valid; dedicated QB removal improves performance. |
| 11. Does the model adequately capture matchup effects? | **NO** | No learned interactions; scalar difference performs almost identically. |
| 12. Are probabilities properly calibrated? | **PARTIALLY** | Reasonable broad calibration, but slope 0.860 and systematic bucket/favorite errors. |
| 13. Would I trust it to rank teams today? | **NO** | Useful as a descriptive input, not a validated standalone ranking; static and upstream-contaminated. |
| 14. Would I trust it to predict games today? | **NO** | It should not be deployed as the primary predictor until temporal reconstruction and full nesting are complete; Elo is currently stronger. |

## Bottom line

There is a legitimate model inside this repository: prior opponent-adjusted offense/defense, recruiting/roster priors, regularization, and a dynamic update can form a strong college-football predictor. The current repository has not yet demonstrated that its sophisticated player layer adds clean, generalizable team-prediction value. The next milestone should not be another feature or weight sweep. It should be a frozen, timestamp-correct historical dataset and a full expanding-window rebuild in which every upstream artifact is learned using the past only.
