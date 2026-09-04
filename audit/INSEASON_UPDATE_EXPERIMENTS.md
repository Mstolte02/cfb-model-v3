# In-season update experiments

## Question

The former shipping update treated the final score as a binary result multiplied by log
margin of victory. This audit asks whether future-game predictions improve when the
update instead uses:

1. a robust residual between observed and expected margin;
2. an adjusted margin inferred from the postgame efficiency line;
3. CFBD's published postgame win expectancy as an adjusted result;
4. residual variance that changes smoothly with pregame certainty;
5. a smooth blend from sign evidence near 50/50 to margin evidence in lopsided games;
6. a continuous pregame-certainty x postgame-surprise overlay.

The replay is expanding and forward-only. For every held-out season, the team model,
postgame-stat model, sigma curve, update learning rate, and blend are chosen without
using that season. The primary comparison is 2023-25 (2,189 games), because each fold
has at least one earlier forward-validation season.

## Baseline and robust margin

The former rule was

```text
delta = K * log(abs(margin) + 1)
          * 2.2 / (2.2 + 0.35 * abs(pregame_logit_gap))
          * (result - expected_win_probability)
```

The robust-margin candidate is

```text
expected_margin = sigma * normal_ppf(expected_win_probability)
margin_score = clip((observed_margin - expected_margin) / sigma, -2.5, 2.5)
delta = K * margin_score
```

It converts the question from "did the favorite win, and by how much?" to "how many
residual-margin standard deviations did this game miss its expectation by?"

## Adjusted margins

The local adjusted-margin model is a ridge regression fitted only on earlier seasons.
It uses home-minus-away differences in total PPA, plays, drives, success rate,
explosiveness, power success, stuff rate, line yards, standard-down PPA,
passing-down PPA, rushing PPA, and passing PPA. The update tunes a blend between that
estimate and the actual score margin.

The public candidate comes directly from `homePostgameWinProbability` in CFBD's
`/games` response. It is converted to margin units with
`sigma * normal_ppf(postgame_win_probability)` and likewise blended with actual
margin. Historical coverage is effectively complete. The 2026 UMass-Rutgers game is
reported as 0% Rutgers / 100% UMass postgame expectancy.

## Results

Lower is better.

| Strict 2023-25 replay | Brier | Log loss | Incremental interpretation |
|---|---:|---:|---|
| Shipping updater | .18149 | .53727 | baseline |
| **Robust raw-margin residual** | **.17899** | **.53147** | −.00250 vs shipping |
| Rich local adjusted margin | .17898 | .53145 | effectively tied with raw margin |
| **CFBD-adjusted robust margin** | **.17864** | **.53073** | −.00034 vs raw margin |
| Heteroskedastic robust margin | .17899 | .53151 | no incremental value |
| Sign/margin interaction | .18040 | .53455 | worse than full robust margin |
| Continuous surprise overlay | .17899 | .53147 | selected interaction = 0 |
| CFBD-adjusted + heteroskedastic | .17864 | .53076 | no incremental value |
| CFBD-adjusted + surprise overlay | .17864 | .53073 | selected interaction = 0 |

Paired season-week bootstrap:

- Robust raw margin minus shipping: **−.00250**, 95% interval
  **[−.00484, −.00020]**, 98.3% probability better.
- CFBD-adjusted margin minus shipping: **−.00285**, 95% interval
  **[−.00541, −.00039]**, 98.8% probability better.
- CFBD-adjusted margin minus robust raw margin: **−.00034**, 95% interval
  **[−.00109, +.00042]**, 81.2% probability better.

The public adjustment receives 0% weight in the 2023 held-out fold and 25% weight in
the 2024 and 2025 folds. Actual score therefore remains the dominant observation.
The incremental public-data gain does not clear the project's predeclared 0.001 Brier
complexity bar and is not statistically resolved from raw robust margin.

## Does sigma vary with pregame certainty?

The fitted curve is `sigma(P) = sigma0 * (1 + beta * abs(P - .5))`. Forward fits are:

| Held-out season | sigma0 | beta |
|---|---:|---:|
| 2022 | 17.54 | −.012 |
| 2023 | 16.49 | +.148 |
| 2024 | 16.70 | +.010 |
| 2025 | 17.12 | −.046 |

Beta is small and changes sign. Residual-margin standard deviation is also not
monotone across certainty bands:

| Pregame probability band | Games | Residual-margin SD |
|---|---:|---:|
| 45-55 | 290 | 16.33 |
| 35-45 / 55-65 | 551 | 16.68 |
| 25-35 / 65-75 | 489 | 16.00 |
| 15-25 / 75-85 | 452 | 15.28 |
| 0-15 / 85-100 | 407 | 16.11 |

The 45-55 group averages a 50.09% home prediction and a 48.62% home win rate. There
is no evidence of a discontinuity or a useful smooth variance curve around toss-ups.

## Moving K follow-up

The follow-up kept the robust-margin score intact and allowed its learning rate to
move continuously as game evidence accumulated:

```text
K(g) = K_open * end_ratio ** (average_prior_games / 11)
```

An `end_ratio` below one means early games move ratings more than late games. The
constant rule (`end_ratio = 1`) was included inside the same forward-only search.

| Strict 2023-25 replay | Brier | Log loss | Difference vs constant K |
|---|---:|---:|---:|
| Raw robust margin, constant K | .178986 | .531468 | baseline |
| Raw robust margin, moving K | .178504 | .530170 | −.000482 |
| CFBD-adjusted margin, constant K | .178644 | .530733 | baseline |
| CFBD-adjusted margin, moving K | .178377 | .530113 | −.000267 |

Every tunable raw-margin fold selected decay: the opening K was `.20`, `.25`, and
`.25`, with end-of-season ratios `.50`, `.50`, and `.75` for the 2023-25 holdouts.
That is directionally coherent with a Bayesian reading—trust each new result less as
the season sample grows—but the raw-margin improvement's 95% paired bootstrap
interval is **[−.001096, +.000102]**. It is promising, but it neither excludes zero
nor clears the project's `.001` complexity threshold. It remains a research
candidate rather than part of the live rule.

## Verdict

The robust raw-margin residual is adopted in production with `K=.20`, the trained
100% dynamic blend, and a 2.5-sigma score cap. It clears the project's materiality
threshold and uncertainty interval while remaining simple. Both the local weekly
state updater and the standard-library scheduled-site replay implement the same
formula.

Keep CFBD postgame expectancy in the research/export path. It is the best point
estimate and is an excellent game-level diagnostic, but its incremental gain over raw
robust margin is too small and uncertain to make the live updater externally
dependent yet. More seasons or a predeclared live shadow test could settle it.

Do not adopt the heteroskedastic curve, toss-up sign/margin blend, continuous
surprise overlay, or moving K yet. Their fitted extra terms either collapse toward
no change, reduce accuracy, or remain below the adoption bar.

Reproduce with:

```powershell
python -m scripts.inseason_evidence_backtest
python -m scripts.inseason_moving_k_backtest
```

Machine-readable results are written to
`artifacts/inseason_evidence_backtest.json` and per-game predictions to
`artifacts/inseason_evidence_backtest_predictions.csv`. The moving-K counterparts
use the same names with `inseason_moving_k_backtest`.
