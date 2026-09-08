# Should the projected margin follow the in-season ratings?

**Reproduce:** `venv/Scripts/python -m scripts.dynamic_margin_backtest`
→ `artifacts/dynamic_margin_backtest.json`, `artifacts/dynamic_margin_backtest_predictions.csv`

## The defect

The site prints a win probability and a spread next to each other and they can name
different winners. Oklahoma at Michigan on the 2026 board reads Oklahoma 58%, projected
score 23-22 Oklahoma, and **Michigan -0.9**.

This is structural, not a rendering slip. `predict()` in `viz/app.js` and
`WeeklyRatingState.predict` in `src/dynamic.py` build the two numbers from two models:

- the **probability** is `(1 - blend)·static + blend·dynamic`, and the shipped 2026
  state runs `dynamic_blend = 1.0`, so it is the in-season rating alone;
- the **margin** is `pred_margin`, the preseason ridge, which never sees a result from
  the current season.

Everything downstream of the margin inherits the split: the spread, the projected
scoreline, and the score halves of the total. `displayScore()` then force-flips the
losing side to a 1-point winner so the scoreline matches the probability, which is where
23-22 comes from. Its comment anticipates this as a coin-flip rounding case; 58% against
-0.9 is well past that. Power ratings already blend dynamically — `pred_margin` is the
one published output the in-season update never reaches.

On the 2026 board, **58 of 710 unplayed games (8.2%)** have a probability and a spread
that disagree about the winner. Over the 2022-25 replay the same disagreement runs
**15.3%** of games, and it is getting worse, not better: 11.6% in 2022 and 18.9% in 2025.

## The proposal

Send the margin through the same probit link the ensemble already uses in reverse
(`p_margin = Φ(pred_margin / σ)`), and that `update_delta` already uses to turn a
probability into an expected margin:

    margin = σ · Φ⁻¹(p)

## Contract

The replay is the same strict expanding window as `scripts.v4_backtest`. Feature set,
knobs, learning rate and blend are **read back** from `artifacts/v4_backtest.json` rather
than re-tuned, so this cannot quietly select a different model than the one that produced
the shipped numbers. Free parameters introduced here are chosen on 2022-24 and reported
once on the untouched 2025 holdout. Per-fold blend was .75 for 2022-24 and 1.0 for 2025;
σ ran 16.7-17.5.

## Result 1 — the link contributes nothing; the in-season information is the whole gain

`m_probit_static` runs the **static** probability through the same link. If the probit
transform were doing the work, this arm would show it.

| arm | pooled MAE | 2025 MAE |
|---|---|---|
| `m_static` (shipped) | 13.65 | 14.17 |
| `m_probit_static` (link only, no in-season info) | **13.69** | **14.27** |

The link on its own is a wash to slightly worse. Every gain below is the in-season
evidence, not the transform.

## Result 2 — the dynamic margin is decisively more accurate

| arm | MAE | RMSE | winner | disagrees with p |
|---|---|---|---|---|
| **pooled 2022-25** ||||
| `m_static` | 13.65 | 17.18 | 67.6% | 15.3% |
| `m_probit_blended` | **12.63** | **15.98** | **71.6%** | 0.0% |
| **2025 holdout, untouched** ||||
| `m_static` | 14.17 | 17.74 | 65.2% | 18.9% |
| `m_probit_blended` | **12.64** | **15.98** | **73.6%** | 0.0% |

Better in every season: MAE 13.36→12.73 (2022), 12.99→12.37 (2023), 14.07→12.78 (2024),
14.17→12.64 (2025). The holdout gain is the largest of the four, which is the opposite of
what an overfit would do.

Two arms were added to check whether the repair needs tuning, and neither is needed:

- the **mix grid** `(1-w)·static + w·dynamic` selected **w = 1.0** on 2022-24 — the data
  asked for full replacement, not a partial blend;
- the **fitted scale** on Φ⁻¹(p) came out at **17.88 points** per unit of normal score,
  against a model σ of 16.7-17.5. The assumed link is already the measured one, so σ
  needs no free parameter beside it.

## Result 3 — it does NOT improve betting against the spread

This is the honest cost, and it points the other way.

| arm | gate 6 | gate 8 (shipped) | gate 10 |
|---|---|---|---|
| **pooled 2022-25** ||||
| `m_static` | 1,303 bets, 49.5% | 923 bets, **52.0%** | 617 bets, **54.4%** |
| `m_probit_blended` | 579 bets, 50.8% | 303 bets, 51.5% | 148 bets, 52.8% |
| **2025 holdout** ||||
| `m_static` | 335 bets, 45.9% | 238 bets, 47.6% | 165 bets, 49.4% |
| `m_probit_blended` | 117 bets, 54.3% | 51 bets, 43.1% | 28 bets, 50.0% |

A more accurate margin agrees with the market more often, so roughly two thirds of the
flagged spread bets stop clearing the gate, and the disagreements that survive are not
better ones. Neither arm clears the 52.4% break-even on the holdout. This does not
contradict `BET_THRESHOLD_CALIBRATION.md`; it restates its finding on a new margin — the
spread flag was a coin before this change and is a coin after it, on a smaller sample.

## Reading

The margin should follow the probability. Describing the game is the model's job and the
dynamic margin does it better by a point and a half of MAE and eight points of winner
accuracy on untouched data, while removing a contradiction that the site currently
resolves by silently overriding the scoreline. The betting consequence is a smaller
spread board with no measured edge either way, which is what the spread market already
was.

The change is a one-line substitution in two places — `WeeklyRatingState.predict` in
`src/dynamic.py` and `predict()` in `viz/app.js` — plus the score split that reads it.
Nothing about the probability, the ratings or the update rule moves.

## What would change the answer

The dynamic margin's advantage is in-season information, so it is worth nothing in week
0 and most in the back half of a season; a week-by-week breakdown would say how the gain
accrues and whether an early-season blend below 1.0 is better than the flat replacement
the mix grid selected. The ATS comparison is also made against CFBD's archived posted
line, not a closing line, so it measures disagreement with a stale number.
