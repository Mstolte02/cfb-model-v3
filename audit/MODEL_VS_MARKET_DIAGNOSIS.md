# Why the model does not beat the market

`BET_THRESHOLD_CALIBRATION.md` establishes that no gap threshold produces a
defensible edge. This document asks the prior question: how far behind is the model,
and where does the deficit come from?

All figures are the strict expanding-window predictions from `v4_backtest`, scored
against the archived posted line on the same 2,716 games with a two-sided moneyline.
The market probability is de-vigged from the pair.

## The headline

| | Brier | LogLoss | Accuracy |
|---|---:|---:|---:|
| model (dynamic) | .19468 | .57238 | .696 |
| **market (no-vig)** | **.18262** | **.54184** | **.720** |
| 50/50 blend | .18526 | .54936 | — |

The market is ahead by **.0121 Brier**. For scale, this project's adoption bar for a
new feature is .001, so the market is roughly twelve adoption-bars ahead of the
model. The gap is stable, not a bad season:

| season | model | market | gap |
|---|---:|---:|---:|
| 2022 | .20708 | .19577 | +.01130 |
| 2023 | .18620 | .17207 | +.01413 |
| 2024 | .19739 | .18548 | +.01192 |
| 2025 | .18838 | .17754 | +.01084 |

Blending the model with the market is **worse than the market alone**. That is the
first sign that the model carries little the market lacks.

## Does the model know anything the market does not?

Regress the outcome on both log-odds together:

```
market logit coefficient   +0.9716
model  logit coefficient   +0.0772
```

A market coefficient near 1 with a model coefficient near 0 is the textbook
signature of an efficient forecast encompassing a weaker one. The model contributes
almost nothing once the price is known. This single result explains every negative
finding in the threshold study: there is no gap size at which the model's disagreement
with the price is informative, because the disagreement is mostly the model's error.

## Where the deficit sits

**It is not a calibration problem.** Reliability buckets miss by .003 to .040 with no
systematic direction — mild overconfidence at the extremes, nothing structural.

**It is not concentrated in a game type.** The gap is near-uniform across market
confidence: .0136 on toss-ups, .0130 on leans, .0105 on clear games, .0116 on strong,
.0118 on locks. A uniform deficit is the signature of a general information shortfall
rather than a specific modelling fault.

**Thin markets are not the refuge.** The common assumption is that an independent
model finds its edge where books pay less attention. The opposite holds here:

| matchup | model | market | gap |
|---|---:|---:|---:|
| P4 v P4 | .19737 | .18879 | +.00858 |
| mixed | .14026 | .13091 | +.00935 |
| **G5 v G5** | .20570 | .18994 | **+.01576** |

The model is nearly twice as far behind in G5 games as in P4 games. Small-conference
games are where the model is *weakest*, not where the market is.

**It closes as the season goes on.** Model-minus-market Brier by week bucket:

| weeks | gap |
|---|---:|
| 1–3 | +.02016 |
| 4–6 | +.01901 |
| 7–9 | +.00526 |
| 10–12 | +.01318 |
| 13+ | **+.00017** |

By week 13 the model has essentially caught the market. The deficit is a **preseason
prior** problem: entering the year the market knows things the model does not, and
the model only draws level once enough on-field results have accumulated for the
dynamic update to overwrite the prior.

## Margin and total

| predictor | RMSE | corr with actual |
|---|---:|---:|
| model margin | 17.568 | .507 |
| market spread | 15.285 | .659 |
| model total | 17.119 | **.099** |
| market total | 15.587 | .379 |

### The totals model is the clearest defect in the system

`corr(model_total, actual_total) = .099` overall, and **.002 in 2025** — no measured
information at all in the most recent season. It is also under-dispersed (SD 5.21
against the market's 7.41 and the outcome's 16.80) and agrees with the market's own
number only at r = .325.

The cause is visible in `betting_backtest.point_totals`: the totals predictor is fit
from `V4.build_frame(...)` **without** `granular=True` and receives **no dynamic
in-season update**, while the win-probability model gets both. Totals were built once
and never given the machinery the rest of the model received.

This matters beyond diagnostics. The site publishes a model total, computes a model
gap from it, and — since the last UI change — offers a "bet to place" on that gap. All
three rested on a predictor with approximately zero skill.

**Repaired.** `TOTALS_MODEL_REPAIR.md` adds the missing scoring level and pace:
correlation rises to .156 pooled and from .005 to .094 in 2025, with an RMSE
improvement whose 95% interval excludes zero. That removes a meaningless published
number. It does not make totals bettable — .156 against the market's .379 — and the
board now says so beside the column.

## Does the model gain edge as it disagrees more?

The threshold sweep asks "which grid point has the best ROI", a maximum over noisy
estimates, which is why its selection null eats the answer. The fairer question is
whether performance *trends* with the size of the disagreement. A trend is much harder
to fake: noise produces a best bucket every time, but not a monotone ordering with a
direction declared in advance. `scripts/disagreement_trend.py` runs it as one
permutation test per market rather than fourteen.

**Spread.** Spearman rho +.0044, p = .42. No trend, and the buckets are not ordered —
the 5.9-8.4 point bucket is the *worst* in the set at .407 hit and −.21 ROI.

**Moneyline.** Spearman rho **−.0984, p = 1.00** against the declared direction. Hit
rate falls monotonically as the model disagrees more: .475, .430, .367, .353, .354,
.331. Bigger disagreement means the model is more likely to be wrong.

**Total.** rho +.0202, p = .14. Weak, positive, not significant, buckets not ordered.

### The confound, and the clean version

The moneyline result is partly mechanical: a larger moneyline gap means backing a
bigger underdog, and underdogs win less by construction. The confound-free test
compares the edge the model *claims* against the edge it *realises*, where realised
edge is the model side's actual win rate minus the market price of that side.

| claimed edge | market price of the side | realised edge |
|---:|---:|---:|
| +1.1% | .480 | −0.6% |
| +3.3% | .428 | +0.2% |
| +5.7% | .392 | −2.5% |
| +8.9% | .360 | −0.7% |
| +13.1% | .341 | +1.3% |
| **+22.0%** | .313 | **+1.8%** |

Regression of realised on claimed: **slope +0.150 ± 0.111, p = .18.** A model whose
edge were entirely real would have slope 1. This one keeps about 15% of what it
claims, and that 15% is not distinguishable from zero.

So the answer to "does confidence buy anything" is: a little, in the right direction,
too small to measure and far too small to bet. When the model says it is 22 points
better than the price, it is worth under 2.

### One place the disagreement does mean something

When model and market pick **different winners** outright — 373 of 2,716 moneyline
games — the model side wins only 41.8%, so the market is right 58% of the time. But
those bets are taken at plus money, and the ROI is **−0.75%** against −4.44% for games
where the two agree. It is the least-losing subset anywhere in this study, and still
losing, on 373 bets whose standard error is about five points.

On the spread the same disagreement carries nothing at all: 48.5% cover against 48.4%,
ROI −5.48% against −5.58%.

## What is actually missing

Ranked by likely contribution to the .0121 gap:

1. **Availability and injury information — nothing at all in the model.** In college
   football a starting quarterback is worth roughly a touchdown. The market prices
   this within minutes of the news; the model cannot see it. This is consistent with
   the deficit being largest early, when the model's roster picture is a preseason
   depth chart and the market's is current.
2. **Roster churn after the preseason snapshot** — portal moves, suspensions,
   academic issues, opt-outs. Same mechanism.
3. **A broken totals model** — see above. Cheapest of these to fix.
4. **Weather**, which matters most for totals and is absent.
5. **Situational spots** — lookahead, letdown, short weeks, travel, bowl motivation.
6. **The market itself as an input.** Excluded by design: the site states that prices
   are context, not model inputs.

## The strategic point

Point 6 is a design decision, not an oversight, and it deserves stating plainly
because it determines what is achievable.

An independent from-scratch model competing head-on with a liquid market is playing
the hardest available game. The market is an aggregate of many models plus current
information plus money that punishes error. To beat it you need either information it
lacks or a better use of shared information — and the encompassing regression says
this model presently has neither.

The standard architecture for a betting product inverts this: start from the market
price and model the *residual*, adding only what you have that the price does not.
The standard architecture for a **forecasting** product is what is built here.

Those are different products. The current model is a good forecasting product — a
.19 Brier from preseason inputs is respectable work. It is not close to a betting
product, and no threshold tuning will convert one into the other.

Finally, the arithmetic that sets the bar: at −110 you need 52.38% to break even. The
model needs to close a .0121 Brier gap *and then exceed the market by enough to clear
the hold*. Matching the market exactly still loses at −110.

## What would move the needle, in order

1. ~~**Fix the totals model.**~~ Done — see `TOTALS_MODEL_REPAIR.md`. The fix was
   scoring level and pace rather than the granular frame; level was the larger of the
   two omissions.
2. **Get an availability feed** and test QB-out as a rating shock. Largest expected
   gain of anything on this list.
3. **Measure CLV instead of ROI.** At four seasons, ROI cannot resolve a 2–3 point
   edge; closing-line value detects one far faster. The forward ledger in
   `market_tracking.json` already collects the right data.
4. **Decide which product this is.** If betting is the goal, model the market residual
   and abandon price-independence. If forecasting is the goal, stop scoring the model
   against closing lines and drop the bet flags.

## Reproduction

The figures here come from `artifacts/v4_backtest_predictions.csv` joined to the
archived lines via `scripts.betting_backtest.game_market_rows`.
