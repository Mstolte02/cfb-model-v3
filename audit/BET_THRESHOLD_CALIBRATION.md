# What model gap is large enough to bet?

## Status

**No market supports a bet-to-place threshold.** The `BET_RULES` values currently in
`viz/app.js` are placeholders with no empirical backing, and this document is the
evidence that none of them can be justified from four seasons of history.

## Why this differs from `betting_backtest.py`

That script already sweeps a threshold grid and selects the best development ROI. It
answers "which grid point looked best", which is a different question, for two reasons:

**No uncertainty.** At −110 a bet returns +0.909 or −1, so one bet's profit has SD
near 0.95 and ROI over n bets has standard error near 0.95/√n. At 200 bets that is
±6.7 points. A threshold showing "+3% ROI" is not distinguishable from a coin.

**No selection penalty.** Taking the best of a fourteen-point grid is fourteen chances
to find noise. This study re-runs the entire selection on bets whose *side* has been
randomised — schedule, prices, gap magnitudes and block structure all preserved — and
reports the ROI a model with no side-picking skill reaches by searching the same grid.

Intervals are 90% blocked bootstraps with season-week blocks, matching the paired
bootstrap the Brier comparisons use: two bets on the same slate share weather,
injuries and a common market state.

## Results, 2022–25, strict expanding-window predictions

### Spread — 2,872 bets at gap 0

Negative at every threshold, with the 90% interval below zero at every threshold
through gap 6. Raising the bar makes it **worse**, not better: −5.6% at gap 0 falls to
−7.9% at gap 3.5, and only approaches break-even at gap 8 where 1,023 bets remain
(−1.2%, interval [−.061, +.035]).

A model gap on the spread is not merely uninformative, it is **inversely** related to
profit over most of the range. Selection null q95 +.005; observed best −.012.

### Moneyline — 2,716 bets at gap 0

Negative through gap .10, turning positive at .12 (+1.8%), .15 (+7.3%) and .20
(+8.0%). Every one of those intervals includes zero, and the samples are thin (777,
502, 250 bets).

The selection null settles it: a no-skill model searching this grid reaches +12.95% at
q95. The observed best is +7.96%. **The apparent high-gap moneyline edge is entirely
explained by searching a grid on small samples.**

### Total — 2,771 bets at gap 0

The flattest surface. Best is +1.31% at gap 2.0 with 2,207 bets, interval
[−.025, +.052]. Selection null q95 is +1.36% — fractionally above the observed best.
Nothing here, but nothing badly wrong either; totals are the one market where the
model is not actively losing.

### Season win totals — 342 bets at gap 0

The only threshold in the entire study whose interval clears zero: **gap 0.5, 209
bets, ROI +14.4%, 90% interval [+.027, +.348], P(ROI>0) = .98.**

It still fails, because the selection null q95 for this market is +17.15% and the
observed +14.4% sits below it. With 342 bets across four seasons and nine grid points,
best-of-grid noise reaches +17% routinely.

Worth noting what the null demonstrates: the average best-of-grid ROI under no skill
is **+2.1%**, against a per-threshold expectation near −2.3% given a 4.6% hold. Search
alone is worth about 4.4 points of fake ROI on this sample, and up to 17 at the tail.
That is the exact bias the existing selection procedure carries.

### Playoff and conference-title futures

**Not calibratable.** No historical price archive exists for these boards; the only
futures market with stored history is season win totals. The 50%-model / positive-gap
rule now flagging rows on the site has never been tested against a price, and cannot
be with the data on hand.

## What to do with `BET_RULES`

The honest setting is that no market earns a flag. If the flags stay for research
purposes, they should be labelled as model screens rather than bets, because that is
what the evidence supports.

If a threshold must be chosen anyway, the least-unsupported readings are:

| market | reading |
|---|---|
| spread | none — negative at every gap; do not flag |
| moneyline | none below .12; above that the sample cannot tell skill from search |
| total | gap 2.0 is the flattest positive region, but indistinguishable from zero |
| win totals | gap 0.5 is the strongest signal found and still fails the selection null |
| playoff / conference | untestable, no price history |

## What would change the answer

More seasons is the only real fix; four is not enough to resolve a 2–3 point edge
against a 4.6% hold. Closing lines rather than archived posted lines would also help:
the current prices are CFBD's stored quote, not asserted closing numbers, and CLV is
the measurement that detects an edge fastest at this sample size. The forward ledger
in `market_tracking.json` is set up to collect exactly that.

## Reproduction

```powershell
python -m scripts.threshold_calibration
```

Artifact: `artifacts/threshold_calibration.json`, with the full ROI surface, per
threshold intervals and selection nulls for every market.
