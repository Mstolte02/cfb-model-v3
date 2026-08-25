# What model gap is large enough to bet?

## Status

**No market supports a bet-to-place threshold.** The `BET_RULES` values currently in
`viz/app.js` are placeholders with no empirical backing, and this document is the
evidence that none of them can be justified from four seasons of history.

Retested 2026-08-25 on predictions from the shipped `core_war_talent_sources`
model (positional recruiting + rated portal added; see
TALENT_SOURCES_EXPERIMENTS.md). The better model did not change the answer: every
market's recommendation is unchanged, and the futures gap-0.5 signal — still the
strongest thing in the study — remains below its selection null.

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

## Results, 2022–25, strict expanding-window predictions (retested on the shipped talent-sources model)

### Spread — 2,872 bets at gap 0

Negative at every threshold through gap 5, with the 90% interval below zero
through gap 5. Raising the bar mostly makes it worse: −4.2% at gap 0, −6.3% at gap
3.0, and only approaching break-even at gaps 6–7 (−2.2%) before turning +1.1% at
gap 8 where 942 bets remain — an interval of [−.039, +.060] and P(ROI>0) = .64,
which is a coin. Selection null q95 +.0095; observed best +.0111 sits inside the
noise band. No threshold qualifies.

### Moneyline — 2,716 bets at gap 0

Negative through gap .10, then positive: +1.1% at .12 (731 bets), +7.7% at .15
(479), +16.1% at .20 (223). The .20 interval [−.003, +.332] touches zero at its
lower edge.

The selection null settles it as before, and more emphatically now: a no-skill
model searching this grid reaches **+14.93% at q95** (mean best-of-grid +3.36%).
The observed best of +16.1% sits barely above it on 223 bets. The apparent
high-gap moneyline edge is still indistinguishable from searching a grid on small
samples. No threshold qualifies.

### Total — 2,771 bets at gap 0

The flattest surface. Best is +1.31% at gap 2.0 with 2,207 bets, interval
[−.025, +.052]. Selection null q95 is +1.36% — fractionally above the observed
best. Nothing here, but nothing badly wrong either; totals are the one market
where the model is not actively losing.

### Season win totals — 342 bets at gap 0

The strongest signal in the study, unchanged by the new model:
**gap 0.5, 209 bets, ROI +14.4%, 90% interval [+.027, +.348], P(ROI>0) = .98** —
the only threshold whose interval clears zero.

It still fails the formal rule, because the selection null q95 for this market is
+17.15% and the observed +14.4% sits below it. With 342 bets across four seasons
and nine grid points, best-of-grid noise reaches +17% routinely.

The dev/holdout protocol in `betting_backtest.py` tells the same story from its
own angle: threshold selected on 2022–24 (+19.1% at gap 0) returned **+11.6% on
the untouched 2025 holdout (127 bets)**, which flags `validated_edge: true` under
that older, weaker standard. Both readings are honest: the holdout corroborates,
and the selection null says four seasons cannot yet separate this from search.
The forward ledger in `market_tracking.json` is what resolves it.

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

**Adopted as tracking thresholds, 2026-08-25.** No market earns a *validated* bet
flag, and the site does not claim one. What shipped in `viz/app.js` is each
market's least-unsupported reading from this study, so every flagged row becomes a
forward observation against a stated rule instead of an untracked placeholder:

| market | tracking threshold | reading |
|---|---|---|
| spread | gap 8.0 | the only positive region (+1.1%, P=.64) — a coin, tracked deliberately |
| moneyline | gap .20 | observed best (+16.1%); sits barely above its null on 223 bets |
| total | gap 2.0 | flattest positive point (+1.31%), indistinguishable from zero |
| win totals | 0.5-win model gap | strongest signal found; interval clears zero, still below its selection null |
| playoff / conference | unchanged (50%-model screen) | untestable, no price history |

The win-total flag also keeps its price screen: the 0.5-win gap is the calibration
study's own quantity (model expected wins vs posted line), applied alongside the
existing de-vigged-price gate rather than instead of it.

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
