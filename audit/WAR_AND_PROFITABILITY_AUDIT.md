# WAR role split and profitability audit

## Decision

Player rankings now use **intrinsic WAR**: the preseason player forecast before the
current depth-chart overlay. Team totals, Heisman opportunity, and what-if scenarios
use **expected contribution** after a soft role allocation. The role layer publishes
an expected snap share, a low/high range, and confidence.

The previous allocator forced every position room to a league-average healthy
starter/backup split. A disputed binary depth-chart label could therefore move a
player from roughly 50th to top ten at his position without changing the room total.
The replacement gives the chart only 25–55% effective weight, based on agreement
between the two current chart sources and competition within the room. Prior usage and
intrinsic value retain the rest of the allocation.

Notre Dame linebacker is the concrete example:

| Player | Intrinsic WAR | Intrinsic LB rank | Expected contribution | Expected snap range |
|---|---:|---:|---:|---:|
| Kyngstonn Viliamu-Asa | 0.340 | 1 | 0.232 | 21–53% |
| Drayk Bowen | 0.204 | 14 | 0.294 | 62–94% |
| Jaylen Sneed | 0.185 | 22 | 0.134 | 11–43% |
| Jaiden Ausberry | 0.151 | 49 | 0.219 | 42–74% |

This preserves the model's judgment that Viliamu-Asa is the best player in the room
without asserting that a public July chart is worthless. Team-room value is preserved
through the role allocation; the separate quarterback composite can still move value
between quarterback rooms by design.

## Player-level validation

The 2025 holdout contains 12,073 roster-player forecasts trained only on seasons
through 2024. This is an all-roster preseason test; players who did not record a 2025
snap remain in the population rather than being selected away after the fact.

| Forecast | Pearson r | Spearman r | MAE | Actual top-decile precision |
|---|---:|---:|---:|---:|
| Intrinsic | **0.592** | **0.410** | 0.0411 | **45.2%** |
| Soft role overlay | 0.582 | 0.404 | **0.0410** | 44.8% |
| Hard role overlay | 0.542 | 0.353 | 0.0426 | 40.7% |

Random top-decile precision is 10%, so intrinsic WAR's 45.2% is useful ranking signal.
The soft role proxy reduces MAE by only 0.0001 while weakening ordering. Hard role
allocation is clearly worse. That supports showing intrinsic player rankings and
reserving role adjustment for expected team contribution.

The exact 2026 role ranges cannot be validated historically because archived preseason
depth charts are not available for every prior season. The historical role proxy uses
prior-season room rank and snap share, both known before the holdout season.

## Profit experiments

### Market-anchored residual model

A regularized expanding-window model was fit to predict only the error left after the
sportsbook probability, spread, or total. This is a stronger test than asking whether
the football model is accurate in isolation.

It did not produce a validated edge:

- Moneyline Brier worsened from 0.17848 (market) to 0.17868.
- Spread MAE worsened from 12.09 to 12.11 points; the selected 2025 rule returned -8.0%.
- Total MAE improved slightly from 12.488 to 12.475, but the selected 2025 rule returned -0.8%.

The website therefore does not label any residual output as a bet.

### Consensus disagreement plus best-price execution

One execution rule is promising. The signal uses the median no-vig home probability
across books, selects games where the model differs by at least 15 percentage points,
and then takes the best archived moneyline for the selected side.

| Period | Bets | ROI |
|---|---:|---:|
| 2022–24 | 392 | +4.8% |
| 2025 | 119 | +10.7% |
| 2022–25 | 511 | +6.2% |

This is mostly an underdog strategy (87.7% of bets; average price +290). It lost in
2023, and a season-week cluster bootstrap gives a wide 95% ROI interval of -6.6% to
+19.5%. The 15% cutoff was examined during the expanded audit, so 2025 is corroborating
evidence, not a pristine holdout. The site calls it a **research watchlist**, defaults
the weekly board to consensus plus best price, and requires 2026 forward validation.

## Highest-value next additions

1. **Timestamped multi-book odds and closing-line value.** Store each quote with its
   retrieval time, then judge the research signal first by whether it beats the close.
   Historical CFBD posted lines do not provide enough timestamp detail for this test.
2. **Availability as a versioned event stream.** Starting-quarterback changes, major
   injuries, suspensions, and depth-chart promotions should create before/after model
   snapshots. These are plausible sources of information the market may incorporate
   with delay.
3. **Predictive distributions, not point gaps.** Estimate matchup-conditional error and
   require the price edge to survive model uncertainty and vig. This should reduce bet
   count rather than manufacture more picks.
4. **Forward-only segmentation.** The candidate is underdog-heavy. Pre-register the
   15% rule for 2026; do not keep slicing conference, price, week, or team subsets until
   an attractive backtest appears.
5. **Track execution quality.** Record requested price, available price, filled price,
   limits, and close. A model cannot overcome a strategy whose historical profit exists
   only at prices that were not actually obtainable.

The practical path over the hump is therefore better market data and disciplined
execution around the model's largest disagreements, not another broad batch of football
features added directly to win probability.
