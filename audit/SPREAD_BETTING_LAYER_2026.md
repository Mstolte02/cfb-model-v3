# Spread betting layer: research status (28 September 2026)

The forecasting model stays. The old eight-point model-versus-line screen remains
historical tracking, not a validated betting edge. The actionable board now fails
closed for future games when its weekly lock or most recent successful line check is
older than six hours. Started-game history is not regraded. Future selections are
labelled **PAPER**, not bets to place.

## Direct cover test

`python -m scripts.spread_cover_research` pairs the strict 2023–25 v5.1/PFF-composite
outer-fold forecasts with CFBD's archived DraftKings spread for the same game. It
fits one regularized cover-probability coefficient on earlier seasons only: the
model's implied margin minus the book's implied margin. The 17.5-point conversion,
ridge 100, and 55% paper gate are fixed in code, not chosen on holdout ROI. Pushes
are excluded from binary scoring. Input hashes are saved with the result.

| Test season | Games | 50/50 Brier | Market + model Brier | Paper selections at 55% |
|---|---:|---:|---:|---:|
| 2024 | 706 | .250000 | .249933 | 36, +0.8% at assumed −110 |
| 2025 | 645 | .250000 | .249977 | 5, +14.5% at assumed −110 |

The fitted model-gap coefficient shrinks from .0183 to .0138 per point. The scoring
gain is minute, and five selections cannot resolve return. This is a **negative
adoption decision**, not a profitable spread signal. The 2025 season has also been
inspected in earlier research, so it is diagnostic rather than pristine confirmation.

The archived line has no quote time or side-specific price. An assumed −110 return
cannot establish an executable edge. The live 2026 board before Week 5 used older
model versions, so there is no legitimate v5.1 forward ATS sample yet.

## What has to happen next

1. Restore a successful, timestamped line feed. The GitHub capture ran on September
   28 but CFBD returned HTTP 429; the last successful `/lines` check in this branch
   is September 22. Do not surface the old board as a current price.
2. Capture side-specific spread odds and additional books, with retrieval times and
   offered-line identity. The existing CFBD `/lines` payload lacks spread-side
   prices and sportsbook-native timestamps. `scripts.spread_shopping_audit` found 40
   of 242 games with at least a one-point DraftKings/Bovada difference at their last
   successful paired check within six hours of kickoff (five differed by two points).
   Availability and price of execution remain unverified.
3. Freeze a *new* spread hypothesis before the next priced week. A sensible next
   candidate is a verified starting-QB availability shock against a contemporaneous
   consensus line, evaluated for incremental cover probability and closing-line
   value. The repo's proxy based on a player missing the previous game did not help.
4. Grade the frozen candidate and actual offered price prospectively. Require
   positive CLV and calibrated cover probabilities before treating net ROI as an
   edge. Do not select a new gap threshold from the 2023–25 curve.

No live wager is recommended from this audit.
