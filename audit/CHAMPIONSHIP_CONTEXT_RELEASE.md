# Championship context update — 20 November 2026

The committee forecast now uses signed championship margin (capped at 35,
scaled by seven) and separate winner/loser opponent terms derived from the
preceding CFP rank, with unranked opponents at 30. Ridge alpha is 30; the
existing 0.25 within-pod head-to-head adjustment is retained. Future game
margins already come from the football simulator's normal margin draw.

The candidate was selected on expanding 2018–23 development folds. On rolling
2024–25 assessment, ranked-team MAE was 1.176 versus 1.316 for the prior alpha-300
specification; ranked championship-participant MAE was 0.810 versus 1.190.
Retuning regularization alone achieved overall MAE 1.172, so the added context
mainly improves championship behavior. This is retrospective evidence with
limited championship samples, not a prospective accuracy guarantee.

Environment adjustments, PPA and special cutoff protection are excluded.
The browser Committee Watch port and R season simulator use the same features.
`python scripts/train_committee_timeline.py` reproduces the released coefficients.

The old historical JSON matched its Git blob checksum but contained invalid
bytes starting at offset 393216. It was reconstructed using ESPN's season-specific
core CFP polls and CFBD regular-season final scores. No 2026 rows enter fitting.
Historical opponent conference flags identify the contemporary power leagues.

Validation: R feature-path tests for capped margins and preceding opponent rank;
browser regression for close versus large title wins; committee, conference,
selection, preseason-lock tests; the complete Vercel browser build checks.
