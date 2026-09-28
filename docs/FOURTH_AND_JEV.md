# Fourth & Jev

Fourth & Jev is an experimental probabilistic decision layer around the existing CFB forecasting system. The current production model remains the source of engineered football features; Jev receives a richer pregame state and returns typed probabilities for wins, margins, tails, variance, and matchup mechanisms.

## Design rules

1. **No leakage.** Every state is reconstructed as of a pregame timestamp.
2. **Football first, market second.** Fundamental Jev forecasts cannot see sportsbook prices. A separate market-stage call may compare frozen football answers with contemporaneous prices.
3. **The old model is an input, not an oracle.** v5/v5.1 probability, margin, ratings, form, PFF, WAR, talent, and returning production become part of state.
4. **Persist every request and answer.** Store game ID, as-of, state schema, question version, Jev model, state hash, answers, latency/token metadata, and outcome.
5. **Calibration before ROI.** Primary metrics are Brier, log loss, reliability, and calibration-in-the-large. Betting evaluation adds CLV and hold-aware expected value.
6. **Tail claims are graded directly.** 7+, 14+, 21+, close-game and other propositions are evaluated from actual final scores rather than inferred from the winner model.

## Architecture

    repository data -> canonical pregame state -> Jev football call
                                            |-> win / tail / variance probabilities

    frozen football answers + market snapshot -> Jev market call
                                            |-> mispricing / tail-opportunity scores

This lets us test whether Jev adds football signal rather than simply echoing market expectations.

## Initial state blocks

Existing repository inputs:
- v5/v5.1 ensemble probability and predicted margin
- preseason member strength and in-season rating movement
- opponent-adjusted offense/defense form
- PFF season-to-date composites
- projected player WAR and relevant position/facet outputs
- recruiting talent and returning production
- historical game results
- archived market snapshots and settlements (second-stage only)

Priority additions suggested by the existing market diagnosis:
- timestamped QB/starter availability
- roster churn, suspensions, opt-outs and portal changes
- kickoff weather
- rest, travel, time-zone and short-week context
- coaching/coordinator continuity
- current-season room/player state only where forward validation supports it

## Historical experiment

Initial v5/v5.1 evaluation window: 2023-2025, matching the outer-fold replay.
The 2022 fold lacks the earlier seasons required to fit the ensemble stack.

For every game, reconstruct the state that existed before kickoff and persist:
- Jev model/version
- state schema and question-set version
- input state hash
- P(home win)
- P(home margin >= 7/14/21)
- P(away margin >= 7)
- P(abs(margin) <= 3)
- tail-shape distribution
- variance distribution
- upset-path distribution
- existing-model probability/margin
- market snapshot in a separate second-stage record
- final result and grading flags

Compare against v5/v5.1, CFBD Elo, and the no-vig market where timestamp-compatible prices exist.

## Credentials

Never commit a key.

    TYPESAFE_API_KEY=...
    JEV_MODEL=jev-latest

Use `python -m scripts.fourth_jev_probe` for a dry payload preview. Add `--live` only after a valid TypeSafe API key is available.

## Build order

1. Canonical state + Jev API scaffold. **Started.**
2. Historical ledger with state hashing and resume-safe writes.
3. Repository adapters for strict pregame state reconstruction.
4. Backfill 2022-2025 existing-model inputs.
5. Add missing availability/weather/situational sources.
6. Run football-only Jev batch and calibration analysis.
7. Freeze football forecasts, then test market residual and tail opportunities.

## Current-slate pilot (Week 5, 2026)

The `fourth-and-jev-live-integration` work freezes the published v5.3
through-Week-4 state into 56 market-blind Week 5 records. Run
`python -m scripts.fourth_jev_export_current --out <new snapshot.jsonl>`
before any game date; it refuses to overwrite a snapshot. Each state includes
the model snapshot SHA-256, a real UTC issue time, date-only kickoff metadata,
four member ratings and form, PFF team composites, in-season WAR, and preseason
position projections. A date-only schedule never masquerades as a precise
kickoff time.

The initial direct-win Jev pilot produced 56 frozen answers in a local,
append-only ledger. Mean absolute disagreement with v5.3 was 5.75 percentage
points; the largest was 20.07 points. These are **ungraded future games**, so
disagreement is not evidence of improvement or a betting edge. The web desk
exports only this direct-win pilot, without raw input states or numeric tail
claims. The local ledger is ignored by Git and must be retained separately
for later scoring.

The first tail trial exposed structural problems: 11 of 56 independent Noul
sets violated basic event nesting, such as P(home by 7+) > P(home wins).
A second, 10-bin Choice distribution was coherent by construction but, in
three pilot games, implied 86-98% home wins while the direct win Noul gave
58-75%. Keep both raw experiments in the ledger; do not publish either as a
calibrated margin distribution. The version 0.2 question set includes the
bucket trial for research only.

Historical grading remains blocked in this checkout because raw CFBD and
licensed PFF/expanded-WAR inputs are external to Git. Restore immutable
pregame source vintages, regenerate the 2023-2025 states, and score paired
identical games and issue times against v5.3 and no-vig market snapshots.
Only after that should a market-opportunity or tail-advantage label ship.
