# Is home field team-specific?

## Status

**Rejected.** One league-wide home-field coefficient beats every team-specific
version tested, at every level of shrinkage, on both metrics.

## The design

The model fits one home-field coefficient for all 136 teams. Altitude at Laramie and
Provo, a hostile night crowd and a long eastward trip into a noon kick are plainly not
the same advantage, so the prior was that this is worth splitting.

A free intercept per team is the wrong way to split it — a team plays six or seven
home games a season, so the estimate is mostly noise and the noisiest teams would get
the largest adjustments. This uses the treatment the coach phase used: **partial
pooling** toward the league mean, weighted by sample size,

```
hfa_team = w * (team home-minus-away margin - league mean),   w = n / (n + k)
```

Differencing a team's own home and away margins removes team strength — a good team
wins everywhere, and what is left is how much *more* it wins at home. Estimates for
season N use completed games through N−1. As k grows the estimates collapse to the
league mean, so **the shipping flat model is nested inside this one** and the
comparison cannot be rigged by construction.

## Result, pooled 2022–25

| candidate | static Δ vs flat | 95% CI | online Δ | 95% CI |
|---|---:|---|---:|---|
| unpooled per team | +.00400 | [+.00171, +.00636] | +.00329 | [+.00124, +.00529] |
| pooled k=6 | +.00290 | [+.00082, +.00507] | +.00236 | [+.00053, +.00418] |
| pooled k=12 | +.00213 | [+.00021, +.00411] | +.00172 | [+.00004, +.00339] |
| pooled k=25 | +.00111 | [−.00044, +.00271] | +.00089 | [−.00045, +.00223] |
| pooled k=40 | +.00060 | [−.00062, +.00187] | +.00047 | [−.00059, +.00153] |

Positive is worse. The ordering is monotone and behaves exactly as the theory says it
should: the less shrinkage, the worse it gets, and heavy shrinkage converges back to
the flat model it is nested in. The unpooled version is the worst thing tested, with
an interval well clear of zero.

The team-level estimates themselves are not absurd — at k=12 the spread is 3.2 points
of margin, Tennessee strongest at +7.5 and Vanderbilt weakest at −8.8. They are simply
not stable enough season to season to pay for themselves.

## A measurement bug worth recording

The first run produced a spread of ±12 points with **Stephen F. Austin, Grambling,
Morehead State, Susquehanna, CSU Pueblo and South Dakota Mines** among the extremes.
Those are FCS and Division II programmes that appear in the schedule only because an
FBS team hosted them. They appear almost entirely as road teams, so their "home edge"
was computed from a handful of games — and, worse, they sat inside the league mean
that all 136 real teams were being shrunk toward.

Restricting the table to rated teams shrank the spread from ±12 to ±8.8, produced a
believable ordering, and made team-specific home field look **worse**, not better:
k=12 went from +.00116 to +.00213 against flat. The pollution had been flattering the
hypothesis.

This is the case that prompted the FBS-only sweep recorded in `src/data/fbs.py`.

## Reproduction

```powershell
python -m scripts.team_hfa_backtest
```

Artifact: `artifacts/team_hfa_backtest.json`.
