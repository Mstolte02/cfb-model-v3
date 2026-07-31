# war_model — player wins above replacement

Was `~/Downloads/rb-win-model`, which named the model after the running backs it
started as and stopped describing it several rewrites ago. It builds a WAR figure for
every FBS player, and `src/data/war.py` in the parent repo consumes exactly three
files from here.

## What it does

Five stages, each reading the one before it:

1. **facets** — `facets.py`, `candidates.py`, `cfbd_facets.py`. A facet is one job
   measured one way: a grade or rate over its own denominator, standardized within
   season and multiplied by volume, so an average snap is worth zero. 98 of them —
   86 generated from the PFF exports, 12 from CFBD play value.
2. **weights** — `build_hybrid.py`, `two_level_weights.py`. How much each facet is
   worth, fitted against the *following* season's wins.
3. **Massey → wins** — the weighted facet total becomes a schedule-adjusted team
   rating, and the rating-to-win-percentage slope puts a player's share of it in
   units of wins. `waa`, then `war` after replacement credit.
4. **roster** — `build_roster_2026.py` joins the Ourlads two-deep to that history.
5. **projection** — `project_2026_v2.py`, a gradient-boosted model over every
   two-deep slot. Its output is what the site shows.

## The weighting, and why it changed

`WAR_WEIGHTS=twolevel` is the default. The old flat non-negative fit put 98 collinear
facets in one regression and let it decide all of them at once, which it cannot do.
Bootstrapped over team-seasons, the QB *block* is stable at 15.5% while inside it
`QB_twp_rate` ranges 7.6–28.1% and `QB_pass` — the actual passing grade — ranges
0–12.6% and is driven to exactly zero in 16% of resamples. NNLS is a sparse solver;
faced with near-duplicates it does not share weight, it zeroes one.

That is harmless for predicting wins and fatal for WAR, which is an *attribution*.
So the questions are separated: groups are fitted against wins, collinear concepts
inside a group split by univariate strength, and facets inside a concept split by
year-over-year reliability. Measured by `weighting_compare.py`:

| | flat NNLS | twolevel |
|---|---|---|
| forward r | .523 | .500 |
| median rank sd over resamples | 532 | 254 |
| mean Spearman vs base ranking | .944 | .985 |
| facets zeroed in ≥25% of fits | 45/98 | 8/98 |

Losing .023 of forward r to halve the resampling spread of the rankings is the right
trade here. The win predictor is downstream in the parent repo and blends this in at
40% of one of six inputs; what WAR is *for* is saying who was worth what.

Set `WAR_WEIGHTS=nonneg` to get the old behaviour back.

## Interface with the parent repo

`src/data/war.py` reads only:

- `hybrid_player_war.csv` — historical WAR per player-season
- `projections_2026_v2.csv` — projected 2026 WAR per two-deep slot
- `talent_noise.json` — one scalar, how much noise the 2026 talent feature carries

`WAR_DIR` overrides the location if you build elsewhere.

## Running it

Use the parent repo's venv (`../venv/bin/python`); there is no separate environment
any more. Order:

```
python build_hybrid.py        # facet weights -> player WAR
python uncertainty.py         # reliability, error bars, talent_noise.json
python project_2026_v2.py     # 2026 projections
python make_diagnostics_report.py
```

## What is not in git

The PFF exports themselves are licensed and stay outside the repo entirely —
`facets.py` points at `~/Downloads/pff_exports` and `build_roster_2026.py` at the
two-deep workbook. Neither is committed and neither should be.

`.gitignore` also drops the CFBD/Ourlads caches (343 MB, rebuildable from the API with
a key), the bulk per-facet intermediates (up to 48 MB each, rebuilt by the stage
above), and the artifacts of the superseded PFF-only build. What is committed is the
source, the fitted parameters, and the three files above.
