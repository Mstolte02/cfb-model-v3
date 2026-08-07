# war_model — player wins above replacement

Was `~/Downloads/rb-win-model`, which named the model after the running backs it
started as and stopped describing it several rewrites ago. It builds a WAR figure for
every FBS player, and `src/data/war.py` in the parent repo consumes exactly three
files from here.

## What it does

Five stages, each reading the one before it:

1. **facets** — `facets.py`, `candidates.py`, `cfbd_facets.py`. A facet is one job
   measured one way: a grade or rate over its own denominator, standardized within
   season and multiplied by volume, so an average snap is worth zero. 103 of them —
   91 generated from the PFF exports, 12 from CFBD play value. `consolidate.py` then
   merges the near-duplicate clusters onto their first principal component, which
   leaves **87**.
   Standardization is role-relative (`WAR_ROLE_NORM=partial`): a receiver is scored
   against the pooled distribution with his depth-chart tier's mean removed, so a
   fifth receiver is not charged for being a fifth receiver.
2. **weights** — `build_hybrid.py`, `two_level_weights.py`. How much each facet is
   worth, fitted against the *following* season's wins. Blocks are (job × position
   group), so positional value comes from that regression rather than from a rule;
   inside a block, weight splits by **validity** — how much of the facet survives
   the player changing teams — rather than by raw repeatability.
3. **Massey → wins** — the weighted facet total becomes a schedule-adjusted team
   rating, and the rating-to-win-percentage slope puts a player's share of it in
   units of wins. `waa`, then `war` after replacement credit. Summed player WAA
   equals `games × slope × rating` **exactly**: the schedule term is allocated to the
   players rather than discarded, which it used to be.
4. **roster** — `build_roster_2026.py` joins the TWO-DEEP two-deep to that history.
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
So the questions are separated: blocks are fitted against wins, collinear concepts
inside a group split by univariate strength, and facets inside a block split by
validity. Measured by `weighting_compare.py`:

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

### The line is two groups

Blocks are (job × position group), so as long as the line was ONE group the level-1
regression had to answer a single question for two jobs — a tackle alone against an
edge rusher, an interior lineman working in double teams — and a facet's weight inside
that block split on a validity figure pooled over both. PFF's blocking export tags
`T`, `G` and `C` separately and every charted depth-chart slot resolves to
LT/LG/C/RG/RT, so the split is available at both ends. Centres go with the guards:
three groups would have fitted the centre block on ~300 players a season, and C-vs-G
is the distinction the grades separate least.

The fit does not think the two jobs are worth the same, though the gap is smaller than
the headline share suggests, because there are three interior linemen and two tackles:

| | share of total facet weight | per starter |
|---|---|---|
| **OT** (2 on the field) | 5.49% | 2.74% |
| **IOL** (3 on the field) | 5.25% | 1.75% |

A tackle prices at **1.57×** an interior lineman per man. The split is not simply
"tackles matter more": `IOL_run_block` (.0319) is the larger of the two run-blocking
facets and `OT_pass_block` (.0100) is half again the interior's, which is the fit
saying the interior is worth more in the run game and the tackles in protection —
exactly the distinction one merged group could not express.

That is worth stating carefully because a first pass at this reported 8.8% against
5.0% and a 2.7× ratio, and it was an artefact. `concepts.py` still listed the facets
under their old `OL_*` names, so every line facet fell through `concept_map`'s
unclaimed branch into its own `_solo_` concept, and the two-level fit priced ten
orphans instead of two blocks. The numbers above are from the build where they sit in
`pass_protection`, `run_blocking` and `discipline` beside the tight ends and backs.

At team level the split is close to neutral, which is the point — it is an attribution
change, not an accuracy claim:

| | before the split | after |
|---|---|---|
| weighted facet total vs this season's wins | .847 | .847 |
| ...vs next season's wins | .527 | .527 |
| season-blocked CV r | .854 | .855 |

The split is for VALUATION only. `build_roster_2026` matches a player to his history
on a COARSE group that collapses OT and IOL back to one line, because that merge is
about identity rather than value and a guard who moves out to tackle is the same man —
matching on the split group refused him and cost the line 4.8 points of coverage.
`build_hybrid`'s CFBD↔PFF join key stays coarse for the same reason, plus a harder
one: CFBD calls roughly nine in ten linemen "OL" and cannot resolve the split at all.

### A block may not span position groups

`receiving` used to hold `TE_core`, `WR_yprr`, `RB_pass_route` and six more, so
whatever rule divided that concept was deciding what a tight end is worth against a
receiver — and no rule at that level can. Splitting by validity gave tight ends the
concept outright and put them at 12.6% of all WAR; splitting by snaps collapsed them
to 2.6% and double-counted volume besides, since value is already z × snaps. Blocks
are (job × position group) now and the level-1 regression decides, which is the one
place positional value can honestly come from.

### Within a block, validity rather than repeatability

A statistic can repeat perfectly and measure nothing about the player — his
quarterback is still there next year. Validity residualises a player's z on his team
and his role and correlates what is left with the same quantity **after he changes
teams**. The most demoted facet is `WR_avg_depth_of_target`, .616 repeatability
against .245 validity, which is aDOT correctly identified as a property of the
offence rather than the receiver; tackling is promoted, being a skill that travels.

Everything above is measured on that transfer test, over 4,509 player-seasons that
moved: **.4137 → .4984**.

### Block weights come with a band

`block_weight_band.csv` and the build's own output carry a 5–95% interval from
resampling the 135 **teams**, not the 1,438 team-seasons — eleven seasons of Alabama
are not eleven independent observations. Passing is 9.5% [7.0–12.3]; the small blocks
are zeroed in a majority of resamples and should be read as "not identified" rather
than "small".

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
python scrape_twodeep.py               # the 2026 charts (cached; --refresh to re-pull)
python build_hybrid.py                 # facet weights -> player WAR
python build_roster_2026.py            # two-deep joined to the WAR history
python depth_correction.py --roster    # pin starters BEFORE projecting
python project_2026_v2.py              # 2026 projections
python uncertainty.py                  # reliability, error bars, talent_noise.json
python depth_correction.py --in projections_2026_v2.csv \
                           --out projections_2026_final.csv
python external_validation.py          # convergence with EA's CFB 27 ratings
python staleness_check.py              # mtimes AND contents
python make_diagnostics_report.py
```

## What the numbers currently are

Straight from the artifacts, not typed in from a previous run — `staleness_check.py`
compares the counts below against the live files and complains when they drift.

| | |
|---|---|
| facets, after consolidation | 87<!--live:n_facets--> (from 103) |
| blocks × groups | 34 concepts in 33 groups |
| weighted facet total vs this season's wins | r = .847 |
| ...vs next season's wins | r = .527 |
| Massey rating vs adjusted win pct | r = .702 |
| de-attenuation k | 0.954 |
| player-seasons / total WAR | 89,893 / 6,069 |
| projection holdout, 2025, ex-ante features only | r = .607<!--live:holdout_r--> |
| ...against the carry-forward baseline it has to beat | .530<!--live:carry_r--> |

The projection's holdout r was published as .631 and is not the same number: that one
was measured with `is_starter` in the feature set, which was computed from the
outcome season in training and from a preseason depth chart at serve time. It was
worth .064 of correlation on an otherwise identical model, measured with
both feature sets on one build.

**The holdout fell .621 → .610 when the line split, and that is a real cost.** The two
figures are not measured on identical populations — splitting the line changes which
slots exist, and the holdout went from 6,027 rows to 5,927 — so they are not strictly
comparable. But the baseline moved with it (carry-forward .534 → .530), and the
model's margin over that baseline still narrowed, **+.087 → +.080**. Splitting a group
means each half is fitted on fewer players, and at the player level that costs a
little forward accuracy. It is bought back as attribution: the model can now say a
tackle and a guard are worth different amounts, which is what WAR is *for*, and at
team level the split is neutral (see the table above).

**It then fell .610 → .607 on the recruiting re-pull (7-Aug-2026), and that is the
price of a better input, not a regression.** CFBD had backfilled the bottom of every
high-school class since the July pull — the 2025 class went from 2,507 recruits to
4,120, and from 12 two-star recruits to 427 — so the cached classes had been, in
effect, a top-of-the-class sample. Filling in the tail gives the low-history players a
real low rating where they used to have a missing one, and missingness was doing some
of that work: "no rating" correlated with "lightly recruited" almost perfectly while
the tail was absent, and now it means something weaker. The feature got more honest
and slightly less discriminating at the same time. Recruiting now covers 84.4% of the
2026 two-deep and 79.7% of the players with no PFF snaps at all, which are the ones
the prior actually carries. (The before-figures are not quoted because `rec_*.json` is
not in git and the re-pull overwrote it; the class sizes above come from comparing the
API against the cache before it was replaced.)

## What is not in git

The PFF exports themselves are licensed and stay outside the repo entirely —
`facets.py` points at `~/Downloads/pff_exports` and `build_roster_2026.py` at the
two-deep workbook. Neither is committed and neither should be.

`.gitignore` also drops the CFBD/Ourlads caches (343 MB, rebuildable from the API with
a key), the bulk per-facet intermediates (up to 48 MB each, rebuilt by the stage
above), and the artifacts of the superseded PFF-only build. What is committed is the
source, the fitted parameters, and the three files above.
