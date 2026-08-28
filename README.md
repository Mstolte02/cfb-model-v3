# CFB Predictive Model v4

## v4.1 — projected rosters, clean time, reciprocal matchups, live evidence

V4 is the production team model. It is a deliberately smaller model than v3 because
the rigorous audit found that part of the old player signal knew which players would
participate—and how many snaps they would take—in the season it was supposedly
predicting. It also found that the static model lost to CFBD's pregame Elo because it
never learned from the season in progress. The complete evidence trail is in
[`audit/CFB_MODEL_V3_AUDIT.md`](audit/CFB_MODEL_V3_AUDIT.md).

The temporal contract is explicit: a season-N forecast may use completed N-1 team
performance, earlier player history, the published season-N roster, and information
known in the N preseason (recruiting and returning production). Every player on the
roster is projected before the position-room top K are selected. It may not use
season-N participation, roles, performance, or realized snaps. Candidate features and
every numeric parameter are selected inside expanding forward folds. Historical CFBD
roster caches do not carry retrieval timestamps, so they remove outcome-based player
selection but are not a substitute for a frozen preseason transaction archive.

The production selector chose opponent-adjusted offense, defense, recruiting talent,
returning production, and all-roster projected WAR. The player extension cleared the
predeclared 0.001 adoption bar in consecutive selection windows (+0.00213 and
+0.00232 Brier). Lagged team PFF/WAR and granular matchup extensions did not clear it.
An NFL-style player-production forecast was also ported to six CFB season markets,
then expanded with production×WAR/usage interactions, two-part participation hurdles,
Gaussian/Laplace WAR losses, and an ensemble. The selected extensions improve
player-level WAR correlation in every modern fold, but team WAR remains mixed and the
game gain is only 0.00030 online Brier beyond projected WAR (bootstrap 95% interval
−0.00090 to +0.00026). Consecutive incremental selection gains are 0.00063 and
0.00020, so the production-derived game features are
rejected and remain research-only. The full evidence is in
[`audit/PLAYER_PRODUCTION_EXPERIMENTS.md`](audit/PLAYER_PRODUCTION_EXPERIMENTS.md).
A separate head-coach experiment passed its mover-graph gate (72 qualifying movers;
65.1% of team-seasons in the largest component), but the apparent 34–36% naive coach
variance collapses to about 0.9% overall, 5.3% offense, and 2.5% defense under partial
pooling, with coach/program correlations around .75. Leakage-safe coach means, O/D
effects, mixed interactions, tenure/change features, and first-year uncertainty
shocks all worsen the v4 replay. The flat uncertainty control also beats every
coach-specific shock on the rating target. Nothing coach-derived ships; see
[`audit/COACH_EFFECTS_EXPERIMENTS.md`](audit/COACH_EFFECTS_EXPERIMENTS.md).
Rest/load context worsened the online replay; travel improved just 0.00010 and was
directionally inconsistent, so none of those features ship. Full results are in
[`audit/CONTEXT_AND_PLAYER_PROJECTION_EXPERIMENTS.md`](audit/CONTEXT_AND_PLAYER_PROJECTION_EXPERIMENTS.md).
That audit now also tests quantitative fragility, reversibility, and structural
weighting both as post-hoc corrections and inside the initial reciprocal fit. Only
the jointly fitted rolling reversibility family clears the research bar; it remains a
candidate pending a dedicated live game-context path rather than being forced into
the season-fixed power-rating frame.
Pace identity, scripted-game windows, tempo control, and prior-year QB time-to-throw
versus pressure were also tested from leakage-safe CFBD drive and PFF/TruMedia data.
Several replace missing current form in a static model, but none clears the adoption
bar after the weekly rating update is retained; they remain research-only.
Team WAR was then broken out by facet and position — 25 units that sum back to the
team total exactly, and 11 preseason position rooms that sum to `war_projected`
exactly — and each offensive unit was pitted against the opposing defensive unit as a
nonlinear antisymmetric cross term. Every family loses to the shipping five columns on
both metrics, and the pairing loses by more than the split it is built on. The reason
is measurement, not football: team WAR persists year to year at r .693 and **every one
of its 25 parts persists less**, from .659 down to .187, with the defensive halves of
the marquee matchups — corner coverage .301, safety coverage .231 — among the least
stable of all. Nothing here ships. The experiment also found that the historical
preseason projection behind `war_projected` never sees most offensive linemen, because
CFBD's roster feed lists them as `OL` and `project_2026_v2.CFBD_TO_GROUP` has no `OL`
key; that is a live defect in a shipping feature and is written up, not yet fixed. See
[`audit/FACET_MATCHUP_EXPERIMENTS.md`](audit/FACET_MATCHUP_EXPERIMENTS.md).

The feature set was cut from fifteen columns to seven in Aug-2026 after an audit found
the recruiting construct entered five times plus a sixth inside `talent`, and
`portal_net_rated` entering as a construction identity of two columns already present.
Six of fifteen coefficients flipped sign across bootstrap refits and the fit wanted
ten times more shrinkage than it had. The five positional recruiting columns are now
one principal component, the redundant portal columns are gone, and `talent` is
orthogonalised against recruiting so its published coefficient stops reading as
"talent makes teams lose". Every coefficient is now sign-stable, the train/test gap
fell fivefold, and the reduced set won the standard forward selection outright. See
[`audit/STANDARDISATION_AND_COLLINEARITY.md`](audit/STANDARDISATION_AND_COLLINEARITY.md).

| Strict expanding replay, 2022–25 | Games | Brier | Log loss | Accuracy |
|---|---:|---:|---:|---:|
| V4 preseason/static | 2,913 | .2025 | .5864 | 67.52% |
| **V4 weekly pregame update** | **2,913** | **.1845** | **.5452** | **71.47%** |
| CFBD pregame Elo, same games | 2,913 | .1898 | .5613 | 70.99% |

V4 now beats Elo on this replay by a margin that clears its own uncertainty: the
paired season-week bootstrap difference is −0.00533 Brier with a 95% interval of
[−0.00976, −0.00118], excluding zero, and Elo is the better model in 0.6% of draws.
Before the feature reduction the same comparison was −0.00270 with an interval of
[−0.00738, +0.00183] and the honest reading was "competitive with Elo". That
qualification no longer applies, though beating Elo is a low bar and no historical
closing-line file is present, so the model still has no market benchmark.

Neutral-site matchup coherence is enforced by construction. Team A versus Team B uses
one antisymmetric feature difference, neither fitted model has an intercept, and the
calibrator is temperature-only. Therefore reversing the teams produces the exact
probability complement and opposite predicted margin. During the season, all games in
a week are predicted from the start-of-week state before any result in that slate is
applied.

### Production workflow

```powershell
# Full forward selection, final fit, 2026 ratings, and empty live state
python -m scripts.train_v4

# Re-run the strict historical replay and same-game Elo benchmark
python -m scripts.v4_backtest

# Re-run the Four-Pass proxy overlay and joint initial-fit experiments
python -m scripts.four_pass_backtest
python -m scripts.four_pass_initial_backtest

# Re-run pace, game-script/control, and quick-pass/pressure experiments
python -m scripts.tempo_style_backtest

# Re-run the facet x position unit and room matchup experiments
python war_model/preseason_group_war.py
python -m scripts.facet_matchup_backtest

# Apply newly completed games once, then publish current ratings
python -m scripts.update_v4
python -m scripts.rank

# Rebuild season/CFP simulation and browser data from v4
python -m scripts.simulate_playoff 20000
python -m scripts.export_viz

# Temporal, reciprocity, selection, and weekly-order invariants
python -m unittest discover -s tests -v
```

The old build remains reproducible with `python -m scripts.train_legacy`, but it is no
longer the default. Current-roster player WAR remains visible in roster reports and
now enters the team prior through the same all-roster projection family tested
historically; individual position contributions are still reporting rather than
causal game-level effects.

### 2026 returning-production definition

The historical v4 feature is CFBD `percentPPA`, so the live frame now reads the
checked-in `data/returning_2026_cfbd.csv` snapshot of that same field. The earlier
live path substituted the separately published Connelly/ESPN estimate from
`data/returning_2026.csv`. Both are reasonable continuity statistics, but they are
not interchangeable model features: across the 135 matched 2026 teams their
correlation is only 0.30. The old path therefore changed the feature definition
between training and prediction. The Connelly offense/defense splits remain useful
descriptive context; they no longer occupy a coefficient trained on CFBD PPA share.

## v3.10 — one build, a split line, and EA back in its lane

Five changes asked for directly, and three silent bugs they exposed.

**The site ships one set of numbers.** It shipped two — the default and a PFF-only
build with no outside opinion anywhere — behind a header toggle, each a complete
build written under its own suffix. The toggle is gone and so is everything behind
it: `SITE_VARIANTS`, `variant_suffix()`, the `CFB_WAR_VARIANT` environment guard, the
variant arguments to `rank` / `simulate_playoff` / `export_viz`, and the `_pff`
artifacts. There is one answer, so there is nothing to choose between.

**EA no longer overrides a player we have evidence about.** The blend had three rules
and the first was that OL, WR and DT took EA's *ordering* at every snap level — not
just for the unproven. That rule is gone. What is left is the one EA is actually good
for: ranking players with no track record to rank them by. The quarterback sheet is
untouched and still has the last word.

| whose number | slots |
|---|---|
| ours (PFF + CFBD facets) | 2,302 |
| EA's ordering, under 300 prior snaps | 3,482 |
| the five-source quarterback composite | 118 |

**709 proven OL/WR/DT slots move back onto our own numbers.** The removed rule never
had a result behind it — every test in `war_model/ea/gap_analysis.py` says EA is
*different* in the tail, none says it is *better*, and the one accuracy test that
exists had EA behind at .42 against .51.

**The offensive line is two position groups.** Blocks are (job × position group), so
one line group meant the level-1 regression answered a single question for two jobs.
PFF's blocking export tags `T`, `G` and `C` separately and every charted depth-chart
slot resolves to LT/LG/C/RG/RT, so the split is available at both ends. Centres go
with the guards — three groups would fit the centre block on ~300 players a season.

| | share of facet weight | per starter |
|---|---|---|
| **OT** (2 on the field) | 5.49% | 2.74% |
| **IOL** (3 on the field) | 5.25% | 1.75% |

A tackle prices at **1.57×** an interior lineman per man, and the split is not just
"tackles matter more": the interior carries the larger run-blocking facet (.0319 to
.0298) and the tackles the larger pass-blocking one. At team level it is neutral —
same-season r .847, next-season .527, CV .855, all within .001 of the merged build.
At player level it costs a little: the 2025 holdout goes **.621 → .610** against a
carry-forward baseline that moved .534 → .530, so the model's margin narrows +.087 →
+.080. Splitting a group fits each half on fewer players. It is bought back as
attribution, which is what WAR is for.

**A new depth-chart source: thetwodeep.com.** All 138 FBS programs, where Ourlads had
136 — North Dakota State and Sacramento State play their first FBS season in 2026 and
were simply absent from the model. Its labels are granular at every charted slot,
which is what lets `broad_group` carry OT and IOL at all, and its class years arrive
as `RS JR` / `SO/TR`, the same shape `parse_eligibility` already read. Ourlads stays
as the fallback. `cfbdepth.com` was evaluated and **rejected**: its terms of service
prohibit automated access, it 403s scripted requests, and it sells data downloads as a
paid tier. It is the only one of the three with a structured injury feed, so injuries
remain manual in `war_model/availability_2026.csv`.

The headline roster match rate falls 80.3% → 76.7%, and none of it is a regression:
for the 5,455 players listed by *both* sources it is unchanged (79.6% against 80.3%),
and the drop is 445 newly-listed players — 208 of them incoming transfers with no FBS
history — plus two teams that have never played an FBS down.

**Class-year filtering on the Players tab**, plus a transfers-only filter. The
position filter is ordered the way a roster is read — QB, RB, WR, TE, OT, IOL, DT,
EDGE, LB, CB, SAF — rather than alphabetically, which had put CB and DT above QB. The
Class column reads `RS SR`. The starters/backups scope filter is gone.

**Three silent bugs, all found by these changes**, all worth recording because none
raised anything — they just produced confident wrong numbers:

- **`is_transfer` meant two different things on the two sides of the model.**
  Training computes it in `project_2026_v2` as `prev_team.notna() & (prev_team !=
  team)` — the player was somewhere else *last season*. Serving passed through the
  depth chart's `/TR` marker, which is true for anyone who has *ever* transferred and
  stays true for the rest of his career. So the feature arrived at **55.6%** of slots
  against a training rate less than half that: the model learned one thing and was
  asked to use another. Exactly the shape of the `is_starter` leak in v3.9.
  `build_roster_2026` now recomputes it off the same PFF history the training side
  uses, so the definitions are identical by construction. **55.6% → 23.7%**, median
  21% per team, Air Force and Army at 0%, Notre Dame's six arrivals naming
  Pittsburgh, Ohio State, Michigan, Alabama and Oregon.

- `src/data/pff.py` keys its fitted positional weights on `"OL"`, so `weights.get("OT",
  0)` returned **zero** and the entire offensive line dropped out of the PFF talent
  feature. Fixed by collapsing the two-deep onto that module's own vocabulary; its
  weights were NNLS-fitted with the line as one group, and splitting .12 in two would
  be inventing a number where one was measured.
- `war_model/concepts.py` still listed the facets under their old `OL_*` names.
  `concept_map` sends anything unclaimed to `_solo_<facet>`, which keeps a facet in
  the model but takes it out of every concept — so `build_ea_war`, which is keyed on
  concepts, dropped **all 1,534 EA linemen** (EA coverage was 88–95% everywhere and
  **0.0%** at OT and IOL), and the two-level fit priced ten orphans instead of two
  blocks. That bad fit is where a first draft of this section got "8.8% vs 5.0%, a
  2.7× ratio" — the corrected figures are in the table above.

## v3.9 — an audit, worked through

An external review of the whole model. Most of it was right, one recommendation was
wrong and is documented as such, and one of the fixes turned up a leak worth 10% of
the flagship accuracy figure.

**The headline accuracy was measured on the split its knobs were chosen on.** At
least a dozen were: the opponent-adjustment strength, the uncertainty lambda, two
talent blends, the ensemble weight, the shrinkage lambda, the Pythagorean exponent,
the dropped-feature list. `scripts/nested_cv.py` puts the tuning inside a held-out
season and scores each season once.

| | Brier |
|---|---|
| LOSO at the tuned config *(what was reported)* | .2026 |
| **nested, honest forward estimate** | **.2079** |

+2.6% selection optimism. `talent_blend` takes all three of its grid values across the
five outer folds, which is the data declining to pin that knob down at all.

**The projection was told who started.** `is_starter` was computed from realised snap
rank in training and read off a preseason depth chart at serve time. Matching the two
sides' *density* had fixed the wrong half. Worth **.064 of correlation** on an
otherwise identical model — r .617 without it against .681 with, both feature sets
scored on one build — so the published .631 was measuring a task nobody can perform. Replaced by `prior_rank`, which is last
season's snaps on both sides.

**The schedule adjustment never reached the players.** WAA was `games × slope × c_t ×
f_contrib`, the exact marginal effect of a player on his own team's rating but not an
allocation of it — the rest, which is *who you played*, is 29.0% of the variance of
the ratings and correlated **−0.0013** with what the players were paid. Allocating it
in proportion to |f_contrib| makes the sum exact. Conference USA falls 1.31 team wins
in 2025 and the Big Ten rises 0.91.

**The opponent adjustment was sitting on a singularity.** The SRS fixed point was 25
rounds of substitution — a power iteration on an operator whose spectral radius is
exactly `alpha` — so the shipped `alpha = 1.0` never converged and the "interior
optimum" recorded below was the solver, not football. Solved directly on the
complement of the constants and re-swept, **1.00 is worse than the whole plateau it
used to sit on top of**: .2135 against .2085 at 0.85, which now ships.

**Nothing had ever been checked against an outside opinion.** Every validation was
internal, and all of them are satisfied by a number that faithfully reproduces PFF's
grades times a snap count — which is the one hypothesis they cannot rule out, since
those are the inputs. `war_model/external_validation.py` scores against EA's CFB 27
ratings, beside raw playing time and beside the partial correlation holding playing
time fixed. Within position group the 2026 projection reaches r .69–.83 against
snaps' .42–.73 (partial .51–.65); historical WAR is a weaker story, and at corner,
linebacker, safety, receiver and on the line raw playing time matches EA about as
well as WAR does.

**The intervals were decorative.** ±0.41 team wins, never scored against an outcome.
Measurement noise is now separated from real year-to-year change, residual sd scales
with the level of the prediction, and a team-level common shock enters the total as
(n·sd)² rather than n·sd². **±1.65**, and out-of-sample the nominal 68% band covers
88.3% — conservative rather than calibrated, recorded in `interval_coverage.json`
rather than tuned away.

**Rejected on the evidence: re-targeting the weights to contemporaneous wins.** Built
in full — `war_model/score_state.py` reconstructs each team's in-game score-state
profile from quarter-by-quarter line scores and projects out the part its own
production does not explain. It moved the concept weights exactly as predicted (pass
protection 2.7% → 5.1%, tackling 0.0% → 4.4%) and was still worse on the transfer
test (.4487 against .4967, the two targets scored on one build), and it inverted
tight ends and the offensive line. Kept behind `WAR_TARGET=contemporaneous` so the
result can be reproduced rather than trusted.

Also: merges key on `player_id` (the PFF loader never read the column, and was
deleting one of two real players on 743 keys); position groups are snap-weighted over
every contributor instead of `[1.0, 0.45]` over the top two by *games*; the 98 facets
are consolidated to 87<!--live:n_facets-->; blocks are (job × position group) so positional value comes
from the fit; within a block, weight splits by transfer **validity** rather than
repeatability; there is one 2026 roster and one QB valuation instead of two and three;
and `staleness_check.py` compares artifact *contents*, which immediately found this
README claiming 98 facets and a stale ×1.64 rescale in `viz/app.js`.

Transfer validity of player WAR, the measure most of this was scored on, goes
**.4137 → .4984**.

## v3.8 — a denominator, an ordering, and a calculator

Three changes, one of which turned up a bug that had silently undone an earlier fix.

**The CFBD defensive facets were measuring playing time.** CFBD publishes no
individual snap counts on defense, so `cfbd_facets.py` denominated havoc, tackle and
pass-defensed rates by the team's defensive plays — every defender on the roster
carrying the same ~870. That makes the facet a proxy for how much a man played:

| facet | r(own snaps, z) before | after | weight |
|---|---|---|---|
| `cfbd_tackle_lb` | **0.917** | −0.165 | 0.01% |
| `cfbd_tackle_db` | **0.877** | −0.236 | 0.01% |
| `cfbd_havoc_lb` | 0.786 | 0.037 | 0.94% |
| `cfbd_havoc_db` | 0.777 | 0.038 | 1.23% |
| `cfbd_havoc_dl` | 0.741 | 0.145 | 1.31% |
| `cfbd_cov_db` | 0.713 | 0.053 | 1.54% |
| *PFF `LB_tackle`, individual-snap denominator* | *0.134* | | |

A rotational defender was charged ~0.75 SD of negative value for not being on the
field, and the two purest playing-time proxies had already been driven to zero weight
by the fit. It also cost those facets their share of the replacement pool — 6.4% of
total facet weight — because `build_hybrid` withheld per-snap credit from any facet
whose volume column is not the player's own playing time.

PFF's defense export has `snap_counts_defense`, so the denominator existed, just in
the other source. Joining it on `(season, team, normalized name)` covers **87.7%** of
FBS defensive rows; the rest have no snap count and are dropped rather than imputed,
since inventing the denominator is worse than not having the row. Shrinkage (k = 200,
half weight at roughly the median snap count) is now needed and applied, because
dividing by a man's own 12 snaps lets one tackle-for-loss post a rate no starter can
reach. **Every facet now pays replacement credit**, up from 92 of 98. (v3.9
consolidates the near-duplicates, so the live count is 82; see above.)

Forward r **.500 → .519**, Massey vs adjusted win pct **.698 → .710**, and the
WAR-equals-wins identity fits slightly better at **r .8344 → .8481**.

**`deattenuate()` was a fixed point that no-ops on every run after the first.** It
calibrated against the *previous* build's `hybrid_player_war.csv` — which had already
been de-attenuated — so it found a converged system and returned exactly `k = 1.0000`.
The correction then never reached the new numbers and summed team WAR went back to
regressing on actual wins above replacement at **1.308**: the precise bug the
de-attenuation exists to fix, reintroduced by rebuilding. It survived because
`k = 1.0000` reads as a healthy answer rather than a skipped step. It now solves
against the build in hand (waa is linear in the slope, so calibrating in place is
exact), and `build_hybrid` **asserts** the identity rather than printing it.

**The quarterback sheet now has the last word.** `depth_correction`'s reweight scales
each team's starter by a team-specific factor — 1.00 to 1.63, median 1.027, set
entirely by how much WAR that room's backups carried — and running it *after* the
quantile map re-sorted the quarterbacks the map had just ordered. Rule 2 moved out of
`blend_projection.main` into `apply_qb_map()`, which `depth_correction` calls last:

| | Spearman vs the sheet | largest rank move |
|---|---|---|
| map before reweight (was) | 0.9885 | 20 places |
| **map after reweight (now)** | **0.9999** | **1** |

The residual is the three-way tie at `Average` = 1.02; every remaining mismatch is a
tie broken arbitrarily. The cost is that a room's backup share no longer lands exactly
on the no-injury target — per-room mean deviation **0.59pp → 1.38pp** — which is the
right trade, since being a third of a point off an injury correction matters less than
the starting quarterbacks being in the wrong order. The *aggregate* share cannot move
at all; the map is a permutation over exactly those rooms.

**A Testing tab with a rating calculator.** One team at a time, every intermediate
between its roster's projected WAR and its power rating: team WAR → z → the
three-source talent blend → the uncertainty shrink into O and D → the six model
features and their coefficients → the mean neutral win probability that is the power
rating. Every input is editable and everything below it recomputes, including the
team's rank against the other 135.

The numbers come from a `derivation` block in `model.json` that `export_viz` fills
straight out of `build_projection_frame` — the same call `scripts/rank.py` makes —
rather than recomputing the steps, which would work right up until one copy changed
and the other did not. The page carries the same guard `livePower()` has: with no
edits the chain has to reproduce the exported O, D and power, and it says so on
screen. It does, for all 136 teams, worst case 6.9e-5.

Mostly it exists to make one thing legible: **talent has a coefficient of exactly zero
and still moves the rating**, through the shrink rather than through its own slot. The
service academies get their own branch, since with no recruiting composite they never
enter the three-source blend at all.

## v3.7 — head-to-head, realistic scorelines, and a résumé for the field

Five changes. Two of them are player-source decisions taken deliberately; three were
put through a backtest, and one of those came back mostly negative and shipped only in
part. That split is the point of the section.

**Quarterbacks and three position groups change source.** OL, WR and DT now take EA's
ranking of every player rather than only the unproven ones, and the starting quarterback
takes the `Average` column of `war_model/ea/qbs_2026.xlsx` — a composite of five
independent opinions (PFSN, PFF, EPA, an execs poll, EA). Both are spliced in by
**quantile map within position group**, which is what keeps them on our wins scale: the
mapping is a permutation of the `proj_war` values already in that group, so every
group's total is identical to the last decimal (largest drift 2.8e-14) and only who is
where changes. 2,198 slots move on EA's ranking, 119 of 124 named starting quarterbacks
match into the two-deep, and the league's 616.6 total WAR does not move at all — team
shifts are transfers between rosters, not new wins. The other groups keep the previous
EA-under-300-snaps rule.

**Who starts is now stated, not only inferred.** The quarterback sheet says two things
— how good the starter is *and that he is the starter* — and using only the first put a
listed backup at the top of FBS: Syracuse's Amari Odom out-snapped Steve Angeli in 2025,
so `fix_depth` flagged him the starter, the projection gave him a starter's value
because `is_starter` is an input **feature**, and the reweight handed him 95% of the
room at **1.83 WAR, above Julian Sayin**. The sheet now pins the starter, and the pin is
applied to `roster_2026.csv` *before* `project_2026_v2.py` — relabelling afterwards only
moves the mislabelled man's value onto the right man.

`war_model/availability_2026.csv` is the companion for what no model can infer:

| status | meaning |
|---|---|
| `out` | unavailable for the season — WAR zeroed, snaps redistributed to whoever is left |
| `starter` | starts at his listed position regardless of the depth chart |

Both feed a new slot-repair pass: each `(team, roster_position)` starts exactly as many
players as the two-deep lists at depth 1. `fix_depth` swaps within a position *group*, so
it could promote a backup centre over a starting right guard — seven OL rooms were
fielding two centres and no right guard, Notre Dame among them. Repair only fixes the
*count*; a position already starting the right number is left alone, because re-picking
by projected WAR would silently override `fix_depth`'s thresholded judgement (it rewrote
461 slots before that restriction).

Run order matters and is enforced in `depth_correction.main`: availability → `fix_depth`
(which now refuses to demote a pinned starter or promote an unavailable one) → slot
repair → reweight.

**Head-to-head joins the committee model; two other candidates did not.** All three
were tested in `scripts/fit_committee.py` against 12 seasons of published rankings,
with the gate being the jackknife — does the gain survive deleting any one season —
rather than a t-test, which at n=12 rejects everything.

| candidate | LOSO Spearman | vs base | verdict |
|---|---|---|---|
| base (`win% + rating + SOS + P4`) | 0.9084 | — | |
| **+ head-to-head** (within 15 places) | **0.9126** | +0.0042 | **kept** — the only one still positive with any season deleted (worst 11/12 +0.0020) |
| + preseason AP poll | 0.9083 | −0.0001 | dropped — its apparent +0.0057 alongside h2h is **2020 and nothing else**; delete that season and it turns negative |
| + loss timing | 0.9075 | −0.0008 | dropped — negative alone and in every subset without the preseason term |

The preseason result is worth stating plainly because it is the opposite of the
intuition: 2020 is the season with 563 games and teams playing between four and eleven
of them, where record and schedule stop meaning the same thing and *any* stable prior
helps. That is a fact about 2020, not evidence the committee anchors on the preseason
poll. Both rejected features are still computed and re-scored on every run, so the
finding is reproducible rather than remembered.

**Scores come in 3s and 7s, and the model now says so — for display, not for win
probability.** `scripts/score_shape.py` fits the actual distribution of the margin given
the predicted one, non-parametrically. The normal the win model uses puts **.038** on a
3-point margin when the real figure is **.106**, and — being smooth and unimodal — it
necessarily ranks a 4-point margin *above* a 3-point one when 3 is nearly three times as
common. The fitted distribution puts .106 on 3. Fixing that does *not* improve win
probability — LOSO Brier .2036 → .2038, better in 2
of 5 seasons — and on reflection it cannot, because a win probability only cares which
side of zero the margin lands on and the normal already gets that right. So the win
model is **unchanged**. What did ship is the display: the projected score is now the
most likely real scoreline near the predicted total and margin, which is exactly right
about the margin twice as often (.0257 → .0518) and about the score four times as often
(.0014 → .0064) at the same mean error per side, plus a key-number row in the matchup
simulator.

**The Ratings tab is now the odds table**, and offense/defense/talent/SOS live only on
Team Breakdown, where there is room to show them properly. **The Playoff tab uses the
space** for a résumé card per bracket team: the route in (power champion, Group of 6
bid, or at-large, with the share), the record it took to get in against the record that
missed, and a stacked bar splitting its committee score into the terms that produced it
— all conditioned on the simulations where that team was actually selected. Plus a
bubble strip of the most common last-team-in and first-team-out.

## v3.6 — the ratings stop believing last season

The question was why a team like Wake Forest carries a **top-10 defence** off an
average defensive roster, and whether strength of schedule was being applied. It was
— fully — and that turned out not to be the problem.

**Strength of schedule was already there, at full strength, and correctly tuned.**
`src/oppadj.py` runs an SRS fixed point on the O/D composites at `alpha = 1.0`, and
it does real work: the schedule term is **31% of the variance** in adjusted defence,
and it swings teams a long way (Akron 33rd → 105th, Florida 81st → 26th). Mean
adjustment by conference runs SEC +0.55 down to MAC −0.83. Two things were checked
and both came back clean: `alpha = 1.0` was the *edge* of the old tuning grid, but
extending past it shows a genuine interior optimum (Brier .2086 at 1.0, .2208 at
1.25), and the 25-iteration loop never formally converges yet its *shape* does —
25 vs 400 iterations correlates 1.00000.

> **Superseded in v3.9.** Both of those conclusions were measuring the solver. The
> iteration's spectral radius is exactly `alpha`, so 1.0 is the divergence boundary
> and the "interior optimum" was the re-standardization hiding it. On an exact solve
> the optimum is a plateau at 0.75–0.90 and 1.0 is worse than all of it.

**The two example teams said opposite things, and neither was the guess.** Wake
Forest played the 45th-toughest set of offences — dead average — and its defence was
strong on all five components independently. Auburn played the **14th** toughest, and
its adjusted defence is 11 places *higher* than its raw one. The adjustment was
crediting a hard schedule, not failing to punish a soft one.

**The real fault: every previous sweep scored this knob on the wrong target.**
`loso3`, `roster_vs_results` and the rest all score on game outcomes, and that
surface is flat — the whole lambda grid spans 0.0010 Brier, because a game's win
probability is dominated by the talent coefficient and barely notices weight moving
between the composites and the roster baseline. The complaint was never about win
probability. New script `scripts/rating_vs_realized.py` scores the *rating*:
each team's entering-season O/D against **what that unit actually produced** that
season, both opponent-adjusted, slopes refit per fold. That target discriminates.

| effective shrink toward the roster | defence r | offence r |
|---|---|---|
| 0.00 — pure prior season | 0.553 | 0.547 |
| **0.11 — what shipped** | **0.569** | **0.568** |
| 0.55 | 0.627 | 0.636 |
| **0.70 — now ships** | **0.636** | **0.652** |
| 1.00 — pure roster | 0.558 | 0.610 |

**Returning production does not identify which teams regress.** At *matched* average
shrinkage, weighting `u` by `(1 − returning production)` is **worse** than not
weighting it at all (defence r 0.597 vs 0.615). The shrink is now flat, which means
`UNCERTAINTY_LAMBDA` is the shrinkage rather than a multiplier on a mean-0.451 index
— the old 0.25 was really **0.11**, about a sixth of the optimum. It costs the win
model nothing (LOSO Brier 0.2035 either way) and all five seasons improve
monotonically, each with its own optimum in [0.65, 0.85]. `WAR_BLEND = 0.40`
independently re-confirmed as optimal on the new target.

**Talent is retired as a separate coefficient.** At 0.70 shrink the composites are
70% talent by construction, so a standalone talent column is nearly a second copy of
them: r(off_edge, talent) = 0.87 and **VIF 7.6**, against the 3.0 that was enough to
retire pythag in v3.1. The ridge did what it always does to a duplicated column — it
flipped talent negative (+0.50 → −0.22) and tripled O/D to compensate. Dropping it
costs nothing (Brier 0.2035 either way) and the weights come back clean. Talent has
not left the model; it is *inside* O and D, still drives the shrink, and is still
exported for display.

**Two bugs surfaced on the way, one of them silent:**

- `MU.assemble` built `u = 1 − returning_production`, but `build_projection_frame`,
  `spreads.py` and `serve.py` passed returning production **straight through**. The
  model trained on one definition and projected with its inverse — the 2026 ratings
  regressed hardest exactly the teams that returned the most. `MU.uncertainty_u` is
  now the single definition.
- `simulate_playoff.py` scaled its roster-uncertainty draw by `model.coef[TALENT_IX]`.
  With talent retired that is exactly zero, so the whole perturbation would have
  become a **no-op while still printing that it ran**. It now propagates through the
  O/D coefficients, which is where talent actually reaches the model.

Held-out 2025: Brier **0.2009 → 0.1996**, log-loss 0.5846 → 0.5813, accuracy .689 →
.686 (two games). What moved is the ratings, which was the point:

| | defence rank before | after | talent |
|---|---|---|---|
| Wake Forest | 7th | **22nd** | +0.10 |
| James Madison | 19th | **65th** | −0.95 |
| Auburn | 6th | 8th | +1.47 |
| Nebraska | 94th | **52nd** | +0.91 |
| USC | 62nd | **26th** | +1.31 |

**Two refinements to the adjustment were tested and rejected**
(`scripts/per_stat_oppadj.py`). Adjusting *per stat* — three features are true pairs
(havoc, red-zone TD, pressure) and can be corrected against their own counterpart
rather than the whole opposing composite — moves Brier .2035 → .2031, inside the
noise, and produces ratings correlating **.998** with the composite ones, moving
teams 2 places on average. Splitting *home from away* is a large real effect (0.51
team-sd on 13,726 team-games) that cancels within a season: everyone plays ~6 home
and ~6 away, so the implied rating bias is 0.035 team-sd against the opponent
adjustment's 0.559 — **16× smaller**. The composite correction is already saturated.

---

## v3.4 — WAR rebuilt on 11 seasons

PFF exports were extended to 2014-2025 (`~/Downloads/pff_exports`, renamed from
`pff_exports_2021_2025` — `src/data/pff.py` pointed at the old path and was crashing).
The WAR build now spans **11 seasons instead of 5**.

**2020 is excluded.** Conferences played in isolation that year, which leaves the
Massey system barely connected between them, and teams played 8.4 games on average
against a normal 11.9 — one played 3. Rating vs adjusted win pct by season:

```
.58 .79 .62 .65 .71 .72 | .22 | .72 .72 .71 .64 .64
2014 ................ 2019 | 2020 | 2021 ............ 2025
```

Every season lands between .58 and .79 except 2020 at .22. Including it dragged the
pooled figure from .67 to .48; excluding it restores .67 on 11 seasons.

**What the extra history bought:**

| | 5 seasons | 11 seasons |
|---|---|---|
| Massey rating vs win pct | 0.673 | 0.673 |
| rating → **next** season | 0.421 | **0.436** |
| team WAR → wins calibration | 0.751 | **0.782** |
| v3 LOSO Brier | 0.2047 | **0.2043** |

Same rating quality, better forward-looking behaviour, better win calibration, and a
small but real gain in the win-probability model. The talent blend was re-swept on the
rebuilt WAR and **38 / 38 / 25 is still the joint optimum**.

**Older recruiting classes were the hidden gap.** Only 4% of 2015-18 training rows
carried a star rating, against 72% for 2024, because the recruiting pull started at
2019. Extended to 2011 (`build_recruiting.REC_YEARS`) — coverage is now 73-81% across
every season.

**The projection gains plateau.** Training the player projection on different windows,
all scored on the same held-out 2025:

| training window | r | MAE | r (no history) |
|---|---|---|---|
| 2022-24 only | 0.803 | 0.0584 | 0.741 |
| 2018-19 + 2022-24 | 0.813 | 0.0577 | 0.731 |
| 2017-19 + 2022-24 | 0.818 | 0.0575 | 0.761 |
| all available (ships) | 0.817 | 0.0577 | 0.735 |

More history helps (0.803 → ~0.817) and then flattens around six seasons. "All
available" ships rather than the nominally-best window, deliberately: picking the
window by its score on the same 2025 holdout being reported would be fitting a
hyperparameter to the test set.

**Extending the win-probability model's game years was tested and rejected.** CFBD has
2015-19, but four of the ten O/D composite inputs come from TruMedia, which only exists
for 2021-25 — older seasons would be six real features and four neutral-filled, so the
model would fit one coefficient across two different definitions of the same variable.
Measured rather than assumed (`scripts/extend_years.py`): Brier 0.2031 → 0.2034,
accuracy .677 → .680. A wash, so `GAME_YEARS` stays 2021-25.

---

## v3.3 — talent weights jointly fitted, and a Method view

**The talent weights were never jointly fitted — now they are.** PFF/CFBD came from a
50/50 test and WAR was bolted on at 25% by a one-dimensional sweep holding that fixed.
`scripts/talent_sweep.py` grid-searches all 45 three-way blends under LOSO. Result: the
shipping **38 / 38 / 25** split is the joint optimum. The one-at-a-time process happened
to land on the right answer.

**WAR is not a duplicate of PFF, and cannot replace it.** The worry was reasonable —
WAR is built from PFF grades — but they correlate at **r = 0.69**, not near one. WAR
weights by snaps, uses facet weights fitted to wins, adds CFBD play value and subtracts
a replacement level; the PFF signal is a position-weighted grade average.

| talent blend | Brier |
|---|---|
| **all three (ships)** | **0.2047** |
| PFF + CFBD, no WAR | 0.2054 |
| PFF + WAR, no recruiting | 0.2057 |
| **WAR + CFBD — WAR replaces PFF** | **0.2062** |
| PFF alone | 0.2067 |
| WAR alone | 0.2072 |
| recruiting alone | 0.2088 |

Dropping PFF for WAR costs more than dropping WAR for PFF. They are complements and
PFF is the stronger of the pair. Caveat worth keeping in view: the whole surface spans
0.0041 and 7 of 45 blends sit within 0.0005 of the best, so the *composition* (three
sources beat two beat one) is the robust finding, not the exact weights.

Note 2021 is excluded from the correlation: with no 2020 grades the PFF signal falls
back to recruiting entirely and correlates at exactly 1.00, which would overstate the
overlap.

**New Method view** (`scripts/export_diagnostics.py` → `viz/data/diagnostics.json`).
Six panels, all exported from the live model rather than hand-written:

1. every data source, what it feeds, and what it actually is
2. the three talent signals, their correlation matrix, and what each combination is worth
3. feature correlations and VIFs, with retired features marked
4. LOSO by season, baselines, and a calibration plot
5. inside the WAR build — heaviest facets and the PFF/CFBD weight split
6. every modelling question asked so far, the answer, and the number behind it

---

## v3.2 — live depth charts, feature audit, formation layout

**Rosters now come from live Ourlads charts.** `scrape_ourlads.py` (in
`~/Downloads/rb-win-model`) pulls all 136 FBS two-deeps — 7,578 slots, one request
every 1.5s, cached to disk. It replaces the 27-June workbook export, which had
already drifted. Roughly eighty distinct alignment labels are normalised onto the ten
position groups; the edge rusher alone appears as JACK, RUSH, BAN, BUCK, LEO, STUD,
VIPER, WOLF, STING, JOKER, CAT, DOG and SPEAR.

**The six model inputs were audited for redundancy and two were retired.**
`scripts/feature_audit.py` (correlation + VIF on the fitted design, leave-one-feature-
out, forward selection) and `scripts/feature_sets.py` (head-to-head LOSO):

| set | n | Brier | Acc |
|---|---|---|---|
| all six (was shipping) | 6 | 0.2054 | .676 |
| **drop fp_margin** | 5 | **0.2045** | .674 |
| drop pythag | 5 | 0.2053 | .680 |
| **drop both (now ships)** | 4 | **0.2047** | .675 |
| O/D only | 2 | 0.2116 | .660 |
| talent only | 1 | 0.2154 | .653 |

- **pythag was redundant**, as suspected: r = +0.61 with O and +0.63 with D, the
  highest VIF in the set (3.0). It is built from points scored and allowed, which is
  what the O/D composites already measure.
- **fp_margin was actively costing accuracy** — forward selection adds it last and
  gets *worse*.
- **talent / returning / WAR are NOT redundant with each other.** Returning is
  nearly orthogonal to everything (max |r| = 0.21) and talent is the single most
  valuable feature in the set (dropping it costs +0.0040 Brier).

Retired features are **zeroed, not removed** (`config.DROPPED_FEATURES`): a zero
column differences to zero and fits a coefficient of exactly zero, so `model.json`,
the JS port and the playoff simulator all keep the same six-wide shape and stay
verifiably in sync. Held-out 2025 accuracy went .670 → **.680**.

**WAR cannot carry the talent slot alone** — tested, and the answer is no:

| talent source | all-six Brier | as the only feature |
|---|---|---|
| PFF+CFBD+WAR blend (ships) | 0.2054 | 0.2154 |
| WAR only | 0.2082 | 0.2288 |

A "WAR" lens would be a materially worse view than either lens that ships, so the
toggle stays **Balanced / Roster-weighted**. WAR earns its place as 25% of the talent
blend, not as a standalone signal.

**Formation layout rebuilt.** Placement is now driven by position group and unit size
rather than a lookup table keyed on the label — anything unrecognised used to stack at
the centre of the field, which is what scattered the secondary. Each level spreads
symmetrically about the middle, so every unit is centred whatever its personnel.
Flanking positions (tight ends, edge rushers) line up outside the group they play
beyond, with side taken from the label. Verified across schemes: Ohio State reads
11 personnel / 4-2 nickel, Air Force 21 / 3-3 nickel, Navy 31 personnel / 3-4.

---

## v3.1 — committee model, bracket fixes, wins ledger, lineup view

**Committee ranking fitted on its own history.** `scripts/fit_committee.py` pulls
every published CFP committee ranking from 2014-2025 (295 ranked team-seasons) and
fits the selection-day ordering from record, opponent-adjusted quality, SOS and power
membership. Leave-one-season-out Spearman against the real final ranking:
**0.885 → 0.908**. The fit reweights the proxy substantially — the committee cares
far more about schedule and conference than about raw quality:

| weight | old (hand-set) | fitted |
|---|---|---|
| win_pct | 10.00 | 10.00 |
| rating_z | 1.00 | **0.22** |
| sos_z | 0.75 | **0.71** |
| power conference | — | **+1.05** |

Notre Dame carries the power flag (CFP contract slot); including it moved LOSO
Spearman 0.898 → 0.908.

**Bracket: the G6 bid is now actually in the field.** The simulation was always
correct — every simulated season sends four P4 champions and one Group of 6 team —
but the *displayed* modal bracket took the twelve highest playoff probabilities, and
because the G6 bid rotates across a dozen candidates no single one ever cleared the
bar. The result was a bracket with no G6 representative and five SEC at-larges. The
modal field is now built the way selection works: each power conference's most likely
champion, then the most likely G6 bid, then at-larges.

**Also fixed:** a team could reach the championship game after losing its semifinal
on the scoreboard — the winner came from win probability while the score came from
the margin model, and those diverge on coin flips. Both now derive from one source.

**WAR now reads as wins.** Summed team WAR does not equal wins, for two compounding
reasons: WAR is attenuated even historically (regressing actual wins on team WAR
gives a slope near 1.6, not 1.0, because the Massey rating underneath is noisy and
OLS pulls the fitted spread inward), and a projection is a conditional mean so it
compresses further. The league mean is right; the spread is not. The team page now
applies **one league-wide de-attenuation factor (×1.64)** around the league mean and
shows a ledger that reconciles to the projected record:

```
Ohio State   1.88 replacement + 8.22 roster − 0.36 schedule = 9.73 projected wins
Toledo       1.92 replacement + 2.01 roster + 6.03 schedule = 9.96 projected wins
```

Per-team scaling was tried and rejected: it hands schedule strength to the players,
and Toledo needed a ×2.96 multiplier that said nothing about its roster.

**Garbage time excluded from CFBD PPA.** Backups held 13.6% of all QB WAR, with a
261-snap backup out-earning full-time starters. Re-pulling PPA with
`excludeGarbageTime=true` takes that to 12.0%. It costs 0.006 of team-level CV
(0.843 → 0.837), which is a real if small price; set `CFBD_GARBAGE=include` to revert.
Empirical-Bayes shrinkage of z by snaps was also tried and **removed** — it made every
measure worse and pushed the backup share *up* to 16.0%, because compressing z
compresses the facet sigma with it. See the comment in `build_hybrid.py`.

**Team page shows a lineup, not a leaderboard.** The two-deep already carries real
positional labels (LT/LG/C/RG/RT, WR-X/Z/SL, NB), so each team's listed starters are
its actual base personnel — no formation had to be assumed. Offence and defence render
as field diagrams with the personnel grouping named from the roster itself
("11 personnel", "4-2 nickel").

**Restyled:** eggshell background, Georgia throughout, warm neutral palette so team
colours carry the emphasis. Schedule spreads use an en dash with letter-spacing.

**Roster freshness:** CFBD published its 2026 rosters some time between 25-Jul and
7-Aug-2026 (15,171 players, 138 teams), and they are now in. Ourlads still supplies
the *chart* — who lines up where — because CFBD has no depth order; an LSU spot-check
against the 27-Jun two-deep xlsx matched 5 of 6 skill starters, with the RB changed.
Refreshing all 136 Ourlads charts would mean scraping a commercial site, so it is left
as a decision rather than done.

What the CFBD roster did fix is **class year**. `project_2026_v2.py` takes class from
the CFBD roster in training and had been falling back to the two-deep's own class
column at serve time, because the endpoint was empty — and the two are not the same
variable. They disagree on 1,162 of 5,339 matched slots, almost all by a year. 90.5% of slots
now take class from the same source the model was trained on; the remainder, whom CFBD
has no row for, still fall back and the run says so.

The displayed class moved with it, so the table cannot print "SR" beside a projection
conditioned on a junior. **GR is deliberately exempt:** CFBD's scale runs 1–4 with no
graduate tier, so a chart GR whom CFBD calls a 4 is one source being coarser, not two
disagreeing — collapsing him would discard a true label and empty the app's Graduates
filter from 192 slots to 18.

---

## What's new in v3 (July 2026) — player WAR + rebuilt web app

**Player WAR joins the talent signal.** `~/Downloads/rb-win-model` builds wins above
replacement per player (PFF grades + CFBD play value → Massey ratings → WAR). Two
questions were tested separately, and the answers differ:

| talent variant | Brier | log-loss | acc |
|---|---|---|---|
| CFBD recruiting only | 0.2083 | 0.6020 | .663 |
| PFF roster-aware | 0.2066 | 0.5978 | .672 |
| **WAR** (as a replacement) | 0.2066 | 0.5981 | .671 |
| blend 50/50 PFF+CFBD (v2 default) | 0.2048 | 0.5930 | .674 |

WAR does **not** beat the PFF grade signal — swapped in for it, LOSO 2022–25 is a
wash (`scripts/validate_war_talent.py`). It earns its place only as an **addition**,
because it carries the CFBD play-value signal and a depth weighting that falls out of
snap counts rather than a fitted position-weight vector. Sweeping the WAR share on
top of the shipping blend (`scripts/sweep_war_talent.py`):

| WAR share | 0.00 | 0.15 | **0.25** | 0.35 | 0.50 | 0.80 |
|---|---|---|---|---|---|---|
| Brier | .2048 | .2042 | **.2040** | .2041 | .2046 | .2059 |

A smooth interior optimum at 0.25 — Brier .2048 → **.2040**, log-loss .5930 → .5913,
accuracy 67.4% → 67.6%. Comparable in size to the entire v2 accuracy upgrade. Set
`config.WAR_BLEND = 0.0` to disable; the frame falls back cleanly if the WAR build is
absent. Note the single held-out 2025 season is flat (.2039 vs .2038) — the 4-fold
LOSO is the more reliable estimate.

**Logos.** `scripts/prepare_logos.py` now downloads ESPN's **primary** 500px mark for
every FBS team into `viz/logos/` (was `~/Downloads/cfb_alt_logos`, a mix of alternate
and retired marks). All 138 teams verified to resolve to a valid PNG. `teams.json`
also carries brand colours plus a contrast-checked foreground, which the app uses
throughout.

**Web app rebuilt** — four views:

- **Ratings** (replaces Top 25): all 136 rated teams, sortable on every column,
  filterable by search / conference / P4 / G6. Adds talent, returning, SOS and
  conference-title odds. Records come from the 20k Monte Carlo over the real 2026
  schedule.
- **Playoff Projection**: now renders the **modal bracket** with a **projected score
  for every game**, computed client-side from the same points model the matchup
  simulator uses, with winners advancing through the fixed bracket. Format unchanged
  and confirmed for 2026-27.
- **Matchup Simulator**: unchanged.
- **Team Breakdown** (replaces Ratings Lab): per-team win-distribution histogram from
  the simulation, player WAR contributions (by position group and top contributors),
  offense/defense split, model inputs on the z-scale, and a game-by-game schedule with
  projected scores, spreads and win probabilities. Team names anywhere in the app link
  here.

**Removed:** the Ratings Lab input editor, the roster-editing modal, and the
client-side playoff re-simulation that existed to fold Lab edits into the odds.
`scripts/serve.py` and its `/api/rosters` endpoint are now unused — the app is fully
static again (`python3 -m http.server 8642 -d viz`). Restore from git history if the
editor is wanted back.

**New/changed exports:** `players.json` (per-team two-deep WAR), `win_dist` and
`bracket` in `playoff*.json`, week/date/venue on `schedule.json`, and talent /
returning / pythag / SOS on `ratings*.json`.

`scripts.export_site_data` adds sportsbook context, the prior final AP poll, and roster
headshot IDs without feeding any market number back into the trained model.

**Forward market ledger (12 August 2026).** `scripts.capture_market_snapshot` now owns
weekly prices. Every successful CFBD retrieval is timestamped in
`data/market_snapshots/checks_2026.jsonl`; a provider quote is appended to
`lines_2026.jsonl` only when one of its price fields changes. This distinction matters:
CFBD supplies open/current fields but no quote timestamp, so the ledger says only when
*we observed* a value. A price is called a qualified close only when a successful
capture occurred within six hours before kickoff and at least two books carried valid
two-sided moneylines. `-100000` provider sentinels are rejected as unavailable prices.

The 15-point consensus-moneyline rule is pre-registered for 2026. It cannot enter the
watchlist with one book. Even with two books, it clears the uncertainty gate only when
the matching historical side/edge bucket's 80% Jeffreys lower win-rate bound exceeds
the best price's break-even probability by one point. The gate is still a research
label, never an automatic bet. `viz/data/market_tracking.json` publishes quote-event,
entry, settlement and consensus-CLV counts. A scheduled GitHub workflow captures every
six hours during the football calendar when the repository's `CFBD_API_KEY` secret is
configured and republishes the site only when tracked state changes.

**Betting validation (August 2026).** `scripts.betting_backtest` joins the strict
expanding-window v4 predictions to 2,872 archived CFBD posted lines and 342 historical
DraftKings team win totals. Thresholds are selected on 2022-24 and evaluated once on
2025. No market survived that holdout: spread -6.5% ROI (709 bets), moneyline -8.4%
(237), total -5.5% (707), and season win totals -8.7% (127). The website therefore
labels current differences **model gaps**, not betting edges, and shows this result
beside the boards. CFBD does not timestamp its snapshot as a true closing line, so the
audit does not call it one.

**Availability history is append-only.** New injuries, returns and starter changes go
into `war_model/availability_events_2026.csv` with observation/effective times and a
source reference. `materialize_availability.py` produces the compact current-state CSV
used by the WAR build; `depth_correction.py` refuses to run if that generated state has
drifted from the event stream. Corrections are new events with status `clear`, not edits
to old evidence.

**Roster eligibility precedence.** Current depth charts now own the displayed and
modeled class year; CFBD fills only a missing label. The prior order let a stale CFBD
year overwrite two current charts (Bear Bachmeier appeared as FR instead of SO) and
disagreed on 1,162 of 5,339 comparable 2026 slots. `class_source` is preserved through
the player export so the published value remains auditable.

**EA replacement decision.** The full-player reboot history is only CFB25-27, and a
stable archived full-player payload for both completed editions was not found. The one
clean team-level test has EA CFB26 at r=.420 against 2025 adjusted win percentage,
versus r=.476 for the current PFF+CFBD+WAR talent, with only +.001 incremental R².
Measured WAR remains the base; EA only reorders low-snap players. The machine-readable
evidence and limitations are in `war_model/war_validity_audit.json`.

**Pipeline (run in order):**

```bash
./venv/bin/python -m scripts.train                       # retrain (WAR blended in)
./venv/bin/python -m scripts.rank                        # 2026 power ratings
./venv/bin/python -m scripts.simulate_playoff 20000      # CFP odds + bracket + win dists
./venv/bin/python -m scripts.prepare_logos               # ESPN basic logos + teams.json
./venv/bin/python -m scripts.export_viz                  # data for the web app
./venv/bin/python -m scripts.export_site_data            # odds, poll + player imagery
./venv/bin/python -m scripts.export_diagnostics          # Method tab — must run last
python3 -m http.server 8642 -d viz                       # http://localhost:8642
```

### One build

The site ships **one set of numbers**. Whose judgement of each player it uses: ours,
out of the PFF and CFBD facets, with EA's *ordering* substituted only for players
under `EA_BLEND_SNAPS` prior snaps, and the starting quarterbacks ordered by the
five-source composite in `qbs_2026.xlsx` (PFSN, PFF, EPA, an execs poll, EA). Every
proven player is on our own number.

It used to ship two — that one and a PFF-only build with no outside opinion anywhere
— behind a header toggle, each a complete build with its own ratings, playoff
simulation and player values written under a `_pff` suffix. That is gone, along with
the toggle, `SITE_VARIANTS`, `variant_suffix()`, the `CFB_WAR_VARIANT` guard that kept
a run and its argument in step, and the variant arguments to `scripts.rank`,
`scripts.simulate_playoff` and `scripts.export_viz`. The build needs
`war_model/ea/blend_projection.py` run before it whenever the underlying projection
changes, to refresh `projections_2026_blended.csv`.

---

## What's new in v2 (July 2026)

**Accuracy (validated by LOSO 2022–25, `scripts/loso_experiments*.py`):**

| change | Brier | log-loss | accuracy |
|---|---|---|---|
| v1 baseline | 0.2052 | 0.5975 | 0.673 |
| drop isotonic calibration (added noise; logistic already calibrated) | 0.2048 | 0.5930 | 0.674 |
| + ridge **margin-model ensemble** (`p = 0.4·p_logistic + 0.6·Φ(margin/σ)`) | **0.2044** | **0.5919** | **0.677** |

Held-out 2025: Brier 0.2045 → **0.2038**. Also tested and NOT adopted (no gain):
two-year O/D priors, Platt calibration, talent-blend/uncertainty-λ/pythag-exponent
retuning, MOV sample weighting. Bug fix: Air Force & Navy were silently dropped
(no 247 composite) — now kept via a low-percentile talent fallback.

**New pipeline (run in order):**

```bash
./venv/bin/python -m scripts.train             # retrain (now saves margin model too)
./venv/bin/python -m scripts.rank              # 2026 power ratings
./venv/bin/python -m scripts.simulate_playoff  # 20k-sim Monte Carlo -> CFP odds
./venv/bin/python -m scripts.prepare_logos     # map/download team logos
./venv/bin/python -m scripts.export_viz        # export data for the web app
./venv/bin/python -m scripts.export_site_data  # odds, poll + player imagery
python3 -m http.server 8642 -d viz             # open http://localhost:8642
```

**Playoff Monte Carlo (`scripts/simulate_playoff.py`):** simulates all 761
FBS-vs-FBS games on the real 2026 schedule + CCGs, then applies the confirmed
**2026-27 CFP format**: 12 teams; auto-bids for ACC/Big 12/Big Ten/SEC champs
(any ranking) + the highest-ranked Group of 6 team (champ or not); 7 at-large;
straight seeding, top-4 byes, first round at the higher seed, fixed bracket.
The committee ranking is proxied by
`10·win% + 0.24·rating_z + 0.71·SOS_z + 1.12·power_conf + 0.056·head_to_head`,
weights fit by `scripts/fit_committee.py` against every published committee ranking
from 2014-2025 (leave-one-season-out Spearman ρ = 0.913).

**Roster-weighted lens (July 2026):** a second rating variant that leans on the
2026 roster instead of 2025 results — 70% two-deep PFF talent (vs 50%) and full
§C uncertainty shrinkage (λ=1.0: low-continuity teams regress all the way to
their talent baseline). LOSO cost is known and small: Brier 0.2053 vs 0.2044,
same 67.7% accuracy. **No longer shipped or buildable** — the variant arguments it
was generated with are gone along with the rest of the multi-build machinery (see
"One build" above). The hyperparameters remain in `config.ROSTER_VARIANT` for anyone
who wants to reproduce it by hand.

**Web app (`viz/`):** four views — Top 25 power ratings, playoff projection
(most-likely bracket + full odds), a client-side matchup simulator (win prob /
spread / total / projected score for any two FBS teams, any venue; JS math
verified equal to the Python model to 4 decimals), and a **Ratings Lab** editor.
Logos come from `~/Downloads/cfb_alt_logos` (all 138 FBS teams; 101 gaps
auto-filled from the ESPN CDN via `scripts/prepare_logos.py`).

**Ratings Lab (editable inputs):** every team's six model inputs (Off, Def,
FieldPos, Pythag, Talent, Returning — the frozen model's per-team feature vector)
are editable in a table. The scoring math stays fixed; edits change only the
inputs, and power ratings, ranks, the matchup sim, and the playoff Monte Carlo
all recompute **client-side** from them. Power is a neutral round-robin over the
136 rated teams (matches the backend to rounding with no edits); the playoff
re-sim is a faithful JS port of `scripts/simulate_playoff.py` (verified: 4k-sim
JS output matches the 20k-sim Python baseline within RNG). Edits persist in
localStorage, per lens, with per-row and global reset. Requires
`viz/data/schedule.json` (emitted by `scripts.export_viz`).

**Roster editing (`scripts/serve.py`) — player-level depth-chart edits.** A small
stdlib-only local server (no Flask) that serves the same `viz/` app *and* a roster
API. Run it instead of `http.server`:

```bash
./venv/bin/python -m scripts.serve      # serves viz/ + API on :8642 (~20s to load)
```

It exposes each team's 2026 two-deep (Ourlads) with every player's 2025 PFF grade
at `GET /api/rosters`. In the Ratings Lab, a 📋 button on each team opens the depth
chart: edit a grade, promote/demote depth, change position, or add/remove players.
`POST /api/recompute` re-derives **talent[2026]** from the edited roster (the only
input a 2026 roster touches) and rebuilds the frame with the **frozen** model math,
returning new vectors; the client recomputes ratings/playoff from there. Only
talent depends on the roster, so opponent-adjusted O/D, the talent→O/D slopes,
Pythagorean and returning are loaded **once** at startup — each edit is
milliseconds. Verified: the no-edit baseline reproduces the exported vectors to
0.00000; dropping Ohio State's QB1 grade 93→60 cut talent 1.56→1.00, power
86.4→83.0, rank #4→#6, and playoff odds 76%→59%. Roster edits persist in
localStorage and re-apply via the API on load. Without the server (plain
`http.server`), the app degrades gracefully: the 📋 buttons hide and the six
direct inputs remain editable.

---

# Original README (v1) — still accurate below except where noted above

A clean, leakage-free, calibrated reimplementation of the **win-probability core**
of the methodology doc, fed by **fresh CollegeFootballData.com (CFBD) pulls**.
Built to improve on the original `CFB_Pred_Model` notebooks.

## What this fixes vs. the original notebooks

| # | Issue in original | Fix here |
|---|---|---|
| 1 | Feature selection ran on the **full dataset before the train/test split** (leaks test data) | Forward-looking design: season N games predicted from season **N-1** stats; held-out test season |
| 2 | Final logistic fit on **unstandardized** features (penalty applied unevenly) | All features z-scored within season before fitting |
| 3 | Calibration was **measured but never applied** | Isotonic calibrator is fit on train, **saved, and applied** at predict time |
| 4 | L1 used despite doc specifying **L2/Ridge** | L2 logistic, `C` chosen by cross-validation on Brier |
| 5 | Power rankings set **every team to home** | Round-robin on a **neutral field** (`is_home=0`) |

## Quickstart

```bash
cd ~/cfb-model
# .env already contains CFBD_API_KEY (gitignored)
./venv/bin/python -m scripts.train     # pull data, train (matchup model), evaluate, save
./venv/bin/python -m scripts.compare   # 1-season: RAW vs TALENT vs CALIBRATED strength
./venv/bin/python -m scripts.loso      # LOSO CV of the strength methods
./venv/bin/python -m scripts.loso2     # LOSO CV: per-stat calibrated vs matchup model
./venv/bin/python -m scripts.rank      # 2026 power ratings + example matchup
```

## Model architecture: true matchup-adjusted, team-level (default)

Each team collapses to **two ratings** per season — an offensive rating `O` and a
defensive rating `D` (means of the standardized offensive / defensive stats). A
game is modeled as a real matchup, offense against the opponent's defense:

```
off_edge = O_home(N-1) - D_away(N-1)      # home offense vs away defense
def_edge = D_home(N-1) - O_away(N-1)      # home defense vs away offense
+ pythag_diff(N-1), talent_diff(N), returning_diff(N), home-field
```

The L2 logistic fits these on game outcomes (**target B**), which both blends the
prior-year ratings with the talent/returning/Pythagorean priors (= regression to
a calibrated mean) and weights offense vs defense. Probabilities are isotonic-
calibrated. This replaces the earlier per-stat *like-vs-like* vector, which wasn't
modeling true matchups.

Learned weights (trained on 2021–24): `talent` is the largest single coefficient
(~0.50), but prior-performance signals collectively (`off_edge`+`def_edge`+
`pythag` ≈ 0.59) outweigh it — matching the diagnostic that performance predicts
next year better than talent.

### Per-team uncertainty index (doc §C, `scripts/loso3.py`)

Teams with little returning production (new QB / roster churn) get their prior-year
O/D ratings pulled toward their talent baseline: `u = 1 - returning_fraction`, and
`O_adj = (1 - λu)·O + λu·(b_o·talent_z)`. LOSO over 2021–25 picks **λ=1.0** (default):

| | Brier | Log-loss | Accuracy |
|---|---|---|---|
| Uncertainty OFF | 0.2120 | 0.6272 | 0.662 |
| **Uncertainty ON (λ=1.0)** | **0.2115** | **0.6207** | **0.667** |

Small but consistent (−1.0% log-loss, +0.5pp accuracy). It also makes the model
*trust the ratings more*: pre-shrinking moves weight onto the O/D edges (0.18→0.30,
0.13→0.19) and off the flat talent term (0.50→0.46).

### Does the matchup structure help? (LOSO, `scripts/loso2.py`)

| Model | Brier | Log-loss | Accuracy |
|---|---|---|---|
| Per-stat calibrated | 0.2131 | **0.6211** | 0.661 |
| **Matchup (default)** | **0.2120** | 0.6272 | 0.662 |

Marginally better Brier (wins 4/5 folds), tied accuracy, slightly worse log-loss
— **essentially a wash for binary win prob.** That's expected: linear off/def
edges nearly collapse to a strength differential. The structural payoff is
interpretability + it's the correct base for **margins/spreads (doc §5.2)** and
interaction terms, where matchup detail actually matters.

## Regression to a *calibrated* mean (doc §C, improved)

Raw prior-season stats over-rate G5 over-performers, so we regress each team's
stats toward a preseason expectation. Two ways were built and tested:

- **TALENT shrinkage:** `adjusted_z[f] = (1-λ)·prior_z[f] + λ·b[f]·talent_z`,
  where talent is the CFBD recruiting composite (leakage-free, known preseason).
- **CALIBRATED projection (default):** a per-stat OLS fit to predict the season's
  actual stat from four preseason signals — prior-year stat, talent, **returning
  production**, and **Pythagorean win expectation** (`PF^x/(PF^x+PA^x)`, luck-
  adjusted). We regress toward this fitted value. Why: talent alone explains only
  R²≈0.09 of next-year strength; the blend reaches R²≈0.31 — a ~3× more accurate
  mean.

### Why we trust the calibrated version: LOSO, not one season

A single held-out season can't separate methods that differ by ~0.2% Brier — on
2025 alone, TALENT actually edged CALIBRATED. Leave-one-season-out CV
(`scripts/loso.py`, coefficients re-fit per fold) averages over 2021–2025:

| Method | Brier | Log-loss | Accuracy | vs RAW |
|---|---|---|---|---|
| RAW (prior stats) | 0.2199 | 0.6360 | 0.646 | — |
| TALENT (λ=0.65) | 0.2146 | 0.6237 | 0.655 | −2.4% Brier |
| **CALIBRATED (default)** | **0.2131** | **0.6211** | **0.661** | **−3.1% Brier** |

CALIBRATED wins on all three metrics and beats TALENT in 3 of 5 folds (it loses
only 2021 and 2025 — and 2025 is exactly the season the one-shot test had used).

> **2026 inputs (checked 7-Aug-2026):** `/talent` still returns empty for 2026, so
> **talent still falls back to the 2025 composite** — `PROJECTION_TALENT_FALLBACK_YEAR`
> in `config.py`, and every run that uses it says so on stdout. This is the one
> placeholder left in the build.
>
> Returning production has two real 2026 sources in the repository. The model uses
> CFBD `percentPPA` from `data/returning_2026_cfbd.csv` because that is the definition
> used in every historical training season. The Bill Connelly / ESPN estimates and
> offense/defense splits remain in `data/returning_2026.csv` for descriptive use.
> The two total-returning measures correlate only about 0.30 in 2026; that is exactly
> why one cannot be substituted for the other after a coefficient has been trained.
> CFBD's PPA share is unbounded (roughly −0.57 to 1.00 in this snapshot), so it should
> not be displayed as though it were the Connelly percentage.
>
> Note: returning production is a roster/continuity metric known *preseason*, so 2026
> values exist now; performance stats and Pythagorean correctly use the completed 2025
> season.
>
> **The talent switch is not automatic**, despite what this note used to claim.
> `load_bundle()` only requests `TALENT_YEARS`, which ends at 2025, so 2026 can never
> enter the talent dict from the API however live the endpoint goes. The live path is
> a `data/talent_2026.csv`, which does not exist. When CFBD publishes, add 2026 to
> `TALENT_YEARS` — nothing will prompt you to.

## How it works

1. **Data** (`src/data/`): pulls `/stats/season/advanced` and `/games` from CFBD,
   cached as JSON in `data/raw/`. No synthetic fallback — fails loudly without data.
2. **Features** (`src/features.py`): z-score each season's advanced stats, flip
   defensive signs so larger = stronger. A game's feature vector is
   `home_prior_stats - away_prior_stats`.
3. **Model** (`src/model.py`): L2 logistic + a learned home-field term, then an
   isotonic calibration map. Saved to `artifacts/model.json`.
4. **Predict** (`src/predict.py`): single matchups and neutral-field power ratings.

## Honest results (calibrated model, LOSO over 2021–2025)

- Brier ≈ 0.213, log-loss ≈ 0.621, accuracy ≈ 0.66 — **out of sample** (LOSO mean).
- Lower than the original notebooks' reported numbers because those were
  in-sample / leakage-inflated. This is the real forward-looking baseline:
  a preseason projection (prior stats + talent + returning production +
  Pythagorean) with no in-season info or injuries. Vegas-level Brier is
  ~0.20–0.21 for comparison.

## Features used (CFBD `/stats/season/advanced`)

`off_ppa`, `off_pass_ppa` (EPA/dropback), `off_rush_ppa`, `off_success_rate`,
`off_explosiveness`, `off_pts_per_opp` (finishing), `def_ppa`, `def_success_rate`,
`def_explosiveness`, `def_havoc`.

**Not reproducible from CFBD free** (would need PFF / TruMedia): `PFFPressured%`,
explosive **rush/pass splits**, `RZTD%` exact, down-and-distance scoring rates.

## Margin / spread model (doc §5.2, `scripts/spreads.py`)

Predicts each side's points as offense-vs-opponent-defense
(`points ~ O_scorer + D_opponent + home`, Ridge), then derives **spread** and
**total**. A win prob falls out via a normal model on the margin
(`P = Φ(margin/σ)`), which we use only as a coherence check.

LOSO over 2021–25:

| | value |
|---|---|
| Margin MAE | **14.85 pts** (home-field baseline 15.95 → **−7%**) |
| Margin RMSE | 18.82 |
| Implied win-prob Brier | 0.2207 (vs dedicated logistic 0.2115) |

Real signal over baseline, but short of Vegas (~10.5 MAE) — expected for a
*preseason-only* projection (no in-season form, injuries, or final depth charts).
The implied win prob sits right next to the logistic, confirming the two models
are coherent; use the **logistic for win prob, the spread model for margins/totals.**
Example 2026 lines: Ohio St −15 / Notre Dame −16.4 vs an average team; neutral
Oregon by ~4.6 over Alabama (total ~50).

## Experiments & findings

- **EWMA recency weighting (`scripts/loso_ewma.py`)** — weighting recent games of
  the prior season more heavily does **not** help next-season prediction; flat
  equal-weight wins (best EWMA +0.43% Brier vs flat). Late-season "form" is a
  noisier sample; the full-season average is the more stable carry-over signal.
- **Offseason regression (`scripts/plot_regression.py`)** — preseason projection
  vs end-of-prior-season rating has **slope ≈ 0.79** (~21% pull to the mean),
  roughly linear. See `artifacts/regression_plot.png`.
- **Feature assembly (`scripts/feature_analysis.py`)** — explosiveness dropped
  (~0 predictive signal); 0.7 collinearity prune keeping the better predictor of
  each cluster; distinct high-signal stats added. Final 6 features (3 O / 3 D, all
  pairwise |r|<0.7): `off_success_rate`, `off_rush_ppa`, `off_havoc`, `def_ppa`,
  `def_line_yds`, `def_havoc`. Net effect: same accuracy as the old 10-feature set
  with half the features (a parsimony win, not an accuracy win); the uncertainty
  index went near-neutral (λ 1.0 → 0.25).

## Model diagnostics (scripts/diagnostics.py)
Validity checks on the final model (out-of-fold where it matters):
- **Collinearity:** all VIF < 3 (clean). **Significance:** all features p<0.05 except
  pythag (p=0.06, kept — helps LOSO). **Overfit:** in-sample vs OOF Brier gap only
  +0.004. **Autocorrelation:** Durbin-Watson 2.04 (none over time).
- **Calibration:** isotonic is now fit on **cross-validated (out-of-fold)**
  predictions (not in-sample), fixing the over-confidence (OOF slope 0.87 → 0.89,
  Brier flat). The in-sample-calibration overfit is gone.
- **Remaining (minor):** spread model heteroscedastic (expected for football —
  point spreads fine, only intervals/totals need a variance model); mild team
  residual clustering (the static-preseason limitation the in-season Elo layer
  would address).

### Mean-regression formulas (scripts/loso_meanreg.py)
Holding average shrinkage constant, the *shape* (global / returning / variance /
rating-dependent / combined) is within ~0.001 Brier of each other and of no
shrinkage — the L2 blend already does regression-to-mean implicitly. Rating-
dependent is worst (offseason regression is linear). No explicit formula adopted.

## Roadmap (next improvements, in priority order)

1. ~~Regression-to-mean + talent prior (doc §C)~~ ✅ **done**.
2. ~~Calibrated mean (talent + returning production + Pythagorean)~~ ✅ **done**.
3. ~~True matchup-adjusted, team-level model (offense vs opponent defense)~~ ✅ **done**.
4. ~~Per-team uncertainty index (doc §C)~~ ✅ **done** (λ=1.0).
5. ~~Margin / spread + totals model (doc §5.2)~~ ✅ **done**.
6. ~~TruMedia feature expansion (RZ TD%, PFF pressure, field position)~~ ✅ **done**.
7. ~~Opponent / strength-of-schedule adjustment (doc §4.4)~~ ✅ **done** (SRS, α=1.0).
8. ~~Mean-regression formulas (doc §C)~~ ✅ tested — no form helps (kept implicit).
9. ~~In-season Elo layer core (doc §5.3)~~ ✅ **done** (−9% Brier, MOV updates).
### Elo parameter tuning (scripts/tune_elo.py)
- **K-factor: decaying 50→30 over the season is best** (just edges constant K=40).
  The by-week split confirms the intuition: high K helps EARLY (wk1-4: K50 0.1750 vs
  K30 0.1760), low K helps LATE (wk10+: K35 0.1899 vs K50 0.1914 — high K hurts late).
  Net pooled gain is small because the early/late effects partly offset and the MOV
  multiplier already adapts updates.
- **Home field: team-specific** beats constant — `HFA = 65 + 25·z(home-minus-away
  margin)`, shrunk. Best combo (K decay + team-HFA) → ~0.1855.
- No market blend by design (the goal is a market-distinct signal for betting edge).

## Roadmap (next improvements, in priority order)

10. **Elo refinements (doc §5.3):** PGWE (continuous postgame) [deferred by request];
    productionize live Elo + Elo→spread into rank/spreads. No market blend (by design).

### LOSO Brier progression (preseason model)
0.2115 (orig CFBD) → 0.2112 (+TruMedia) → 0.2093 (+opponent adj) →
**0.2048 (+roster-aware talent blend)**; log-loss → ~0.60.

### In-season Elo layer (scripts/backtest_elo.py) — the biggest jump
Seeding Elo from the preseason model and updating after each game (MOV multiplier)
cuts Brier **~9% overall** vs the static model, robust across 2022-25 (−5.9 to
−12.2% per season), growing through the year (wk1-4 0.180→0.175, wk5-9 0.212→0.190,
wk10+ 0.217→0.191). No blend needed — Elo starts at the preseason rating and only
improves. Fixes the static-preseason team-clustering the diagnostics flagged.

### Talent signal — roster-aware PFF + CFBD blend (DEFAULT)
- Historical *team-level* PFSN talent ≈ flat/worse than CFBD (redundant with prior
  performance already in the model).
- **Roster-aware PFF talent** (season-N roster × N-1 grades, transfer-aware) with
  position weights **re-optimized on PFF grades** (NNLS vs win%; far more balanced
  than PFSN's QB/CB-heavy weights), **blended 50/50 with CFBD**, beats CFBD on all
  metrics (LOSO 2022-25): Brier 0.2096→**0.2048**, log-loss 0.6123→**0.5998**, acc
  0.664→**0.676**. Roster turnover is the value-add; CFBD stays as the orthogonal
  recruiting anchor. 2026 uses the Ourlads two-deep × 2025 PFF grades.
- **Signal map (scripts/compare_signals.py):** three independent axes — prior
  performance (pythag≈prior_strength r=0.92), recruiting (CFBD, orthogonal),
  continuity (returning, ~0 corr with all). PFF_roster (r=0.46 vs win%) is the
  strongest talent signal and only 0.36-correlated with CFBD → complementary.
- **Interactions / nonlinearity:** screened all pairwise interactions + quadratic +
  tempo×efficiency → none materially useful; model is additive/linear. Logs N/A
  (features are standardized z-scores).
- **pythag vs O/D:** the 0.92 redundancy is with raw prior_strength, NOT a model
  feature; dropping pythag *hurt* LOSO (0.2093→0.2098), so it's kept.
