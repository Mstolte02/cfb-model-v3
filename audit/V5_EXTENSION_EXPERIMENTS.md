# Extending v5: pace, game control, PFF scheme data, confidence-weighted updates

Started 22 September 2026, the day v5 went live. Every test here changes one thing
inside the live ensemble and scores it against v5 itself, game by game.

## The harness

`scripts/v5_extension_backtest.py`. The full `prior_decay_backtest` spends most of its
40 minutes choosing each member's model knobs, score step, form half-life and stack
penalty. Those were chosen inside each outer fold's training pool, so reusing them is
leak-free; the harness refits the members' preseason models, replays the seasons, and
changes one thing. A complete four-member replay of 2023–25 takes about 8 seconds.

**Parity first.** With `--backtest-scaling` the harness reproduces the published
backtest exactly: .177832 pooled, .172646 / .182796 / .177975 by season. Without it
the baseline is v5 as served (production WAR scaling): **.177837**. Everything below
is compared with that baseline, on the same 2,189 games of 2023–25, with the
season-week block bootstrap. A negative difference is better.

## 1. Pace, and 2. game control

Two style dimensions, both from cached CFBD data (`src/style.py`), start-of-week so a
slate never sees itself:

- **pace**: game-clock seconds per offensive play, from the drive summaries;
- **pass environment**: each offense's neutral-situation pass rate (downs 1–2,
  quarters 1–3, score within 14), from the play-by-play, with no fitted expectation.

Mark's two-part idea is built directly, per dimension:

- **what style a team wins in**: a shrunk slope of its margin *versus expectation* on
  the game's realized style. Expectation is CFBD's pregame Elo, so the residual is
  independent of this repository's models;
- **whether it can force that style**: how far the realized game moved toward its own
  preference rather than the opponent's, on [−1, 1];
- combined into an **expected style** for the upcoming game (each side's preference
  weighted by its control) and a **style edge** in points.

Pace was also tested as a **variance effect**: a longer game should be more
predictable, so the rating terms were interacted with the game's expected pace. And the
earlier drive-based families (`src/tempo.py`) were re-tested inside v5, because they
were last scored against v4.

| Variant | Brier | vs v5 | 95% interval | seasons better |
|---|---:|---:|:---|---:|
| v5 baseline | .177837 | — | — | — |
| pace as variance | .177931 | +.000093 | [−.000321, +.000514] | 2/3 |
| style: pace (preference, fit, control) | .178178 | +.000341 | [−.000218, +.000919] | 0/3 |
| style: pass environment | .178090 | +.000252 | [−.000716, +.001229] | 1/3 |
| style edge alone | .178282 | +.000444 | [−.000287, +.001196] | 1/3 |
| style: all six | .178340 | +.000503 | [−.000708, +.001717] | 1/3 |
| tempo identity | .178647 | +.000810 | [+.000225, +.001404] | 0/3 |
| tempo scripted windows | .178254 | +.000417 | [−.000318, +.001158] | 1/3 |
| tempo state/control | .178840 | +.001002 | [+.000404, +.001610] | 0/3 |
| tempo matchup/control | .178248 | +.000411 | [−.000088, +.000938] | 1/3 |

The 2023 fold's stack trains on one season, so the added columns were also given
extra shrinkage (penalty ×11 and ×100 on those columns only). That only walks them
back toward the baseline — style-all goes +.000247 then +.000004 — and never past it.

**Reading.** None of it adds information v5 lacks, and two families are worse with
intervals that exclude zero. This matches the September result against v4 and now has
a clearer reason: v5 already carries season-to-date opponent-adjusted efficiency beside
the rating walk, and a team's pace and style are largely a restatement of how good it
is and how its games have gone. Style *fit* in particular is estimated from twelve to
sixteen games a team, which is too few to separate "wins fast games" from noise. Not
shipped. This is not evidence that tempo is irrelevant to football; it is evidence
that these pregame summaries do not add to this model.

## 4. Confidence-weighted updates (team level)

Inside every member the constant score step is replaced by a per-team extended Kalman
gain (the September filter, `scripts/inseason_kalman_backtest.py`, ported into the v5
walk). Mark's three signals map onto its parameters:

- **prior confidence**: the week-0 variance is scaled by `exp(beta · doubt)`, where
  doubt is the within-season z-average of low returning production, portal volume in
  plus out, and a first-year head coach — all known before week 1;
- **sample size**: each game shrinks the team's variance, so the gain falls as a team
  is measured;
- **track record**: `gamma` inflates the variance after results the filter found
  surprising.

`v0`, weekly drift `tau`, `beta` and `gamma` are chosen per member on the fold's
earlier seasons only (their out-of-sample preseason models), 36 combinations.

| Variant | Brier | vs v5 | 95% interval | by season (2023 / 2024 / 2025) |
|---|---:|---:|:---|:---|
| per-team gain | .177611 | −.000226 | [−.000859, +.000346] | +.000005 / −.000187 / −.000500 |

The selection set `beta = 0` and `gamma = 0` in every member of every fold, so the
ablations without prior doubt and without track record are identical to the full
variant. **The whole gain is the sample-size part.** Why the other two did not earn a
place is measurable directly: across 396 team-seasons, the size of a team's average
miss against its week-0 rating correlates +.01 with returning production, +.07 with
portal volume and +.05 with a coaching change (first-year coaches miss by 7.3 points
against 6.6). The signals point the right way and are too weak to set a gain.

**Candidate, not yet shipped.** It is better in two seasons and level in the third,
with a 78% probability of improvement. Under this repository's policy that is a
genuine but small gain; it changes the live walk, so it is Mark's call and would ship
behind a week boundary like every other live change.

Player-level confidence (in-season WAR updates weighted by snaps and prior) is not
tested here: it needs weekly player data, which is the PFF work below.

## 3. PFF team tables, scheme matchups, and personnel

### Data staged (outside Git, `source-data/pff_api/`)

- `team_stats_weekly/`: all seven PFF team-stat categories for 2021–25, **season to
  date through each week** (490 tables), via `/v2/ncaa/teams/stats?weekIds=0,...,N`
  (`python -m scripts.sync_pff_api --team-stats-weekly`). PFF week 0 is early games
  CFBD files under week 1; PFF week N is CFBD week N otherwise, so a week-W game reads
  `thru_w{W-1}` and sees only earlier games. Conference championships are PFF week 17,
  so for 2021–23 `thru_w14` includes them; nothing reads that file before them.
- `player_weekly/`: single-week `offense` and `defense` position reports for 2021–25
  (150 files, every FBS player's snaps per week), via
  `python -m scripts.sync_pff_weekly_players`. The offense report's snap column is
  `snap_counts_total`.
- PFF's budget is 100 requests a minute, with no daily cap seen.

### Results

`src/pff_form.py` reads the weekly tables. Each cross-section is standardized over the
FBS teams in the model's universe (PFF also charts FCS schools, which compress the
FBS spread if they are left in).

| Variant | Brier | vs v5 | 95% interval | seasons better |
|---|---:|---:|:---|---:|
| v5 baseline | .177837 | — | — | — |
| **PFF outcome composite** (one offence, one defence number) | **.176612** | **−.001225** | [−.002573, +.000208] | **3/3** |
| PFF outcome, four columns (EPA and success, for and against) | .177003 | −.000835 | [−.002312, +.000685] | 2/3 |
| PFF process (pressure, sacks, contact yards, tackling) | .178020 | +.000183 | [−.000897, +.001173] | 1/3 |
| PFF scheme clash (protection v rush, blocking v run defence) | .177946 | +.000109 | [−.000518, +.000800] | 2/3 |
| composite + per-team gain | .176540 | −.001298 | [−.002703, +.000176] | 3/3 |

The composite is the mean of the EPA-per-play and success-rate z-scores for each side.
The four-column version is worse because the pair correlates .82–.89 and the stack
spends its weight trading between near-duplicates. The defensive sign was checked
against CFBD's defence rating (correlation −.80, as it should be).

**Where it helps:** weeks 2–4 −.0039, weeks 5–9 −.0010, weeks 10+ −.0002, week 1
nothing (there is no table yet). By season: 2023 −.000035, 2024 −.001064, 2025
−.002598. 2023 is effectively level; its stack is trained on 2022 alone, the thinnest
fold. The gain grows as the stack sees more seasons, which is the direction a real
signal should move.

**Two checks that it is real information, not an artifact:**

1. *It is not the one-game threshold.* v5 uses CFBD form only once both teams have two
   games; the PFF tables are used after one. The control — CFBD form after one game —
   gains only −.000157. PFF's early-season measurement is simply better than CFBD's.
2. *It is not a leak.* A deliberate placebo that reads each week's table *including*
   that week's own games (`pff_LEAK_PLACEBO`) scores .151195, a −.0266 gain, twenty
   times the honest result. The honest version is nowhere near what a leak looks like.

**Live as v5.1 (22 September 2026).** Each member keeps its v5 base stack, byte for
byte, and gains `stack_pff`: the same columns plus `pff_O_diff` and `pff_D_diff`,
fitted on the same cross-fitted designs with the same penalty. A slate uses the PFF
stack when PFF's table through the previous week exists (week 1 reads an empty table,
as the backtest did) and the base stack otherwise, so a missing pull degrades to v5
rather than to a stack fed zeros it never saw. In the production fit PFF's offence
composite takes a weight of about .36 and CFBD's offensive form drops to about −.04;
PFF's defence composite takes little.

- **Data**: `data/live/pff_form_2026.json`, {cutoff week: {team: [O, D]}}, written by
  `publish_pff_form` in the capture with only the two overall-success categories. A
  cutoff is re-pulled on every capture until the next week's first kickoff, then frozen,
  so late charting lands before that week is priced and a table never changes after
  its games start. It needs a `PFF_API_KEY` repository secret; without one the capture
  keeps the committed table.
- **Checks**: `train_live_ensemble --check 2025` replays 2025 through the published
  block with the CI's own stdlib composite built from the staged tables and matches the
  harness to 6.0e-12 on all 719 games (Brier .175320 both). The stdlib composite
  matches `src/pff_form.load_through` on real 2024 tables (test). On the rendered site
  every market-board week, Pick'Em export, Tracking view and futures panel is identical
  before and after, and the matchup lab equals the Python runtime.
- **Payload** is schema 6, so a cached schema-5 page refuses it instead of silently
  ignoring the PFF stacks.

Still open: whether PFF has charted Saturday's games by the Monday 12:30 PM ET lock.
`fetched_at` in the PFF table records when each cutoff was last pulled; check it
against the lock on the first live Monday.

### Personnel: who played last week

`src/availability.py`. A **regular** is a player with at least 60% of his side's snaps
over two or more earlier games; he is **missing** if he took under 10% in the team's
most recent game. Each missing regular counts his season N−1 WAR. The signal is
genuine — 2022 flags Bryce Young before Alabama's week 7, Will Levis before Kentucky's
week 7, Taulia Tagovailoa before Maryland's week 10 — and it is also where its limit
shows: Young played in week 7.

| Variant | Brier | vs v5 | 95% interval | seasons better |
|---|---:|---:|:---|---:|
| missing WAR | .177865 | +.000028 | [−.000249, +.000294] | 1/3 |
| missing quarterback WAR | .177945 | +.000108 | [−.000051, +.000264] | 1/3 |
| all three | .177722 | −.000116 | [−.001058, +.000833] | 1/3 |

Measured where it should matter — the 142 games with a missing-quarterback gap above
.3 WAR — it makes things worse (+.0020 and +.0075). Absent last week is a poor proxy
for absent this week, and the rating walk has already absorbed the game he missed. A
real availability channel needs injury reports. PFF's roster endpoint carries injury
status, but only for today, so it cannot be backtested; it could be recorded from now
on and tested next season.

### Not pursued, and why

- **Alignment versatility** needs a per-player pivot request (`/v1/player/position/pivot`
  takes one player id), thousands of requests for a signal the unit-level clash
  results give no reason to expect.
- **Run-direction fit** needs a per-team report per season (about 700 requests) for a
  matchup family of the kind that just failed twice (scheme clash here, quick-pass v
  pressure in September).
- **Coaching tendencies residualized against talent, opponent and state** overlap the
  pass-environment style test above and the rejected decision-profile experiments in
  `audit/DECISION_PROFILE_EXPERIMENTS.md`.

## Player-level confidence (idea 4, second half)

Not built. The WAR projection already shrinks each player by his own reliability
(`war_model/uncertainty.py`), and the in-season team update absorbs results directly.
An in-season *player* update would need weekly player grades turned into weekly WAR,
which the weekly reports now make possible; the September result
(`audit/BAYESIAN_UPDATING_RESEARCH.md`) says where it would pay — receivers shrunk
toward their own prior, quarterbacks early only.

## Reproduction

```powershell
python -m scripts.v5_extension_backtest --only base --backtest-scaling   # parity
python -m scripts.v5_extension_backtest                                  # everything
```

Results: `artifacts/v5_extension_backtest.json`, per-game predictions in
`artifacts/v5_extension_predictions.csv`.
