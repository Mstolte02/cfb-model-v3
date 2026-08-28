# Unit against unit: WAR split by facet and position, and pitted at the matchup

## The question

The model carries one WAR number per team. WAR is *built* from 87 facets and eleven
position rooms, and the roster reports show that breakdown, but the game model never
sees it — and it never compares a team's receivers against the corners those receivers
will actually face. Two things were worth testing, in order:

- Does breaking WAR out by facet and position predict games better than the total?
- Does pitting each unit against the opposing unit predict better than either?

**Neither does.** The split is worse, and the pairing is worse than the split. Nothing
here ships. The evidence and the reason are below, and the reason is more useful than
the verdict.

## What was built

Two feature bases, kept apart because they obey different contracts.

**Realized, lagged, facet × position — 25 units.** `hybrid_player_war_by_facet.csv`
splits every player-season's WAR across 87 facets, and the split is exact: the facet
columns sum to `war` to machine precision. Each facet already carries a predeclared
football concept in `concepts.json` / `consolidated_facets.json`, and each player
carries a position, so (concept, position group) is a well-defined cell that costs no
new judgement. The 25 units sum back to team WAR exactly — asserted in
`tests/test_facet_matchup.py`, not assumed. Season N reads completed N-1 totals only,
the same discipline as `war_lag`.

**Preseason, projected, position room — 11 rooms.** `war_model/preseason_group_war.py`
stops the leakage-safe roster projection one level earlier than the shipping team
total, so season N gets a projected WAR per room from N-1 and earlier. Room sums
reconcile with `preseason_team_war.csv` to 1.3e-15. This is the base the production
selector actually rewards (`war_projected`).

**The matchup term** is the odd contrast `V4.MATCHUP_PAIRS` already uses:

```
edge_ha * |edge_ha| - edge_ah * |edge_ah|
```

zero when two rooms are level, faster than linear in a real mismatch, and exactly
negated when the teams swap. That last property is not decoration — the v4 win
probability is a complement and its margin an opposite only because every feature
negates, so a matchup term that survives a swap is not admissible at all. Every one of
the 248 columns is asserted antisymmetric at runtime and in the test suite.

A *linear* cross term was not tested, because there is nothing to test:
`b(O_h − D_a) + b(D_h − O_a)` expands to `b(O_h − O_a) + b(D_h − D_a)`. A linear
"matchup" model is the like-for-like model with worse notation, and with separate
coefficients the like-for-like form is strictly more general
(`RATING_ARCHITECTURE_EXPERIMENTS.md`). Only the nonlinearity is new, and only the
nonlinearity is on trial.

## Result

Strict expanding replay, test seasons 2022–25, every family fit on the same games and
scored on the same games, against the shipping five columns
(`O, D, talent, returning, war_projected`). One weekly-update setting for all
families, so the online column compares families and not their tuning budgets.

The realized unit split needs season N−1 and the WAR build has no 2020, so every unit
column is zero in 2021 and the unit families are byte-identical to the reference in the
2022 fold. Including that fold does not punish them — it averages in a tie. The window
where every family is live is below; the four-season numbers are in the artifact and
tell the same story slightly more gently.

| family, 2023–25, n = 2,189 | static Brier | Δ vs shipping | 95% paired CI | online Brier |
|---|---:|---:|---|---:|
| **shipping five columns** | **.20720** | — | — | **.18895** |
| rooms only, no total | .20769 | +.00049 | [−.00096, +.00191] | .18954 |
| rooms + total | .20777 | +.00057 | [−.00091, +.00202] | .18956 |
| **room matchup crosses** | .20818 | +.00098 | [−.00028, +.00223] | .18948 |
| room crosses, no total | .20852 | +.00132 | [−.00012, +.00272] | .18978 |
| rooms + room crosses | .20948 | +.00228 | [+.00054, +.00406] | .19112 |
| **unit matchup crosses** | .20970 | +.00250 | [+.00054, +.00449] | .19020 |
| both cross families | .21032 | +.00312 | [+.00069, +.00561] | .19058 |
| all 30 room crosses | .21039 | +.00319 | [+.00136, +.00511] | .19113 |
| clean core (no WAR at all) | .21134 | +.00414 | [+.00260, +.00580] | .19171 |
| units only, 25 columns | .21375 | +.00655 | [+.00310, +.00993] | .19360 |
| units + unit crosses | .21395 | +.00675 | [+.00303, +.01043] | .19288 |
| units, no total | .21819 | +.01099 | [+.00723, +.01476] | .19663 |
| all 156 unit crosses | .22229 | +.01509 | [+.00963, +.02079] | .19391 |

Every point estimate is positive. Not one family is better than the number it was
meant to improve on, on either metric, in either window. Families clearing the 0.001
adoption bar against the shipping features in two consecutive selection windows:
**none**.

Three readings worth naming:

**The position split of projected WAR carries nothing beyond its own sum.** Eleven
rooms that add up to `war_projected` score +.00049 against `war_projected` alone. The
model is not missing anything by seeing only the total.

**The facet × position split is actively harmful.** 25 realized unit columns lose
+.0066, and dropping the total so the units have to carry it alone loses +.0110 — worse
than having no player information at all (clean core, +.0041). Twenty-five noisy
columns on 1,500–2,200 training games is not a decomposition, it is a variance leak.

**The pairing costs more than the split it is built on.** Unit crosses (+.0025) beat
unit linear (+.0066) only because there are 13 of them instead of 25; against the
shipping features both lose. Adding crosses on top of the linear split is worse than
either alone in both bases — the double-count result from
`RATING_ARCHITECTURE_EXPERIMENTS.md`, again.

## Which facets actually interact

Every offence unit crossed against every defence unit — 156 realized pairs, 30 room
pairs — each scored alone on top of the shipping features. This is in-sample
description, printed to be read; nothing downstream selects from it.

**At room level the football theory is visible.** WR against CB ranks 2nd of 30 and
QB against SAF 5th, both with the sign football predicts.

| room pair | rank of 30 | in-sample Brier gain | coefficient |
|---|---:|---:|---:|
| WR vs CB | 2 | .00061 | +.0438 |
| QB vs SAF | 5 | .00036 | −.0275 |
| OT vs EDGE | 7 | .00035 | −.0079 |
| TE vs LB | 12 | .00021 | +.0281 |
| RB vs DT | 16 | .00016 | +.0183 |
| IOL vs DT | 19 | .00009 | −.0053 |

The largest gain in the whole table is .00063 with one free parameter on 2,913 games,
which is noise-scale, and out of sample all of it disappears. The line pairs are also
confounded by the roster-coverage defect recorded below.

**At facet × position level the theory is invisible.** The predeclared football pairs
rank 18, 22, 30, 35, 48, 51, 69, 83, 112, 115, 120, 127 and 153 out of 156. The two
most physical matchups in the sport pay nothing at all: the back against the interior
front ranks 115 with a gain of .00000, interior protection against interior rush ranks
120 with a coefficient of −0.0000, and tackles run-blocking against edge-setting ranks
153 with a *negative* gain.

What tops the scan instead is not football. Six of the top eleven pairs are some
offensive unit against `d_prsh_2nd` — pass rush by linebackers and defensive backs —
and every one of them carries a **negative** coefficient, meaning a larger offensive
edge lowers the home team's win probability. That is one noisy column absorbing
variance through whatever it is multiplied by, not an interaction. A scan that ranks
"receiving backs vs blitzing safeties" above "guards vs three-techniques" is measuring
its own noise.

## Why it fails, and this is the part worth keeping

The units are not redundant. Mean absolute correlation between the 25 unit columns is
0.172 and the first principal component explains only 21% of their variance, so the
split really does carry independent information about a team.

The independent part just does not survive the off-season.

| | year-over-year r |
|---|---:|
| **team WAR total** | **.693** |
| interior run blocking | .659 |
| tackle run blocking | .515 |
| QB passing | .514 |
| havoc | .507 |
| receiving, WR | .501 |
| *… 14 units, including corner coverage at .301* | *.301–.465* |
| pass rush, second level | .291 |
| offensive penalties | .277 |
| run defence, secondary | .265 |
| coverage, linebackers | .245 |
| coverage, safeties | .231 |
| ball security | .187 |

**The total is more stable than every single one of its parts.** Aggregation is doing
real work: it averages 25 noisy measurements into one that persists at .69, and the
model was already using the good version. Splitting it hands the regression 25 columns
each measuring next season worse than the number it replaced.

For matchups specifically it is worse than that, because a cross term multiplies two
of these together. The marquee pairs are exactly the ones with an unstable defensive
half — WR (.501) against CB coverage (.301), QB (.514) against safety coverage (.231).
The product of a signal and a coin flip is a coin flip. And the one place the split is
genuinely stable, interior run blocking at .659, has no correspondingly stable opponent
to be pitted against — interior run defence sits at .465 and linebacker run defence at
.309.

This is a measurement result, not a football result. It does not say that guards do not
matter against three-techniques. It says that a season of graded snaps measures that
particular room-against-room quality too noisily, one year ahead, for the comparison to
survive contact with a held-out season.

## Found on the way: the historical roster projection has no offensive line

Not part of the experiment, and it should be fixed regardless.

`project_2026_v2.CFBD_TO_GROUP` maps `OT`, `G`, `C` and `OG`, but CFBD's historical
roster feed overwhelmingly lists linemen as **`OL`** — 2,606 of them in 2024 against 44
`OT`, 32 `G` and 18 `C`. Anyone whose position is not in the map is dropped by
`load_rosters`, so the historical preseason projection never sees them. The same is
true of `DL` (2,076 rows) collapsing the entire front into `DT`, and `DB` (2,216)
crowding out `SAF`.

Teams with a populated room in the historical projection:

| season | OT | IOL | EDGE | SAF | QB / RB / WR / TE / LB / CB |
|---|---:|---:|---:|---:|---:|
| 2022 | 20 | 29 | 112 | 127 | 137 |
| 2023 | 15 | 19 | 104 | 127 | 137 |
| 2024 | 13 | 15 | 102 | 116 | 137 |
| 2025 | 13 | 7 | 75 | 82 | 136–137 |

The two-deep allots 4 OT and 6 IOL slots of 44. So `preseason_team_war.csv` — and
therefore `war_projected`, a feature the production selector chose and ships — is a sum
over roughly 8.9 of 11 rooms for a typical team-season, with the offensive line
systematically absent. It is not a leak and it is not fabricated; it is a silently
narrower feature than its name implies, and the 2026 serving path does not share the
defect because it reads the two-deep, which labels every slot properly.

That mismatch between the training population and the serving population is the same
shape as the `is_starter` and `is_transfer` bugs in v3.9/v3.10. Fixing it means adding
coarse `OL`/`DL` rooms to the historical map, or resolving CFBD line positions against
the PFF history that does distinguish T/G/C and DI/ED. Either changes a shipping
feature and needs its own replay, so it is recorded here rather than done here.

## What this does not settle

The verdict is about **season-ahead game prediction from preseason and prior-season
inputs**. Three things could change it and none is tested here:

1. **In-season units.** Every unit here is at least a full off-season stale. A pass
   protection number from three weeks ago is a different measurement from one from last
   November, and the weekly update path is where it would belong.
2. **A longer window.** The team stats reach back only to 2020, so the whole replay is
   four test seasons and the realized-unit families get three. The facet WAR itself
   covers 2014–25. A wider stat pull would roughly triple the power.
3. **Availability.** Pitting a room against a room is most interesting when someone is
   hurt, and the model has no historical injury archive to make that a preseason fact.

The unit and room tables are also worth keeping as **reporting** even though they lose
as features. "Third-best interior run blocking in the conference against the worst
interior front" is a true and useful sentence about a matchup. It is just not a better
prediction than the rating already is.

## Reproduction

```powershell
python war_model/preseason_group_war.py      # room-level projections, once
python -m scripts.facet_matchup_backtest
python -m unittest tests.test_facet_matchup -v
```

Artifacts: `artifacts/facet_matchup_backtest.json`,
`artifacts/facet_matchup_backtest_predictions.csv`,
`artifacts/facet_matchup_interactions.csv`. Unit taxonomy and predeclared pairs live in
`src/facet_matchup.py` and are serialized into the result JSON, so a later reader can
see exactly which cells and which pairings were tested.
