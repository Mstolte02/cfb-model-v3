# Phase 7 — nonlinear talent

## Origin

This candidate was not proposed. It appeared as a **control** in
`DECISION_PROFILE_EXPERIMENTS.md`: PROE correlates +.338 with talent, so `proe×talent`
is partly quadratic talent, and a curvature control was needed to tell a decision
effect from a nonlinearity the four-column core could not express. The control beat
every decision candidate and the core itself.

A post-hoc winner from someone else's experiment is not a finding. This phase tests it
properly.

## The objection this phase had to answer

`clean_core` is four columns. Production chooses among ten candidates, several
carrying granular per-stat inputs, lagged player talent and matchup interactions —
all already nonlinear in roster quality. If curvature only rescues a stunted baseline,
the finding is "the four-column core is underspecified", which the production selector
routes around by not picking it.

## Predeclared rule, fixed before running

Every arm is compared to **its own parent family**, never to clean core. Curvature is
adopted only if it improves the family the production selector actually picks, by at
least .001, with a 95% paired bootstrap interval excluding zero, **on both the static
and online metrics**. Beating clean core alone is explicitly insufficient.

Eight parents × two arms × two metrics = 32 comparisons, reported rather than hidden.

## Result

Every arm against its own parent, pooled 2022–25. Negative means curvature helps.

| arm | static Δ | static 95% CI | online Δ | online 95% CI |
|---|---:|---|---:|---|
| clean_core + curvature | −.00449 | [−.00850, −.00086] | −.00153 | [−.00407, +.00091] |
| core_pff_lag + curvature | −.00465 | [−.00873, −.00096] | −.00138 | [−.00402, +.00108] |
| core_war_lag + curvature | −.00442 | [−.00845, −.00074] | −.00148 | [−.00400, +.00094] |
| **core_war_projected + curvature** | **−.00382** | **[−.00759, −.00046]** | −.00093 | [−.00321, +.00122] |
| core_players_lag + curvature | −.00424 | [−.00824, −.00063] | −.00151 | [−.00406, +.00092] |
| core_matchups + curvature | −.00494 | [−.00918, −.00115] | −.00152 | [−.00424, +.00109] |
| granular_clean + curvature | −.00480 | [−.00907, −.00099] | −.00180 | [−.00468, +.00091] |
| granular_players + curvature | −.00494 | [−.00930, −.00102] | −.00157 | [−.00438, +.00110] |

`talent_sq` alone recovers most of it in every family (−.00327 to −.00434), so the
effect is specifically **talent is nonlinear**, not "some second-order term helps".

With curvature on the menu the selector picks a curvature variant in three of four
folds (2023, 2024, 2025).

### The objection is answered

Curvature does **not** only rescue the stunted core. It helps all eight families by a
similar amount, including the granular and matchup families that were the strongest
candidates for already capturing it. Effect sizes cluster in [−.0033, −.0049] with no
scatter — this is one finding replicated eight times, not eight chances at noise, and
Bonferroni is the wrong lens for eight nested restatements of a single hypothesis.

## Adoption verdict

**Not adopted.** The predeclared rule requires both metrics, and the online metric
fails on both production-selected families: `core_war_projected` at −.00093 does not
even reach the .001 bar in magnitude, and its interval includes zero.

## Why static and online disagree — and why it matters elsewhere

The split is not noise, it is mechanism. The static metric scores the model's
**preseason prior**; the online metric scores it after the dynamic in-season update
has absorbed results. Curvature is a correction to how preseason talent maps to
strength. Once a team has played games, the update overwrites the prior and the
nonlinearity stops mattering.

This is the same fact `MODEL_VS_MARKET_DIAGNOSIS.md` finds from the other direction:
the model's deficit against the market is +.0202 Brier in weeks 1–3 and +.0002 by
week 13. **The preseason prior is where this model is weak**, and curvature improves
exactly that.

### Consequence for the preseason products

The futures board, projected win totals, playoff odds and the season forecast table
are all static-prior products — they are published before any games are played, where
the dynamic update contributes nothing. Curvature's measured gain lives entirely in
that regime.

That is a real candidate for those surfaces even though it fails the weekly-game rule,
but it has **not been tested as such**. Adopting it there requires a phase 8 scoring
the preseason products on their own targets — projected wins against realised wins,
playoff probability against realised fields — rather than reusing a game-level Brier
that the weekly rule was written for.

## Reproduction

```powershell
python -m scripts.curvature_backtest
```

Artifacts: `artifacts/curvature_backtest.json` and `.csv`.
