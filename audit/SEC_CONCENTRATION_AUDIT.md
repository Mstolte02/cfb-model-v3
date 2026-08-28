# Why the SEC holds 13 of the top 25

Checked 28-Aug-2026, after the 2026 preseason board came back with 13 SEC teams in
the top 25 and all 16 SEC teams above the national mean.

> **Follow-up, same day.** The feature-design problem this audit found in §"The
> structural problem" was investigated in
> [`STANDARDISATION_AND_COLLINEARITY.md`](STANDARDISATION_AND_COLLINEARITY.md) and
> fixed: the model now ships seven features instead of fifteen. Every decomposition
> below was computed against the fifteen-feature build and is kept as the record of
> what was found. **The conclusion is unchanged under the new model** — the top 25 is
> still SEC 13, Big Ten 6, Big 12 3, ACC 2, Independent 1. LSU corrected from No. 5 to
> No. 8, as §"One genuine defect found" predicted.

This follows [`VANDERBILT_SEC_SANITY_CHECK.md`](VANDERBILT_SEC_SANITY_CHECK.md),
which tested one team and the opponent-adjustment strength. This one asks the
conference-level question directly: which inputs put the SEC there, and is the
result defensible against the historical replay.

## The scale of it

| Board | SEC | Big Ten | Big 12 | ACC | Other |
|---|---:|---:|---:|---:|---:|
| Model top 25 | **13** | 6 | 3 | 2 | 1 Ind |
| Final 2025 AP top 25 | 7 | 6 | 5 | 2 | 5 |

Six SEC teams are in the model's top 25 that the final AP poll did not rank at all:
LSU (No. 5), Missouri (14), Tennessee (17), Florida (18), South Carolina (20),
Auburn (22). The teams they displace are the ones that earned a 2025 ranking on the
field — Tulane, James Madison, Navy, North Texas, Virginia, Iowa, Houston, TCU.

A preseason projection *should* disagree with a final résumé poll, so this is not by
itself an error. It is the thing to explain.

## Where the gap comes from

Mean logit contribution, SEC minus Big Ten, decomposed over the 15 shipped features:

| Block | SEC − Big Ten | Share of the +0.635 gap |
|---|---:|---:|
| Recruiting (5 `rec_*`) | +0.416 | 65% |
| `portal_blue_in` | +0.136 | 21% |
| Prior-season O and D | +0.204 | 32% |
| `war_projected` | +0.122 | 19% |
| `talent` blend | −0.180 | −28% |
| Everything else | −0.062 | −10% |

Recruiting and blue-chip portal additions are 87% of the gap. Actual on-field
offence and defence are 32%, and the `talent` blend takes a quarter of it back.

## The counterfactual

Re-running the full round-robin power calculation with one block zeroed at a time
(client-side replication agrees with `ratings.json` to 5.4e-5):

| Board | SEC in top 25 | Notes |
|---|---:|---|
| Baseline | 13 | |
| Zero the 5 `rec_*` features | **8** | AAC, Pac-12, MAC, C-USA all re-enter |
| Zero `talent` | 12 | |
| Zero `war_projected` | 13 | |
| Zero the 5 portal features | 13 | |
| Zero prior-season O/D | 12 | |
| On-field only (O, D, returning) | 10 | |

Recruiting is the only block that moves the count materially. On prior-season play
alone the SEC still takes 10 of 25, so most of the concentration is earned. The
recruiting block adds the last three **and closes the top 25 to the Group of 5
entirely**.

## The structural problem

Recruiting is not one feature. It is five, and they are near-duplicates:

| | talent | war_proj | rec_qb | rec_ol | rec_skill | rec_front7 | rec_sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| talent | 1.00 | .76 | .79 | .89 | .90 | .90 | .90 |
| rec_skill | .90 | .82 | .82 | .90 | 1.00 | .94 | .92 |
| rec_front7 | .90 | .83 | .84 | .91 | .94 | 1.00 | .92 |

Two consequences:

1. **The block outweighs on-field play.** Σ|coef| is 0.736 for the five recruiting
   features against 0.468 for O and D combined. One construct, split five ways, ends
   up carrying more than the two features that describe what teams actually did.

2. **`talent` has become a suppressor.** Its coefficient is −0.301. The blend is
   documented as a PFF / recruiting / WAR composite, so recruiting enters the model
   twice — once inside `talent`, once as the five components — and the negative
   coefficient is the fit cancelling the double count. At correlations of .90 the
   split between `talent` and `rec_*` is not identifiable; those individual
   coefficients are an artefact of the regularisation path, not estimates of
   separable effects.

This is worth fixing on its own terms. It is not evidence that the SEC rank is wrong.

## Does the model actually over-rate the SEC? No — it under-rates it

Strict expanding replay, 2022–25, 2,913 games, per-season conference alignment.
Residual is actual win rate minus predicted, so positive means the model was too low.

| Conference | Sides | All games | Interconference |
|---|---:|---:|---:|
| Big Ten | 732 | +1.15pp | **+5.71pp** |
| Pac-12 | 317 | +1.42pp | +4.94pp |
| SEC | 669 | +1.26pp | **+4.65pp** |
| Big 12 | 629 | +0.68pp | +3.86pp |
| ACC | 688 | −0.28pp | −1.04pp |
| Mountain West | 516 | −0.43pp | −1.45pp |
| Sun Belt | 601 | −0.50pp | −1.85pp |
| American | 573 | −0.79pp | −2.91pp |
| Mid-American | 547 | −1.61pp | −6.01pp |
| Conference USA | 382 | −1.90pp | −6.26pp |

The model under-predicts SEC teams in interconference play by 4.65pp — and
under-predicts the Big Ten by *more*. It over-predicts the MAC and C-USA by about
6pp. This reproduces the 6.7pp figure in the Vanderbilt check on the current build.

**There is no empirical case for an SEC haircut.** The direction of the error is the
opposite of the intuition that prompted the question.

### Control: is this just favourites vs longshots?

Partly. The model is under-confident at both extremes — a compression consistent
with the L2 penalty at `C=0.1`:

| Predicted | n | Predicted | Actual | Residual |
|---|---:|---:|---:|---:|
| 0.0–0.1 | 244 | 6.0 | 2.9 | −3.12pp |
| 0.2–0.3 | 634 | 25.2 | 21.9 | −3.30pp |
| 0.7–0.8 | 634 | 74.8 | 78.1 | +3.30pp |
| 0.9–1.0 | 244 | 94.0 | 97.1 | +3.12pp |

P4 teams are usually the favourite in interconference games, so some of the
conference pattern is this effect wearing a conference label. Matching P4 against G5
within probability buckets leaves the sign mostly intact but the cells get small and
noisy (one G5 cell has n=1), so the honest statement is: **the P4/G5 split is
partly, not wholly, the favourite-longshot effect, and nothing in it points at the
SEC specifically.**

## One genuine defect found: `portal_blue_in` at LSU

`portal_blue_in` is a raw count of 4-star-and-up portal arrivals, z-scored. 94 of
136 teams have zero, so the median is −0.61 and the distribution has a long right
tail. The values are quantised at 0.4145 per transfer:

| Team | z | Blue-chip transfers | Logit |
|---|---:|---:|---:|
| **LSU** | **6.85** | **18** | **+0.783** |
| Ole Miss | 3.54 | 10 | +0.404 |
| Texas | 2.71 | 8 | +0.309 |
| Notre Dame / Miami | 2.29 | 7 | +0.262 |

LSU is nearly double the next team. Every other feature in the model tops out
between 2.3 and 4.3; this one reaches 6.85. The coefficient is linear, so the 18th
blue-chip transfer is priced identically to the 1st, and +0.783 is the single
largest term in LSU's rating — larger than its offence, defence or projected WAR.
With 94 teams at zero, the fit is determined almost entirely by the 0–3 range and
the value at 18 is extrapolation beyond training support. Diminishing returns are
also obvious on the football: there are only so many starting spots, and portal
additions displace each other.

Capping the feature at Ole Miss's value moves **LSU from No. 5 to No. 8** and
changes nothing else — the SEC count stays at 13. So this is a real defect, but a
local one. It explains LSU, not the conference.

## Incidental: two stale claims in the shipping code

- [`viz/app.js:106`](../viz/app.js) states "Talent is a retired feature with a
  coefficient of exactly zero". True of `model.json`, the retired v3 model. The app
  loads `model_v4.json`, where `talent` is **−0.301**. The comment is the basis for
  the reasoning in that whole block and should be corrected.
- `model_v4.json` has `whatif: null`, so `WI.enabled()` is false and the Players tab
  correctly disables the WAR inputs with a read-only tooltip — but the tab's own
  intro still says "edit expected WAR and the power ratings and matchup simulator
  respond." That copy no longer describes the build.

## Conclusion

The SEC's 13 of 25 is roughly 10 parts earned and 3 parts recruiting weight. The
board is not inflated by a conference bias — measured against four seasons of
results the model is too *low* on the SEC, and lower still on the Big Ten. What the
audit does find is a feature-design problem that happens to point the same way as
the SEC's strength: one recruiting construct is entered five times plus a sixth time
inside `talent`, giving it more weight than on-field performance and forcing a
negative suppressor coefficient to cancel the double count.

Worth doing, in order:

1. **Winsorise or log-compress `portal_blue_in`** before the z-score. It is a count
   with a linear price and one team 3 SD past everybody. Cheap, local, clearly right.
2. **Collapse the five `rec_*` features to one or two**, or regularise them as a
   group, and re-fit. Test whether the block should be entered alongside `talent` at
   all. The current split is not identifiable and its total weight exceeding O/D is a
   choice nobody made explicitly.
3. **Leave the conference alone.** The replay does not support a haircut.
4. Fix the two stale claims above.

Nothing here should ship on intuition; each of 1 and 2 needs to clear the strict
expanding replay on pooled Brier the way every other feature decision has.
