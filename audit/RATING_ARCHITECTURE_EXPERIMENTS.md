# Where a feature belongs: inside the rating, or beside it?

## The question

The v4 model differences a team vector and hands the whole thing to one logistic:

```
logit = b1*(O_h - O_a) + b2*(D_h - D_a) + b3*(talent_h - talent_a)
      + b4*(ret_h - ret_a) + b5*(war_h - war_a)
```

Talent, returning production and roster WAR are therefore **parallel terms next to**
the O/D power rating, not inputs to it. Every experiment in this repo has added its
candidate the same way. The alternative is to fit the rating first — season N's
opponent-adjusted O and D predicted from prior O/D plus every feature — and let the
win model see two columns.

The second form is the more principled one on its face, and it is the only form in
which a new feature reaches the published power rankings, the playoff simulator and
the projected scores, all of which read the rating rather than the logistic's feature
list.

**It is also worse. Measurably, on both metrics.**

## Result, pooled 2022–25, against the shipping five-column model

| candidate | model columns | static Δ | static 95% CI | online Δ | online 95% CI |
|---|---:|---:|---|---:|---|
| **features beside the rating, plus the new ones** | 15 | **−.00539** | **[−.00835, −.00254]** | **−.00232** | **[−.00411, −.00055]** |
| rich rating, two columns out | 2 | −.00143 | [−.00628, +.00333] | −.00004 | [−.00341, +.00311] |
| rich rating + nonlinear O-vs-D cross | 3 | −.00097 | [−.00582, +.00368] | +.00040 | [−.00303, +.00361] |
| rich rating **and** features beside it | 15 | +.00028 | [−.00470, +.00512] | +.00079 | [−.00299, +.00428] |
| basic rating (talent, returning, WAR) | 2 | +.00168 | [−.00182, +.00487] | +.00154 | [−.00128, +.00409] |

Features beside the rating win by a wide margin and are the only candidate whose
interval excludes zero on both metrics. Folding the same features into the rating
recovers about a quarter of the gain and cannot be distinguished from the baseline.

Doing **both** — enriching the rating *and* keeping the columns — is the worst of the
three. The double-count costs more than it adds, which is at least a clean answer to
whether the two are redundant.

## Why compressing into a rating loses

Two columns cannot carry eleven predictors. The rating fit is a linear projection of
those features onto realised O and D, so anything that predicts *winning* but not
*opponent-adjusted efficiency* is discarded before the win model ever sees it. The
rating target and the model target are not the same target.

The enriched ratings are perfectly reasonable **ratings** — they correlate .62 with
realised O and .58–.61 with realised D in 2025 — they are simply a lossy channel for
feeding a win model.

Note also that the richer rating is not even a better rating: adding positional
recruiting and the portal moves 2025 rating correlation from .629/.605 to .624/.576.
The extra features help the *win* model and do nothing for the *rating*.

## The pairwise-matchup question, settled algebraically

A related concern is that the model differences like for like — home offence against
away **offence** — rather than home offence against away **defence**.

The linear cross term is already exactly representable:

```
b*(O_h - D_a) + b*(D_h - O_a)  ==  b*(O_h - O_a) + b*(D_h - D_a)
```

The two expand to the same expression. Further, the cross form is antisymmetric — the
requirement that swapping the teams negates the prediction — **only** when both
coefficients are equal. With separate coefficients on O and D, which the model fits,
the current form is strictly *more* general than the cross form, not less. Verified
numerically: at b1=1.3, b2=0.7 the cross form gives +0.710 forward and −0.877
reversed, so it is not a valid matchup vector at all.

What the linear form genuinely cannot express is a **nonlinear** mismatch: an elite
offence against a poor defence being worth more than the sum of the parts. That is
what the five granular `MATCHUP_PAIRS` terms already do for stat pairs, and adding one
for the headline rating (`inside_rich_cross` above) makes the model slightly worse on
both metrics.

So: the pairwise comparison is present, the linear version is already exact, and the
nonlinear version does not pay.

## What this does not settle

The measurement is about **game prediction**. If the goal is for positional recruiting
and the portal to move the published power rankings and the playoff simulator, they
have to enter the rating, because those surfaces read the rating. That is a product
decision with a measured price attached: about .004 of static Brier.

## Reproduction

```powershell
python -m scripts.rating_architecture_backtest
```

Artifact: `artifacts/rating_architecture_backtest.json`.
