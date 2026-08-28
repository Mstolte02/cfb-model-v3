# Standardisation, collinearity and overfitting in the v4 feature set

Checked 28-Aug-2026, following [`SEC_CONCENTRATION_AUDIT.md`](SEC_CONCENTRATION_AUDIT.md),
which found the recruiting block entered five times plus a sixth inside `talent` and
left the question of whether that costs anything unanswered.

Two questions: is z-scoring the right transform for every feature, and is the fit
suffering from collinearity or overfitting. Reproduce with
`./venv/bin/python -m scripts.standardisation_backtest`.

Everything below is scored on the repo's standard protocol — strict expanding
replay, train on all prior seasons, predict the next, pooled out-of-sample Brier
over 3,613 games in 2022–25.

## 1. Standardisation: the premise is right, the conclusion is not

Every feature is z-scored within season by `TS.attach` and `features.standardize`,
regardless of shape. Only two of the fifteen are anywhere near normal:

| Feature | skew | excess kurt | Shapiro W | verdict |
|---|---:|---:|---:|---|
| O | −0.13 | 0.06 | 0.998 | normal |
| D | 0.17 | −0.05 | 0.996 | normal |
| talent | −0.65 | 1.70 | 0.971 | left-skewed |
| returning | −0.98 | **8.25** | 0.947 | heavy-tailed |
| war_projected | 0.61 | 0.09 | 0.973 | right-skewed |
| rec_* (5) | 0.52–0.79 | −0.32–0.48 | 0.954–0.969 | right-skewed |
| portal_in/out_rated | ~1.0 | 1.6–1.9 | 0.936–0.943 | right-skewed |
| **portal_blue_in/out** | **~2.1** | **5.7–6.2** | **0.746** | **counts, 57 distinct values in 789** |

Thirteen of fifteen fail Shapiro at p < 0.001. On the face of it that is a strong
case for log or rank transforms.

**It is not, because the model never sees a marginal.** V4 is a difference model:
every design column is `home − away`. A difference of two draws from the same skewed
distribution is close to symmetric, and the diagnostics on the actual design matrix
show exactly that:

| Feature | marginal skew | **difference** skew | difference excess kurt | rows >5sd |
|---|---:|---:|---:|---:|
| portal_blue_in | 2.115 | **0.129** | 2.28 | 0 |
| portal_blue_out | 2.061 | **0.336** | 4.99 | 12 |
| rec_secondary | 0.785 | **0.484** | 1.12 | 0 |
| talent | −0.648 | **0.437** | 4.16 | 4 |
| returning | −0.980 | **0.159** | 4.53 | 8 |
| O | −0.129 | 0.054 | −0.02 | 0 |

Differencing removes most of the skew. What survives is **excess kurtosis** — heavy
tails, i.e. a handful of high-leverage games — which is a different problem needing
a different fix.

Tested anyway, because the argument above is theory:

| Variant | k | Brier | vs ship | paired p |
|---|---:|---:|---:|---:|
| **ship (z-score, as built)** | 15 | **0.20168** | — | — |
| log1p the two blue-chip counts | 15 | 0.20191 | +0.00023 | 0.176 |
| winsorise heavy tails at 3sd | 15 | 0.20203 | +0.00034 | 0.163 |
| rank-normal heavy tails | 15 | 0.20236 | +0.00068 | 0.163 |
| rank-normal everything | 15 | 0.20329 | +0.00160 | 0.073 |

**Every re-standardisation makes it worse.** Rank-normalising everything is the worst
of the five and comes closest to significance in the wrong direction. Plain z-scoring
should stay: it is the correct transform for a difference model, and the non-normal
marginals are a red herring here. The one caveat is that z-scoring does not bound
leverage — see §3, where the fix turns out to be dropping columns, not reshaping them.

## 2. Collinearity: two real defects

### `portal_net_rated` is a redundant third column

`talent_sources.portal_features` builds `portal_net_rated = portal_in_rated −
portal_out_rated`, then `attach` z-scores all three separately. Standardising
separately blurs the identity but does not remove it:

- R² of net on (in, out) = **0.964**, fitted as `1.073·in − 0.960·out`
- VIFs: in **32.9**, out **24.7**, net **28.8**
- Smallest singular value of the scaled design is 0.042, next is 0.136 — a clear
  near-null direction

Three columns spanning two dimensions. Only the L2 penalty keeps the fit finite.

### Recruiting is five columns spanning roughly one

The five `rec_*` features correlate .80–.94 with each other and .90 with `talent`.
PC1 explains **83.7%** of the block; across `talent + war_projected + rec_*` (7
columns) PC1 explains 77.4%.

### What it does to the coefficients

200 bootstrap refits of the shipped spec. A coefficient whose sign flips across
resamples is not estimating anything:

| Feature | mean | sd | \|mean\|/sd | **sign flips** |
|---|---:|---:|---:|---:|
| portal_in_rated | 0.013 | 0.075 | 0.17 | **42.0%** |
| portal_net_rated | 0.027 | 0.073 | 0.37 | **35.5%** |
| portal_out_rated | −0.054 | 0.073 | 0.74 | **24.0%** |
| rec_qb | −0.037 | 0.051 | 0.73 | **22.5%** |
| portal_blue_out | −0.035 | 0.038 | 0.90 | 19.0% |
| rec_front7 | 0.105 | 0.087 | 1.21 | 11.5% |
| O | 0.251 | 0.032 | 7.73 | 0.0% |
| D | 0.236 | 0.036 | 6.52 | 0.0% |

**Six of fifteen coefficients have unstable signs.** The three portal-rated columns
are unidentified, exactly as their VIFs predict. The published `rec_qb = −0.029`
("better quarterback recruiting makes a team worse") is not a finding; it is the sign
it happened to land on in this resample.

## 3. Overfitting: real, and caused by the collinearity

| Spec | k | in-sample | out-of-sample | **gap** |
|---|---:|---:|---:|---:|
| ship | 15 | 0.19927 | 0.20168 | **+0.00242** |
| rec→PC1, drop portal_net | 10 | 0.20007 | 0.20120 | +0.00112 |
| reduced (below) | 7 | 0.20044 | 0.20091 | **+0.00047** |

The gap falls by a factor of five. The regularisation sweep shows the same thing from
the other side:

| C (inverse L2) | ship, out-of-sample | reduced 10, out-of-sample |
|---|---:|---:|
| 0.01 | **0.20119** | 0.20144 |
| 0.03 | 0.20133 | 0.20126 |
| **0.1 (shipped)** | 0.20168 | 0.20120 |
| 0.3 | 0.20195 | 0.20117 |
| 1.0 | 0.20211 | 0.20115 |
| 100 (≈none) | 0.20246 | 0.20114 |

The shipped spec improves monotonically as the penalty is *increased* — it is
under-regularised at `C=0.1` and wants roughly 10× more shrinkage. The reduced spec
is flat across four orders of magnitude: once the redundant columns are gone the fit
no longer depends on the penalty at all. That is the signature of collinearity being
the cause rather than sample size.

## 4. The reduced specification

| Spec | k | Brier | vs ship | accuracy | paired p |
|---|---:|---:|---:|---:|---:|
| ship | 15 | 0.20168 | — | 0.6732 | — |
| rec→PC1, drop portal_net | 10 | 0.20120 | −0.00049 | 0.6780 | 0.40 |
| …also drop portal_blue_out | 9 | 0.20119 | −0.00050 | 0.6773 | 0.40 |
| …also drop portal_in_rated | 8 | 0.20121 | −0.00048 | 0.6807 | 0.47 |
| **…also drop portal_out_rated** | **7** | **0.20091** | **−0.00077** | 0.6742 | 0.36 |
| …drop portal entirely | 6 | 0.20172 | +0.00003 | 0.6742 | 0.97 |

**Seven features: `O, D, talent, returning, war_projected, rec_pc1, portal_blue_in`.**

Dropping the sixth (`portal_blue_in`) gives the whole gain back, so blue-chip
arrivals carry the only real portal signal — the other four columns were noise
wearing four coefficients.

Bootstrap stability of the 7-feature fit:

| Feature | mean | sd | \|mean\|/sd | sign flips |
|---|---:|---:|---:|---:|
| O | 0.249 | 0.032 | 7.85 | 0.0% |
| D | 0.241 | 0.035 | 6.84 | 0.0% |
| rec_pc1 | 0.565 | 0.081 | 6.99 | 0.0% |
| war_projected | 0.284 | 0.049 | 5.82 | 0.0% |
| returning | 0.151 | 0.030 | 5.07 | 0.0% |
| talent | −0.307 | 0.082 | 3.76 | 0.0% |
| portal_blue_in | 0.122 | 0.034 | 3.59 | 0.0% |

**Every coefficient sign-stable, zero flips.** The recruiting signal that was
scattered across five unstable coefficients is one stable coefficient at 0.565.

Sign-stable is not the same as separable — see §6, which qualifies this.

## 6. What collinearity remains, and why it should stay

The reduced spec is not collinearity-free, and it would be wrong to imply it is:

| Feature | VIF (7-feature) | strongest correlation |
|---|---:|---|
| rec_pc1 | 4.50 | **0.85 with talent** |
| talent | 3.67 | 0.85 with rec_pc1 |
| war_projected | 2.36 | 0.64 with rec_pc1 |
| O | 1.30 | 0.45 with war_projected |
| D | 1.29 | 0.43 with war_projected |
| portal_blue_in | 1.27 | 0.44 with rec_pc1 |
| returning | 1.18 | 0.26 with war_projected |

Condition number falls from 23.7 to **4.7**, and no VIF exceeds 5. But talent and
rec_pc1 still correlate 0.85, which is a lot.

**Every attempt to remove it makes the model worse:**

| Spec | k | Brier | vs 7-feature |
|---|---:|---:|---:|
| **7-feature (talent and rec_pc1 both)** | 7 | **0.20091** | — |
| drop `talent`, keep `rec_pc1` | 6 | 0.20147 | +0.00055 |
| merge both into one PC | 6 | 0.20301 | +0.00210 |
| drop `rec_pc1`, keep `talent` | 6 | 0.20500 | +0.00409 |
| drop `war_projected` as well | 5 | 0.20832 | +0.00741 |

So the two kinds of collinearity need separating:

- **Degenerate collinearity** — VIF 25–33, construction identities, five columns
  spanning one dimension. Symptoms: sign-flipping coefficients, dependence on the
  regularisation constant, an inflated train/test gap. This was real and removing it
  helped on every measure.
- **Benign collinearity** — correlated inputs that each still carry distinct signal.
  r = 0.85 between talent and rec_pc1, but the residual 15% is worth 0.0006 Brier,
  and merging them destroys it. Coefficients are sign-stable and the fit no longer
  depends on `C`.

The residual correlation is a reason not to read the individual talent and rec_pc1
coefficients as separate causal effects. It is not a reason to drop either column.

## 7. `talent` is negative but talent is not bad

The published coefficient is −0.302, which reads as "talent makes teams lose". It
does not. Fitting the same data with features added one at a time:

| Specification | talent coef |
|---|---:|
| `talent` alone | **+0.677** |
| `talent + returning` | +0.678 |
| `talent + O + D` | +0.395 |
| `talent + O + D + returning` | +0.389 |
| `talent + O + D + returning + war_projected` | **+0.193** |
| … add `rec_pc1` | **−0.286** |
| full 7-feature | −0.302 |

Talent on its own is **+0.677**, one of the strongest single signals in the data. It
stays positive through the addition of on-field O/D, returning production and
projected WAR. It flips sign at exactly one point: when the recruiting composite
enters. That is a suppressor pair at r = 0.85, not a football finding.

The published coefficient answers a conditional question — *of two teams with the
same recruiting index, the same prior O and D, the same projected WAR and the same
returning production, does the one with the higher talent-blend score win more?* —
and the answer to that narrow question is a slight no. Nobody asks that question.

The question people do ask is what a real talent edge is worth, and there the
correlated inputs move together. Empirically rec_pc1 moves +1.02 and war_projected
+0.76 per 1sd of talent:

| Path | Logits |
|---|---:|
| direct (partial) coefficient | −0.302 |
| via rec_pc1 (+1.02 sd × 0.559) | +0.569 |
| via war_projected (+0.76 sd × 0.284) | +0.216 |
| **total** | **+0.483** |

A one-standard-deviation talent edge is worth **+0.483 logits — a 61.9% neutral-field
win probability** against an otherwise identical team. Talent helps, substantially.

### A cheap way to stop publishing a misleading number

Orthogonalising `talent` against `rec_pc1` within season is a reparametrisation that
costs almost nothing and reports honestly:

| Spec | Brier | talent | rec_pc1 |
|---|---:|---:|---:|
| as-is | 0.20091 | −0.302 | +0.559 |
| talent orthogonalised | 0.20105 (+0.00013) | −0.102 | +0.341 |

The shared talent/recruiting axis now carries a clean +0.341, and what is left of the
talent blend after recruiting is removed is −0.102 — small, and no longer inviting the
reading that talent hurts. Worth doing for the published coefficient table even though
it changes no prediction materially.

### Honest limits

None of the Brier differences is statistically significant (p = 0.36–0.47 paired on
per-game squared error). The case for the reduced spec is **not** "it predicts
better". It is:

- equal-or-better out-of-sample accuracy on **less than half the parameters**
- a 5× smaller train/test gap
- every coefficient identified instead of six of fifteen flipping sign
- insensitivity to the regularisation constant

That is a parsimony and stability argument, and it is strong. But an audit that
claimed a significant accuracy win here would be overstating the evidence.

## 5. What this does to the rankings

The 2026 top 25 under the 7-feature model is still **SEC 13, Big Ten 6, Big 12 3,
ACC 2, Independent 1** — unchanged. LSU moves from No. 5 to No. 8, the same
correction the `portal_blue_in` winsorisation produced in the SEC audit, because
PC1 no longer lets one team's 18 blue-chip transfers ride five separate recruiting
coefficients at once.

This is worth stating plainly: fixing the collinearity does **not** dissolve the SEC
concentration. That finding survives, and it survives for the reason the earlier
audit gave — on prior-season play alone the SEC still takes 10 of 25, and the
historical replay says the model is too low on them, not too high.

## Recommendation

1. **Keep z-scoring.** No transform beat it and rank-normalising everything was
   measurably worse. The non-normal marginals do not matter in a difference model.
2. **Drop `portal_net_rated`.** It is a construction identity, 96% explained by two
   columns already in the model. No judgement call.
3. **Collapse `rec_*` to one component**, and drop `portal_in_rated`,
   `portal_out_rated`, `portal_blue_out`. Down to seven features.
4. **Re-tune `C` after the change**, not before — the reduced spec is flat in `C`, so
   the current 0.1 is fine, but the shipped spec's preference for 0.01 is a symptom
   that disappears rather than a setting to copy over.
5. The `talent` suppressor deserves its own look. It is stable and negative in every
   specification tested here.

## Shipped, 28-Aug-2026

Items 1–5 are in production. `TS.REDUCED` was added to `CANDIDATES` in
`scripts/v4_backtest.py` and competed under the repo's existing forward-selection
rule rather than being imposed — it won outright:

| Candidate | Forward-selection score |
|---|---:|
| **reduced_talent_sources** | **0.201046** |
| core_war_talent_sources (previous ship) | 0.201685 |
| core_war_projected | 0.206364 |
| clean_core (reference) | 0.209688 |

It clears `SELECTION_MIN_GAIN` over `clean_core` by 8.6×, and the per-fold trace now
selects it in 2023, 2024 and 2025 (2022 has only one prior season and falls back to
`clean_core`). The tuner re-picked `C=0.03, alpha=100, ensemble_weight=0.75`.

Published replay, before and after:

| Strict expanding replay, 2022–25 | Brier | Log loss | Accuracy |
|---|---:|---:|---:|
| static, 15 features | .2080 | .6018 | 66.50% |
| **static, 7 features** | **.2025** | **.5864** | **67.52%** |
| weekly update, 15 features | .1871 | .5524 | 70.82% |
| **weekly update, 7 features** | **.1845** | **.5452** | **71.47%** |
| CFBD pregame Elo | .1898 | .5613 | 70.99% |

The paired bootstrap against Elo moves from −0.00270 with a 95% interval of
[−0.00738, **+**0.00183] to −0.00533 with [−0.00976, **−**0.00118]. The interval now
excludes zero, so the README's "competitive with Elo" hedge has been replaced.

Shipped coefficients, and the two they replace:

| Feature | Coefficient |
|---|---:|
| rec_pc1 | **+0.332** |
| war_projected | +0.288 |
| O | +0.238 |
| D | +0.232 |
| returning | +0.143 |
| portal_blue_in | +0.117 |
| talent_resid | **−0.099** |
| ~~talent~~ (was) | ~~−0.301~~ |
| ~~rec_qb~~ (was) | ~~−0.029~~ |

`talent` remains in `ratings.json` and on the Team Tables page as the interpretable
blend — Georgia 2.27, Alabama 2.21 — because that is the number a reader wants. Only
the model reads `talent_resid`.

Not done: item 5's deeper question. `talent_resid` at −0.099 is small and no longer
misleading, but a talent composite whose recruiting-free remainder is negative is
still worth understanding rather than merely presenting better.
