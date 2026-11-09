# Most Deserving v2: wins above a playoff-caliber team

Status: production. Replaces the Colley ranking and the unmerged
`feature/record-first-deserving-v1` branch.

Every number in this report comes from `scripts/deserving_calibration.cjs`, which
imports the production engine (`viz/ranking-history.js`). Raw numbers are in
`audit/output/deserving_calibration.json`.

Data: CFBD regular-season FBS-v-FBS results, 2014–2025, conference title games
included, 2020 excluded (conference-only, disconnected schedules). FCS games and
bowls do not count, as before.

## 1. The final formula

For team *i* and each completed FBS game *g*:

- `q_g` = the chance that a playoff-caliber reference team wins game *g*: the same
  opponent, at the same venue.
- `W_g` = 1 for a win, 0 for a loss, ½ for a tie.

```
D_i = −ω·L_i + Σ_g (1 − q_g)          ω = 1 (one loss = one point)
    =  W_i − Σ_g q_g                  (when ω = 1: wins above the reference team)
```

`E_i = Σ(1 − q_g)` is the number of games the reference team would be expected to
lose on team *i*'s schedule. So `D = −L + E`. This is the "−L + Q" framework from
the brief, with Q no longer a weighted guess: Q is measured in the same unit as a
loss.

Each game, in loss units:

| Result | Contribution | Example |
|---|---|---|
| Win | `+(1 − q)` | Road win over an elite team, q = .35 → **+0.65**. Win over a weak team, q = .97 → **+0.03** |
| Loss | `−q` | Loss to a weak team, q = .97 → **−0.97**. Close road loss to a champion, q = .35 → **−0.35** |

The displayed split is exact: `score = losses + schedule credit`, with
`schedule credit = quality-win credit (from games won) + offsets (from games lost)`.

```
q_g = Φ( (r_ref − r_opp(−i) + h·v_g) / σ )
```

- `r_opp(−i)`: opponent rating from a ridge fit of this season's capped FBS margins
  **with every game of team i removed** (leave-team-out).
- `r_ref`: the playoff-caliber reference strength. It is a **distribution**,
  `r_ref ~ N(μ, s²)`, the same for every team in a snapshot (section 4). The score
  is the expectation over it, which is exact in closed form:
  `E[Φ((r_ref − r + h·v)/σ)] = Φ((μ − r + h·v)/√(σ² + s²))`.
- `v_g` = +1 home, −1 away, 0 neutral. `h` = home field. `σ` = win-probability scale.

Shown beside the score (not used to rank):

- an **80% score range**: the score against the 10th- and 90th-percentile
  playoff-caliber team;
- exact résumé rarity `p_i = P(W*_i ≥ W_i)`, `R_i = −log10 p_i`, averaged over the
  reference distribution (exact Poisson-binomial at 25 equal-probability points),
  with its own range (section 5);
- the **top-12 chance**: the share of the reference distribution under which the
  team ranks in the top 12.

Head-to-head (section 2a) then orders close neighbours. It never changes a score.

## 2. Conditional overrides

A team with *k* more losses ranks higher exactly when

```
E_A − E_B  >  ω · (L_A − L_B)
```

that is, when its schedule would cost a playoff-caliber team more than *k* extra
expected losses. There are no pairwise rules. The score is one number per team,
so the order is transitive and deterministic. The "Why is A ahead?" view prints
this inequality with the real numbers, and the net always equals the score
difference.

Historical frequency, final regular season, all pairs inside each season's top 25:

| Override | Model A (production) | Model B (rarity) | Colley (old) | v1 branch |
|---|---|---|---|---|
| 1 loss over unbeaten | 19 of 83 | 15 of 83 | 16 of 83 | 0 of 83 |
| 2 losses over unbeaten | 4 of 179 | 2 of 177 | 8 of 173 | 0 of 197 |
| 2 losses over 1 loss | 57 of 417 | 56 of 406 | 57 of 394 | 0 of 460 |
| 3 losses over unbeaten | 0 of 159 | 0 of 157 | 0 of 154 | 0 of 185 |

(Model A here is the production model: reference distribution and head-to-head on.
Head-to-head changes none of these counts; with a point reference instead of the
distribution, 2-loss-over-unbeaten is 5 of 179.)

All five two-loss-over-unbeaten cases are weak-schedule unbeatens:

- 2016 Penn State 11-2 over Western Michigan 12-0 (+0.53)
- 2017 Ohio State 11-2 over UCF 11-0 (+0.09)
- 2018 Georgia 10-2 over UCF 11-0 (+0.83)
- 2023 Missouri 9-2 over Liberty 13-0 (+0.01; Oregon 10-2 was also above Liberty
  with a point reference, and now sits 0.01 below it)

Examples of one-loss over unbeaten (score gap, schedule gap):

- 2023 Alabama 11-1 over Florida State 12-0: schedule worth 1.12 more expected
  losses, so Alabama clears the one-loss gap by 0.12. Rarity (Model B) disagrees
  narrowly: FSU's 12-0 is rarer (R 1.58 vs 1.43). This is a genuine borderline case.
- 2021 Alabama, Georgia and Michigan (each one loss) over Cincinnati 12-0 (+0.56,
  +0.16, +0.07).
- 2016 Ohio State 11-1 over Western Michigan 12-0 (+1.30).
- 2019 Oklahoma 11-1 over Clemson 12-0 (+0.67).

The v1 branch claimed conditional overrides. In 11 seasons it produced none: its
`4 × (Colley SOS − .5)` term never exceeded one loss, so it was a sort by loss count.

## 2a. Head-to-head between close résumés

**Rule.** After sorting by score, a team moves directly above the team just ahead
of it when (1) it won their games against each other on balance and (2) the two
scores differ by less than the closeness window, **0.39** loss units. Passes repeat
until nothing moves. A pair swaps at most once.

**Why 0.39.** It is measured, not chosen: the standard deviation of the score gap
between two nearby top-25 teams (within five places) when every opponent rating
moves by one standard error and the reference strength is redrawn from its
distribution (1,210 pairs, 40 draws each, 2014–2025). Closer than one SD, two
résumés are not reliably different, so the game between them decides.

**Why a reorder and not a bonus.** I first built head-to-head as a zero-sum score
bonus that faded with the gap. It failed two of the brief's requirements, and the
backtest showed both:

- With the window from the noise measurement, one team had **9** head-to-head games
  inside it. Each one's bonus shrinks as the team pulls away, so a better result
  elsewhere could *lower* that team's score (result sensitivity broken).
- The bonus moved winners and losers relative to teams they never played: with it
  on, 2019 Georgia 10-2 ranked above Clemson 12-0 (Georgia had close wins over
  Florida and Auburn); with it off, it did not. Georgia and Clemson did not play.

The reorder keeps the score pure (all score properties hold unchanged), keeps one
strict deterministic order, cannot loop on a cycle (A beat B, B beat C, C beat A),
and moves no team because of a game it did not play. The cost: head-to-head only
acts between neighbours. If a third team sits between the two, the game does not
reorder them; the head-to-head view says so.

**Effect.** Across 2014–2025 it reversed 65 final orders, 15 inside the top 25,
for example:

- 2014 Baylor 10-1 over TCU 10-1 (scores 0.04 apart)
- 2017 Ohio State 11-2 over Wisconsin 12-1 (0.22; Big Ten title game)
- 2017 Michigan State 9-3 over Penn State 10-2 (0.24)
- 2021 Ohio State 10-2 over Michigan State 9-2 (0.06)
- 2025 Texas Tech 11-1 over BYU 10-2 (0.001; Big 12 title game)

It changed none of the override counts in section 2. The page marks swapped teams
with **H2H**, and "Why is A ahead?" names the game and the gap.

## 3. Review of the v1 branch

| Problem | Effect |
|---|---|
| Schedule credit used Colley ratings that include the team's own result | Beating an opponent lowers that opponent's rating, so a win partly cancels its own schedule credit; losing to an opponent raises it |
| `4.0` per Colley point was not calibrated | Overrides never happened (table above) |
| `0.16 × W/12` "extra wins" term | Double counts the record and rewards extra games, not results |
| Margin term not opponent-adjusted | A 28-point win over the worst team earned the same as over the best |
| Benchmark: none | No meaning for "one loss equivalent" |

## 4. Opponent strength and the reference team

**Opponent model (calibrated).** Ridge least squares on capped margins:
`r_home − r_away + h·v = clip(margin, C) + noise`, penalty `λ Σ r²`. Chosen by
chronological out-of-sample log loss: for every week *w* ≥ 3, fit on weeks < *w* of
the same season, predict week *w*. Grid: λ ∈ {1,2,3,4,6,9}, C ∈ {14,21,28,38,60},
h ∈ {1.5…3.5}; σ by maximum likelihood.

| | Train 2014–19, 2021 | Test 2022–2025 (held out) |
|---|---|---|
| Selected | λ = 2, C = 60, h = 2.5, σ = 14.8 | — |
| Log loss | 0.548 | **0.564** (2,613 games) |
| Brier | 0.186 | 0.192 |
| Accuracy | — | 71.1% |

Test calibration (favorite's predicted vs observed win rate): .55/.58, .65/.66,
.75/.73, .85/.83, .94/.95. By phase: weeks 3–5 0.556, weeks 6–9 0.584, weeks 10+
0.553. Production parameters refit on all 11 seasons: λ = 2, C = 60, h = 3.0, σ = 15.05.

Only this season's results enter. No preseason ratings, talent, recruiting,
conference or polls.

**Leave-team-out.** For team *i*, all opponent ratings come from a fit with *i*'s
games removed. Team *i*'s results therefore cannot change how strong its opponents
look to it. This gives a strict proof of result sensitivity: changing one of *i*'s
results changes only that game's term, and a better result raises that term.

**Reference team (playoff-caliber): a distribution, not circular.** The current
#12 team is not used. "Playoff-caliber around #k" is the pooled set of
end-of-season ratings of the teams ranked **k−4 … k+4 in earlier seasons only**,
summarised as `N(μ_k, s_k²)`. The spread `s_k` carries two kinds of uncertainty at
once: how different playoff-caliber teams are from each other, and how the bar
moves from year to year. The distribution is fixed for every team in a snapshot
and does not depend on any current record.

| k (centre) | Ranks pooled | μ: points vs average FBS, neutral | s: SD of the pool | SD of the k-th team across seasons |
|---|---|---|---|---|
| 4 | 1–8 | 20.8 | 4.1 | 2.1 |
| 8 | 4–12 | 17.1 | 2.7 | 1.8 |
| **12 (production)** | **8–16** | **14.4** | **2.1** | **1.6** |
| 16 | 12–20 | 12.4 | 1.7 | 1.1 |
| 25 | 21–29 | 9.2 | 1.2 | 0.7 |

How the uncertainty is used:

- **Score (ranking):** the expected score over the distribution, exact in closed form
  (the win-probability scale widens from 15.05 to √(15.05² + 2.05²) = 15.19 points).
- **80% range:** the score against the 10th- and 90th-percentile playoff-caliber
  team. A stronger reference team expects fewer losses, so every score falls
  together; the range is wide (about ±0.4 for a top team) but ranks move much less.
- **Top-12 chance and rank range:** the full ranking, head-to-head included, at 25
  equal-probability points of the distribution.
- **Rarity:** exact Poisson-binomial tail at each of the 25 points, averaged. Games
  are independent *given* the reference strength; averaging over it is what makes
  them correlated, and this handles that correctly without Monte Carlo noise.

Backtests use only prior seasons for each test season (2014–2016 use 2014–2016,
flagged in-sample).

Why k = 12: with ω = 1 the score is exactly schedule-fair in expectation for a team
of the reference quality (expected score 0 on any schedule). Measured: a
#12-caliber team's expected score differs by only **0.08** loss units between the
hardest and easiest schedule quintiles. The 12-team cutoff is where fairness
matters most. With a #4 reference this bias becomes −0.46 against hard schedules at
the cutoff, and in simulation its two-loss-over-unbeaten calls were right about
the stronger team only 53% of the time (79% with #12).

## 5. Model B: playoff-caliber résumé rarity

`W* = Σ X_g`, `X_g ~ Bernoulli(q_g)` independent. The distribution is the exact
Poisson-binomial, by dynamic programming, O(G²), with no sampling noise.
`p = P(W* ≥ W)`, `R = −log10 p`. Tests check hand-computed cases and agreement with
200,000 Monte Carlo draws within four standard errors at every tail.

Monte Carlo is not needed for the production number: games are independent given
fixed q. Uncertainty about the reference strength is handled as a sensitivity
(±1 SD, other k), not mixed into p.

Evaluation against Model A:

| Test | Model A | Model B |
|---|---|---|
| Distinguishes a bad loss from a good loss | **Yes** | No: only the count of wins matters |
| Exact additive per-game explanation | **Yes** | No |
| Top-12 overlap, reference #8 vs #12 / #4 vs #12 | 11.7 / 11.3 | 11.9 / **11.8** |
| Top-12 overlap, reference distribution −1 SD vs +1 SD | 11.3 | **11.8** |
| Opponent-rating noise (one SE per team), top-12 overlap | 10.9 | 10.9 |
| Weekly rank movement, top 25, week 9 | 4.7 | 4.5 |
| 2 losses over unbeaten | 4 | 2 |
| Schedule bias for a #12-caliber team (hardest − easiest quintile) | 0.08 loss units | 0.06 rarity units |

B is slightly more stable and a little more record-first, but it cannot tell
which games were lost, which the brief asks for directly ("how damaging should a
loss to a weak opponent be?"). It is also not decomposable into the components the
page must show. So B is reported next to every score as a significance measure,
not used to rank.

## 6. Model C: what the rarity probabilities imply about the loss unit

Pairwise ordering agreement between Model A with loss weight ω and Model B, top 40
of every season:

| ω | 0.6 | 0.8 | **0.9** | 1.0 | 1.1 | 1.2 | 1.4 | 1.7 | 2.0 |
|---|---|---|---|---|---|---|---|---|---|
| Agreement | .908 | .967 | **.973** | .966 | .954 | .942 | .919 | .898 | .884 |

The tail-probability view agrees best with ω ≈ 0.9–1.0. It does **not** support a
loss premium above 1. The record-first behavior comes from the size of the unit
(one full expected loss of schedule difficulty is a lot), not from a multiplier.

Combined with the fairness result, the hybrid is: rank by A at ω = 1 (calibrated
by B), show B's rarity with every team, and print both in the head-to-head.
Adding R to D would count the same q_g twice in different shapes, so it is not done.

## 7. Challenged assumptions

**A heavier loss penalty (ω > 1).** This is the obvious way to make the ranking
"more record-first". It has a measurable cost: at ω = 1.5, a team of exactly
playoff-cutoff quality loses **1.02** points, on average, just for playing a
hardest-quintile schedule instead of an easiest-quintile one. Record-first beyond
ω = 1 is the same as rewarding weak schedules. It stays available in the
sensitivity explorer.

**Margin / convincingness.** Tested as `α · Σ(s_g − W_g)`, with `s` the
venue-adjusted, capped (28 points) probability that the margin shows the better
team (calibrated from game noise 16.2 and true-strength spread 13.3 points). At
α = 0.05 it did not improve any measure (simulation override accuracy .768 vs
.765), increased schedule bias at the cutoff (0.12 vs 0.08), and added two
borderline two-loss-over-unbeaten cases (2016 Michigan, 2023 Ole Miss, both by
under 0.03). **Production uses α = 0.** Running up the score earns exactly nothing.
The term remains in the explorer.

**Close losses vs blowout losses.** Under α = 0 they cost the same; the opponent
and venue set the cost (a road loss to an elite team costs about a third of a
loss to a weak team). That is the record-first answer: a loss is a loss, but whom
you lost to matters.

**Opponent strength "at the time of the game".** Not used. A win is judged by how
good the opponent really was, using every result available on the ranking date.
Early-season form is a worse estimate of the same quantity. Temporal integrity is
about the ranking date: no result after it is used.

**Committee agreement.** Not a target. For description only, mean overlap with the
final CFP top 12: Colley 10.4, Model A 10.3, Model B 10.2, v1 9.4.

## 8. Simulation: are overrides signal or noise?

1,650 simulated seasons (150 per season), real schedules, results drawn from
end-of-season ratings with 16.2-point game noise, opponent strength known exactly.

| | Model A | Model B | A, ω = 1.5 | A, #4 reference |
|---|---|---|---|---|
| 1 loss over unbeaten (top-25 pairs) | 18.9% | 14.3% | 10.8% | 12.6% |
| …when it happens, the 1-loss team is truly stronger | **76.5%** | 78.4% | 81.3% | 78.4% |
| …when it does not, the unbeaten is truly stronger | 72.3% | 69.9% | 68.5% | 69.1% |
| 2 losses over unbeaten, per season | 0.32 | 0.18 | 0.01 | 0.06 |

Overrides are at least as reliable as keeping the unbeaten team ahead. They are not
noise. "Truly stronger" is a sanity check, not the objective.

## 9. Required properties

| Property | How it holds | Test |
|---|---|---|
| Transitivity | One scalar per team | sorted-score check |
| Determinism | Canonical game order, neutral games canonical | input shuffle, reversal |
| Result sensitivity | Leave-team-out; each game term rises with the result | 12 random flips on the live schedule; score and rarity never fall; a loss flipped to a win is worth ≥ 0.5 |
| Schedule sensitivity | 1 − q larger for stronger opponents and road games | synthetic leagues; road win > home win |
| Loss accountability | −1 per loss, before any credit | identical schedules |
| Margin restraint | Margin enters only the opponent fit (with the team removed) | every win turned into a 70-point win: gain exactly 0; an extra home win over the weakest FBS team: at most +0.003 (rarity +0.001) |
| Temporal integrity | Snapshots use games ≤ date; reference from prior seasons | random changes to later weeks altered 0 of 8 snapshots |
| Explainability | `score = losses + schedule (+ performance)` exactly | every row, every pair |
| Small samples, disconnected schedules | Ridge prior keeps every fit solvable; ratings shrink to average | two-game disconnected league |
| Neutral-site representation | Same game whichever side is listed first | swap test |

Rank (not score) monotonicity cannot be guaranteed by any schedule-adjusted
system: a better result also raises the rating of teams you beat, which helps
*their* other opponents. The score guarantee is strict.

Early season: through week 3 the top 12 shares 4.3 teams with the final top 12;
by week 5, 8; by week 11, 9.8. Weekly top-25 movement falls from 13 places
(week 3) to 3 (week 11). The page says early rankings move sharply.

## 10. Calibrated vs philosophical parameters

| Parameter | Value | Type | Source |
|---|---|---|---|
| Ridge λ, margin cap C, home field h, scale σ | 2, 60, 3.0, 15.05 | **Calibrated** | Chronological log loss |
| Reference distribution N(μ_k, s_k²) | 14.4 ± 2.1 (k = 12) | **Calibrated** given k | Prior seasons, ranks k−4…k+4 |
| Head-to-head window | 0.39 | **Calibrated** (1 SD of score-gap noise); "1 SD" is the assumption | Noise measurement, section 2a |
| Reference centre k | 12 | Philosophical, justified | Fairness at the playoff cutoff |
| Loss weight ω | 1.00 | Philosophical, supported | Schedule fairness; rarity implies 0.9–1.0 |
| Convincingness α | 0 | Philosophical, tested | No measured benefit |
| Performance cap | 28 | Philosophical | Only used if α > 0 |

The data cannot identify a unique "correct" k or ω, because deservingness is
normative. They are shown as assumptions, and the page's Cutoff sensitivity view
re-ranks every team under 36 combinations (k ∈ {8,12,16}, α ∈ {0, 5%, 10%},
ω ∈ {1, 1.25}, head-to-head off/on).

## 11. Separation from other systems

Unchanged: Power Ratings, game probabilities, betting outputs, the R committee
simulation, projected CFP brackets and `resume_scores.json`. Mark's Playoff uses
the new résumé scores (it always used Most Deserving) with the same P4/G6/Notre Dame
selection rules, and its committee comparison recomputes at the committee's games
cutoff with the same model.

## 12. Limitations and recommended next steps

1. FCS games are excluded, so an FCS loss costs nothing. Adding FCS opponents to the
   rating fit as one pooled team would fix this.
2. Uncertainty covers the reference strength only. Opponent-rating uncertainty is
   measured (section 5, and it sets the head-to-head window) but is not added to the
   displayed range. Adding it would widen every range by roughly the 0.39 noise SD.
   Head-to-head acts only between neighbours, by design (section 2a).
3. Opponent ratings early in the season shrink toward average, so early scores are
   compressed. This is honest but makes early rankings noisy.
4. 2023 Alabama vs FSU and 2023 Missouri vs Liberty are within 0.12 points.
   Ties this close are inside the opponent-rating noise (section 5). The sensitivity
   view shows these teams as "bubble"; the page should not imply false precision.
5. Historical committee rankings are only used for description; a separate study of
   where the committee departs from Model A would be useful.

## Reproduce

```
node scripts/deserving_calibration.cjs <path to CFBD games_YYYY.json directory>
node --test tests/test_most_deserving.cjs
```

The script rewrites `viz/data/deserving-model.json` and
`audit/output/deserving_calibration.json` (about 45 seconds).
