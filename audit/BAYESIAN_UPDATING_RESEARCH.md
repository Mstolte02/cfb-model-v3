# Bayesian in-season updating

Written 18 September 2026. Four questions, four answers, and only one of them says
ship something.

## Summary

| question | answer |
|---|---|
| Does a variance-tracked gain beat the constant K? | Better, **−.00050 Brier, below the .001 bar and the interval crosses zero.** Do not ship. |
| Is the gain's value in the season-long decay, or in differing between teams? | **Entirely between teams.** Shared-variance decay is worth −.00008; per-team is worth −.00042 with an interval that excludes zero. |
| Is refreshing the preseason model's inputs better than the Elo walk? | **No, and not close.** +.0088 Brier worse, interval nowhere near zero. Adding it beside the walk is worth nothing either. |
| Should an in-season player value shrink toward the league mean or toward himself? | **Toward himself for receivers**, +2 to +7% MAE at every cut. Early-season only for quarterbacks. Nothing for backs or tight ends. |

## What the code does today, verified

`src/dynamic.py` holds **one scalar per team** and moves it by a constant gain,
`K = .20`, for every team and every week. There is no uncertainty anywhere:
`ratings` is `dict[str, float]`, and every game converts to a probability through the
same fixed `margin_sigma = 17.105`.

**The player layer has no in-season channel at all.** `war_projected` is a feature of
the STATIC model, and the shipped state has `dynamic_blend = 1.0` — confirmed in
`artifacts/2026_dynamic_state.json` and in the browser payload
`viz/data/model_v4.json`, where `winpTeams` computes
`(1 - blend) * pStatic + blend * pDynamic`. With `blend = 1` the static model
contributes **exactly zero** to every published win probability, projected spread and
flagged bet. Player WAR survives only as the week-0 value of the team rating. A team
that loses its quarterback in week 3 gets no direct model response at all.

## 1. Variance tracking versus a constant gain

`scripts/inseason_kalman_backtest.py`, three arms, strict expanding replay, parameters
selected on earlier seasons only. `kalman_shared` keeps ONE variance for the league,
so its gain decays through the season but is identical across teams; `kalman` keeps
one per team, so the gain also differs between them. Splitting it this way is the
point of the experiment.

| pooled 2023-25, n = 2,189 | Brier | log loss | vs constant | 95% CI |
|---|---:|---:|---:|:---|
| constant_k (shipping) | .178986 | .531468 | — | — |
| kalman_shared | .178904 | .531144 | −.000082 | [−.000818, +.000641] |
| kalman | **.178483** | **.530158** | −.000504 | [−.001163, +.000140] |
| kalman vs kalman_shared | | | **−.000422** | **[−.000880, −.000007]** |

Better in three folds of four (2023 .17430→.17396, 2024 .18394→.18364,
2025 .17863→.17775; 2022 is a wash). But −.000504 is half the project's .001
materiality bar and its interval contains zero, so **it does not ship**, on the same
grounds the moving-K arm did not.

The decomposition is the part worth keeping. `kalman_shared` — a gain that decays over
the season and nothing else — is worth **nothing at all** (−.00008, interval wide open
around zero). Everything the filter buys is in `kalman` minus `kalman_shared`, the
per-team half, and that difference is the only interval here that excludes zero. The
implied gain in 2025 bears it out:

| week | 1 | 3 | 5 | 7 | 9 | 11 | 13 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean gain | .234 | .224 | .217 | .209 | .197 | .195 | .188 |
| range across teams | — | .180–.256 | .171–.257 | .168–.242 | .160–.229 | .155–.227 | .131–.213 |

The season-long decay is .234 to .188, a ratio of .79. The spread **across teams in a
single week** is wider than that for most of the year. `audit/INSEASON_UPDATE_EXPERIMENTS.md`
read the moving-K folds' preference for decay as "directionally coherent with a
Bayesian reading"; this says the coherent part was the wrong half. A schedule for K
over the season is not where the information is. Which team you are updating is.

### Why the margin-space pilot said four times as much

`scripts/kalman_rating_pilot.py` is a standalone reimplementation in margin space with
a weak carry-forward prior, and it put the same comparison at **−.00218**, 95%
[−.00427, −.00008] on held-out 2024-25, with the gain decaying .199 to .090 (ratio
.45). It is kept because the gap between the two numbers is the lesson: **a filter's
value is almost entirely a function of how bad the prior is.** Hand it last season
regressed toward the mean and it looks worth .0022; hand it the v4 preseason model
and it is worth .0005. Any future "Bayesian X would help" claim on this repo should
be read against that ratio before it is believed.

The pilot also shows two things the real arm confirms: the fitted `v0` is not at a
grid edge, and using the filter's own variance as a per-game sigma makes probabilities
WORSE — the fitted scale on the epistemic part is exactly zero, and forcing it in
costs .0017 Brier. That agrees with the existing finding that a sigma curve keyed to
pregame certainty had no value. Per-game credible intervals are a presentation idea,
not an accuracy one.

## 2. Refreshing the preseason model versus walking the rating

`scripts/inseason_refit_backtest.py`. The shipping updater sees only final scores.
The alternative is to keep the preseason model's structure and recompute the part the
season has actually changed — the opponent-adjusted O/D composites — from games played
so far. `refit` swaps the model's O and D columns for to-date composites built the
same way the preseason ones are (same eight `game_advanced` fields, z-scored within
season, same SRS-style fixed point), leaves talent, returning production, WAR, portal
and recruiting at their preseason values, and refits the combination forward-only.

| pooled 2023-25, n = 2,189 | Brier | log loss | vs elo | 95% CI |
|---|---:|---:|---:|:---|
| preseason (no update at all) | .201817 | .583494 | +.023167 | [+.016136, +.030123] |
| **elo (shipping)** | **.178650** | **.530574** | — | — |
| refit | .187426 | .553375 | **+.008776** | **[+.004747, +.012938]** |
| refit_elo (both) | .178435 | .529188 | −.000215 | [−.001947, +.001391] |

**The answer is no.** Refreshing the inputs is worse than walking the rating by
.0088 Brier, an interval nowhere near zero, and it is worse in every fold (2023
.17847 vs .17329, 2024 .18963 vs .18394, 2025 .19423 vs .17863). Putting both in the
same model recovers the difference and adds nothing on top of the walk alone
(−.0002, interval straddling zero).

Two things stop this being a dull negative. In-season efficiency is clearly
informative — `refit` beats the never-updated preseason model by .014, so the
composites are carrying real signal; they are simply carrying less of it than the
scores are. And the null on `refit_elo` is a genuine null rather than a wide one,
which says the two channels are close to redundant here. That is a different result
from `audit/RATING_ARCHITECTURE_EXPERIMENTS.md`, where features beside the rating beat
features folded into it; the distinction is that those were preseason features the
rating could not see, and in-season efficiency is a second measurement of something
the rating already tracks.

2022 is not a fold: it has only 2021 behind it, and this design needs one season to
fit the team model and another to generate in-season rows for the arm coefficients.

## 3. What an in-season player value should shrink toward

`scripts/player_prior_stabilisation.py`. `src/qbwar.py` estimates opponent-adjusted
per-play value with a ridge on player + opponent one-hots. The ridge penalty IS the
shrinkage, and with no offset it pulls every player toward zero, which is the league
mean. `fit_season_values` now takes an optional `prior`, which fits on
`ppa - prior[player]` and adds the prior back — the exact Gaussian posterior with
`alpha = sigma^2 / tau^2`, so a player is pulled toward his own last-season value
instead. **The default is `None`, which is the original behaviour byte for byte**,
because `build_qb_values` feeds `artifacts/qb_values.csv`, which feeds the preseason
WAR build and therefore the shipped model.

Graded against the rest of the same season, itself opponent-adjusted so neither side
of the comparison is raw:

| position | cut wk 2 | wk 3 | wk 4 | wk 5 | wk 6 | wk 7 | wk 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **WR** (n≈1,500) | **+7.0%** | **+7.0%** | **+4.5%** | **+3.3%** | **+3.7%** | **+3.5%** | **+4.1%** |
| QB (n≈450) | +1.6% | +2.9% | +3.3% | +0.8% | −0.5% | −1.9% | −3.3% |
| RB (n≈980) | +1.0% | −1.1% | −0.3% | −0.7% | −1.7% | −1.3% | −0.9% |
| TE (n≈460) | −0.3% | +0.6% | +0.4% | −0.1% | −0.9% | −1.4% | −0.6% |

(MAE improvement over shrinking to the league mean. Correlation with rest-of-season
improves for QB, RB and WR at nearly every cut even where MAE does not — at WR week 3
it goes .09 to .22 — so the prior is helping ORDER players everywhere and helping
their LEVEL only sometimes.)

**Receiver is the robust result.** It is positive at every cut, and it stayed positive
across every sensitivity I ran: ridge alpha 3/10/30, target alpha 1/10, and regressing
the prior by phi of .4/.6/.8/1.0 — more than twenty cells, all positive. Quarterback
helps through about week 4 and then turns negative, and that reversal is real rather
than an artifact of the target's own shrinkage (it is unchanged at target alpha 1).
Back and tight end are nothing.

So the defensible change is scoped: **use the prior for receivers, and for
quarterbacks only early.** A blanket switch would be picking the cells that worked,
which is the trap `audit/BET_THRESHOLD_CALIBRATION.md` spends most of its length
rejecting.

**Only quarterback had ever been tested before this.** It is also worth recording why
the position list stops where it does: CFBD's per-game PPA feed contains **no
defenders at all**. 2022, 2024 and 2025 all return QB, RB, WR, TE and a trickle of FB
and nothing else. No defensive player can be updated in season from this source, which
is where PFF keeps its value — and PFF's exports here are season totals, so that half
cannot be refreshed weekly either.

An earlier version of this measurement compared RAW per-game PPA against an
opponent-adjusted prior, which loaded the comparison in the prior's favour, and a
version before that required a player to appear in every week to date — which selects
only teams that have not had a bye, collapsed the sample from 547 to 69, and read as
"in-season play never stabilises". Both are fixed. The current script adjusts both
sides and cuts by week rather than by appearance.

## 4. Availability data: what exists as of today

`war_model/scrape_twodeep.py` records that no structured injury source exists because
cfbdepth.com forbids automated access. **That is out of date**, and the replacement is
better than what it describes.

**CFBD does not have it.** Their OpenAPI spec lists 84 endpoints and not one matches
injury, availability or depth. (Note also that `src/data/cfbd_client.py` mentions a
`/depth` endpoint that is no longer in the spec.) Two endpoints there ARE worth a look
for the player work above, though, because both are better inputs than raw per-game
PPA: `/wepa/players/passing` and `/wepa/players/rushing` are opponent-adjusted EPA per
player, and `/player/usage` carries snap share.

**The Big Ten publishes it, and the first reports are live.** Four submissions a week
— three days out, two days out, one day out, and two hours before kickoff — with
Probable / Questionable / Doubtful / Out / Out (1st Half) midweek and Game Time
Decision / Out / Out (1st Half) on gameday. Players not listed are available. The
landing page is <https://bigten.org/fb/availability-reports/>, and the reports
themselves are an embedded third-party widget at
`app.hdintelligence.com/?source=B10&sport=Football&conf=B10&type=report`, with an
`&type=archive` sibling. Read on 18 September it returns this Saturday's slate
already: per game, per team, grouped by status, each row carrying position, jersey
number and name. There is a JSON API behind the widget — `POST /api/public-load`
answers 200 while several sibling calls answer 401 — so the integration is a real one
rather than PDF scraping, but it is an undocumented private API and that question
should be settled before anything is built against it.

**The SEC and ACC publish their own.** SEC at <https://www.secsports.com/fbreports>,
mandatory since 2024; ACC at least 48 hours before conference games. I did not fetch
the SEC page: `secsports.com/robots.txt` explicitly disallows `anthropic-ai`, and
while that rule is about AI crawlers rather than about Mark, it is their stated
preference and worth honouring. Its `User-agent: *` block only disallows `/cms-wmt`,
`/null` and `/undefined`, so the reports path is not excluded for an ordinary client;
`bigten.org` allows the availability page and disallows `bigten.org/api/`;
`theacc.com` ends its bad-bot list with `Allow: /`.

**Coverage is conference games only.** A Big Ten team's non-conference opener has no
report, and the Big 12 is not in the list above at all. So this is a partial feed and
whatever reads it has to say so.

## What to do next

1. **Take the receiver prior.** It is measured, it is robust, and `fit_season_values`
   already accepts it. Scope it to WR, and to QB before week 5.
2. **Leave the filter as research**, recorded beside moving-K. If it is revisited, the
   thing to chase is the per-team half on its own — a gain that varies by team without
   a full filter — because the shared-decay half is now known to be worthless.
3. **Do not pursue an in-season refit of the preseason model.** Answered.
4. **Settle the HD Intelligence terms before building a scraper.** If they allow it,
   the availability channel becomes possible; if not, the SEC and ACC pages are
   ordinary HTML.
5. **When the availability channel is measured, measure it conditionally.** Personnel
   shocks affect a few percent of games, and pooled over 2,189 any real effect dilutes
   to nothing — which is a fair description of what happened to the production-informed
   WAR channel in `audit/PLAYER_PRODUCTION_EXPERIMENTS.md` (+.00103 static, +.00030
   after the weekly update, interval crossing zero). Restrict to games within a week or
   two of a reported change, and grade against closing-line value, which
   `audit/BET_THRESHOLD_CALIBRATION.md` already names as the fastest-detecting
   measurement at this sample size.

## Reproduction

```powershell
python -m scripts.inseason_kalman_backtest
python -m scripts.inseason_refit_backtest
python -m scripts.player_prior_stabilisation
python -m scripts.kalman_rating_pilot          # standalone, no API key needed
```

Artifacts: `inseason_kalman_backtest.json`, `inseason_refit_backtest.json`,
`player_prior_stabilisation.json`, each with per-fold parameters and bootstraps.

## Sources

- Glickman & Stern (1998); <https://www.glicko.net/research/glicko.pdf> and
  <https://www.glicko.net/research/dpcmsv.pdf>
- Yurko & Benz, "College Football Volatility: A Bayesian state-space model of the
  transfer portal and NIL impact", NESSIS 2025,
  <https://www.nessis.org/nessis25/Ron-Yurko.pdf>. Their four-model comparison on
  9,003 FBS games found plain constant-innovation-variance Glickman & Stern best on
  held-out 2023-24 and the stochastic-volatility extension by far the worst, which is
  why `tau` is held constant here.
- Duffield et al., "A State-Space Perspective on Modelling and Inference for Online
  Skill Rating", <https://arxiv.org/abs/2308.02414>
- Big Ten 2026 availability policy, <https://bigten.org/fb/article/60284/>; reports at
  <https://bigten.org/fb/availability-reports/>
- ACC availability reports,
  <https://www.espn.com/college-football/story/_/id/45794684/acc-start-releasing-injury-reports-conference-games>

## Implementation — 21 September 2026

The supported player estimator now lives in `src/player_updates.py`, invoked by
`python -m scripts.update_player_values`. It uses alpha 10, a prior fitted on the
immediately preceding season, WR personal priors throughout and QB personal priors
through week 4. RB/TE and IDs absent from prior data retain league-mean shrinkage.
The minimum fit population matches the research: 20 observations and five players.

Only completed regular-season FBS-vs-FBS game IDs enter the fit. The default export
uses the last fully completed scheduled week; `--through-week` supports explicit
historical cutoffs. Current-season API caches refresh; prior-season inputs reuse
cache. Stable IDs carry priors across transfers. Unobserved players receive no
fabricated current-season value. Insufficient data leaves the previous export intact.

A Monday workflow publishes `viz/data/player_values.json` and deploys the site.
Team Overview displays estimates, sample sizes, cutoff and prior type separately
from preseason WAR. This implements the measured **player-estimation** improvement;
it does not claim a validated team-prediction gain or convert EPA/play into WAR.
The audit did not estimate that conversion, and the production dynamic blend of 1
would ignore a change made only to the static WAR feature. A team-level overlay
therefore still needs a forward-only incremental test before adoption.

The remaining findings retain their original decisions: no Kalman/moving-K change,
no efficiency-refit replacement, and no availability scraping. Second-pass opponent
lift is already implemented in Most Deserving (see `DESERVING_RANKINGS.md`).
