# WAR: sources in, sources out, and what it is worth alone

Three questions about the WAR model, answered together because they share a frame.

## 1. Quarterback WAR is now PFF, and EA is out of the build

**Done.** Two changes, both narrowing the inputs to PFF, CFBD and production:

* The quarterback map read the `Average` column of `qbs_2026.xlsx`, a mean of five
  z-scores across PFSN, PFF, EPA, an execs poll and EA. It now reads `PFF Z` alone.
* `EA_BLEND_SNAPS` is 0, so the rule that handed EA's ordering to players under a
  snap threshold no longer fires. The published projection is built from
  `projections_2026_v2.csv` directly rather than the EA-blended file.

What moved: **112 of 273 quarterback slots** changed, mean 0.059 WAR, max 0.485.
Removing the EA blend moved **5,354 of 5,633 non-quarterback slots**, mean 0.012.

| team | player | before | after |
|---|---|---:|---:|
| Kansas | Isaiah Marshall | 0.462 | 0.947 |
| Navy | Braxton Woodson | 0.632 | 1.103 |
| Oklahoma | John Mateer | 0.646 | 0.226 |
| Texas A&M | Marcel Reed | 0.798 | 0.388 |
| TCU | Jaden Craig | 0.585 | 0.186 |

The single-source column carries the literal string `Unknown` where PFF has no
opinion, which the averaged column hid. **Ten of 124 named starters** have no PFF
grade — a true freshman or a JUCO arrival — and they now keep the projection's own
number rather than an imputed one, which is the right answer: there is no PFF evidence
to override it with.

## 2. Would opponent-adjusted production replace PFF? No

`horse_race.py` already compared the PFF facet set against the CFBD production set and
found production behind. That comparison was unfair, because the CFBD facets are
**raw**: a quarterback's PPA per play is credited to him whether he faced the best
secondary in the league or the worst.

`production_oppadj.py` builds the fair version. Per facet, every game contributes one
observation of `season-mean production of team i ~ attack[i] − defence[j]`, solved as
a ridge over the season's schedule; each player's grade shifts by his own team's
correction, and everything downstream is untouched.

| facet set | same-season CV r | next-season CV r |
|---|---:|---:|
| PFF | .830 | **.475** |
| CFBD production, raw | .737 | .361 |
| CFBD production, opponent-adjusted | .749 | **.376** |

Opponent adjustment helps — +.012 same-season, +.015 next-season — and comes nowhere
near closing a gap of nearly .10. **Replacing PFF with production would cost about a
fifth of the model's forward correlation.**

Two reasons it is not close, and neither is fixable by a better adjustment:

* **The correction is small.** Mean |shift| is 0.018 against a grade SD of 0.224 — 8%
  of a standard deviation. Twelve-game schedules in one conference are not that
  different from each other.
* **CFBD has no offensive line data at all.** `horse_race.py` stage 2 shows the OL row
  as "no data for this group". PFF grades five OL facets. No amount of adjusting the
  facets that exist creates the ones that do not.

The honest read is the one already in the repo: production belongs **alongside** PFF,
not instead of it. The union beats either (next-season .489 against .475 and .374).

A caveat on method: this adjustment is season-aggregate, because facets run 2014–2025
while the play cache starts in 2021. A play-level adjustment would be stronger. Given
a .10 gap and an 8%-of-an-SD correction, it is very unlikely to change the verdict.

## 3. WAR as the whole model

Roster WAR currently enters as one input among six, blended at a fraction of one of
them, which makes it hard to see what it is worth. Run as the entire model:

| candidate | inputs | static Brier | online Brier | accuracy |
|---|---:|---:|---:|---:|
| **WAR alone** | **1** | **.20939** | .19008 | .668 |
| WAR projected + lagged | 2 | .20869 | .18976 | .670 |
| power rating alone | 1 | .21466 | .19093 | .649 |
| recruiting talent alone | 1 | .22647 | .19548 | .634 |
| last season's WAR alone | 1 | .22424 | .19681 | .625 |
| clean core | 4 | .20958 | .18824 | .664 |
| core + WAR | 5 | .20695 | .18669 | .664 |

**One column of projected roster WAR matches the four-column core on static Brier**:
−.00018, interval [−.00497, +.00460]. It beats the power rating alone by .0051 and
recruiting talent alone by .0169, both with intervals excluding zero.

That is a real result for the WAR model. As a *preseason* forecast, the roster
projection carries as much as last year's ratings, this year's recruiting and
returning production put together.

It loses online (+.00184), and the reason is structural rather than a flaw: the core's
O/D ratings get overwritten by the in-season dynamic update while WAR stays a
preseason number. The same pattern as the curvature phase — WAR is a prior, and priors
stop mattering once results arrive.

`core + WAR` remains the best of the set (−.00263 static, −.00155 online, both
intervals excluding zero), which is what the shipping configuration already does.

## Reproduction

```powershell
python -m scripts.war_only_backtest
python war_model/production_oppadj.py
python war_model/depth_correction.py --in projections_2026_v2.csv --out projections_2026_final.csv
```

Artifacts: `artifacts/war_only_backtest.json`, `war_model/production_oppadj.json`.
