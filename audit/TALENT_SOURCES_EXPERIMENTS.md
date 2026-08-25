# Talent inputs the model was not using

## Status

**Adopted: positional recruiting plus the rated transfer portal.** Together they beat
the clean core by **−.00632 static and −.00330 online**, with both 95% intervals
excluding zero — 6.3× and 3.3× the .001 adoption bar. This is the strongest predictive
result any phase in this project has produced.

## What was missing

Team talent is three axes blended into one number: the CFBD recruiting composite,
PFF roster-aware grades, and roster WAR. Two things CFBD publishes were in none of
them.

**The transfer portal.** A team that lost four rated starters and replaced them with
three looks identical to the recruiting composite as one that stood still. Entries
carry origin, destination, a recruit rating and a star grade, so arrivals and
departures can be priced separately. Coverage grows with the era it describes: 1,770
entries in 2021 against 4,499 in 2025.

**Recruiting by position group.** The shipping talent figure is one number per team.
The same composite splits into quarterback, offensive line, skill, front seven and
secondary, which is what makes "stacked in the wrong places" expressible at all. Each
season averages the three classes that make up the bulk of a roster.

## Results, pooled 2022–25, against the clean core

| candidate | static Δ | static 95% CI | online Δ | online 95% CI |
|---|---:|---|---:|---|
| portal, all entries | +.00045 | [−.00003, +.00096] | +.00013 | [−.00017, +.00043] |
| portal, rated only | −.00064 | [−.00199, +.00070] | −.00046 | [−.00157, +.00063] |
| positional recruiting | −.00558 | [−.00799, −.00322] | −.00307 | [−.00480, −.00135] |
| portal (all) + recruiting | −.00526 | [−.00766, −.00290] | −.00277 | [−.00446, −.00106] |
| **recruiting + rated portal** | **−.00632** | **[−.00917, −.00351]** | **−.00330** | **[−.00521, −.00138]** |

## The portal needed to be built properly before it showed anything

The first construction summed a rating over every entry, imputing the 44% with no
recruit rating at the league mean. That turns "incoming talent" into a head count
wearing a quality label: a team that took ten unrated transfers scored like a team
that took ten rated ones. It made the model **worse**.

Counting only rated players, and tallying blue-chip (4★+) arrivals and departures as
their own columns, flips the sign. On its own that is still inside the noise —
−.00064 with an interval spanning zero — but added to positional recruiting it
contributes a further −.00074 static and −.00023 online, and the combined candidate is
the best of the set on both metrics.

So the portal earns its place, and it earns it **alongside** positional recruiting
rather than instead of it. Which is roughly what the sport looks like: where a roster
is strong and who it just gained or lost are two different facts, and neither
substitutes for the other.

## Why the portal is not larger on its own

Part of it is already priced. The PFF roster-aware talent axis is *this year's roster
crossed with last year's grades*, so an incoming transfer who graded well elsewhere is
already counted in the blend, and roster WAR does the same through
`war_model/portal_2026.json`. What the new columns add is the part those cannot see:
the recruit-rating view of arrivals with no PFF history, and the cost of departures,
which a roster-based axis registers only as an absence.

## Reproduction

```powershell
python -m scripts.talent_sources_backtest
```

Artifact: `artifacts/talent_sources_backtest.json`. The two endpoints are
`cfbd_client.transfer_portal` and `cfbd_client.recruiting_groups`.

## Not yet done

These are measured but **not wired into the shipping talent blend**. Adoption means
deciding whether they enter `config.TALENT_BLEND` as new axes or stay as separate
model features, re-running `scripts/train_v4.py`, and re-exporting. The evidence
supports doing it; the plumbing is a separate change.
