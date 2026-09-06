# Power Ratings, Most Deserving and Stock Watch

Power Ratings retains the existing predictive neutral-field model. Most Deserving
is a separate result-based résumé score, with no current AP poll inputs or roster
what-if effects. It recalculates from the published schedule whenever the page loads.

## Second-pass opponent lift

Feature definitions are copied from the tested `research/idempotence-rating`
experiment. For completed regular-season FBS games:

1. Cap home scoring margin at ±28, then subtract 2.5 points at non-neutral sites.
2. First pass: average team-perspective adjusted margin.
3. Second pass: first pass plus average opponents' first-pass values.
4. Lift: population-standardize `z(second pass) - z(first pass)`.
5. Quality: ridge Massey system `(BᵀB + .25I)r = Bᵀmargin`, centered then standardized.
6. Combine record, quality, SOS, power-conference status, second-pass lift and
   near-neighbor head-to-head using fitted historical weights. H2H is based on
   provisional ranks within 10 places, before the final H2H contribution.

`python -m scripts.fit_deserving --schedules-dir DIR --rankings-dir DIR` fits
275 ranked team-seasons, 2015–2025. Inputs are the sportsdataverse
[CFB schedule release](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/cfb_schedules)
(`cfb_schedules_YEAR.csv.gz`) and
[CFBD-derived ranking files](https://github.com/pdaly42/sports-edge/tree/main/data/raw/cfb)
(`rankings_YEAR.csv`). The final CFP committee poll is the training target.
Weights are rescaled to keep win percentage at 10. They are stored separately in
`viz/data/deserving-model.json`; the predictive and playoff models are unchanged.

The original research reported +0.00601 leave-one-season-out Spearman from the lift
feature, with 8/11 seasons improving. This is a season-end evaluation, not an
in-season validation. Early disconnected schedules can produce ties and abrupt
movement. Exact equal scores share ranks. Teams need one completed FBS matchup;
FCS results count toward record and receive the audited -2 SD SOS proxy. No
completed matchups produces an empty state, not preseason fabricated résumé ranks.

Browser feature values and H2H matched the Python research for all 2025 teams
(max numerical discrepancy < 5e-11 against rounded reference values).

## Stock Watch

Uses the final two published `ratings.history` snapshots for weekly movement and
the explicitly labeled Preseason snapshot for season movement. Missing baselines
produce an empty state. Sorts by rank places gained/lost; shows percentage-point
rating changes alongside rank changes. It includes all teams and opens Team History
on selection. Partial-week labels are preserved.

## Retro marks

`viz/data/retro-logos.json` records the exact archive year page, archive team name,
image source and local asset for each of 138 FBS teams. Marks are sourced from
[SportsLogos.Net's historical archive](https://www.sportslogos.net/), with local PNGs
for stable loading. Only the ranking boards and Stock Watch use these overrides;
the current team metadata stays intact. Logos belong to their respective owners.
The marks are historical school athletic identities, including schools whose
football programs started after the selected logo era.
