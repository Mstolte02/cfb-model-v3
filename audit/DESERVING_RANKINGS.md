# Most Deserving ranking

Power Ratings retains the predictive neutral-field model. Most Deserving is a
separate result-based résumé score with no current AP poll inputs or roster what-if
effects. It recalculates from the published schedule whenever the page loads.

## Second-pass opponent lift

The feature definitions come from the tested `research/idempotence-rating`
experiment. For completed regular-season FBS games:

1. Cap home scoring margin at ±28, then subtract 2.5 points at non-neutral sites.
2. First pass: average team-perspective adjusted margin.
3. Second pass: first pass plus the average first-pass value of each opponent.
4. Lift: population-standardize `z(second pass) - z(first pass)`.
5. Quality: solve the ridge Massey system `(BᵀB + .25I)r = Bᵀmargin`, center it,
   then standardize it.
6. Combine record, quality, SOS, power-conference status, second-pass lift and
   head-to-head against nearby teams using the fitted historical weights.

`python -m scripts.fit_deserving --schedules-dir DIR --rankings-dir DIR` fits 275
ranked team-seasons from 2015–2025. The final CFP committee poll is the training
target. Weights are rescaled to keep win percentage at 10 and stored in
`viz/data/deserving-model.json`. The predictive and playoff models are unchanged.

The research reported a +0.00601 leave-one-season-out Spearman improvement from
second-pass opponent lift, with 8 of 11 seasons improving. This evaluates final
season résumés, so early in-season rankings can contain ties and move sharply.

## Stock Watch

Stock Watch compares the latest two published `ratings.history` snapshots for the
weekly view and the explicitly labeled preseason snapshot for the season view.
Missing baselines produce an empty state. Teams sort by rank places gained or lost,
with neutral-win-rate percentage-point changes shown alongside rank movement.
