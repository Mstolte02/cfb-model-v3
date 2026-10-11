# Power Rating X-Ray

Open `viz/power-xray/index.html` through the site's HTTP server, or visit
`https://mstolte02.github.io/cfb-model-v3/power-xray/` after deployment.
The Power Rankings navigation links to this independent read-only page.

## Data and calculation

The explorer fetches `model_v4.json` and `ratings.json` on each load with caching
 disabled. It reads only the live ensemble, never the frozen v4 display features.
It mirrors `scripts/ensemble_replay.py` stack selection, form eligibility, zero
fallbacks, scaling, clipped sigmoid and equal member weighting. Current power is
the mean neutral win probability against every other team in `ratings.json`.
It warns if the independently reconstructed result differs from the published
rating by more than 0.000002 probability units (0.0002 percentage points), and
withholds the rank in that case. The two files can briefly be out of sync.

Seven feature groups are attributed using exact Shapley values over all 128
subsets. A missing group means its matchup-difference term is zero for both teams,
not that the model is refitted. Contributions sum to power minus the 50% baseline.
The current opponent field stays fixed. These are descriptive model attributions,
not causal effects, prior-to-current movement, or feature importance in training.

The sandbox reuses those subset values. It does not change the live model, roster
scenarios, stored inputs, picks, kickoff locks or frozen pregame records. Its
switches do not affect the full-model opponent probabilities or comparison table.
Links support `?team=Notre+Dame&compare=BYU`.

The preseason input is aggregated in the published payload. The explorer labels
this limit rather than inventing talent/position-level decompositions. Unused
columns remain visible and labeled. Model disagreement is not an uncertainty
interval. Missing PFF/WAR inputs follow the production zero fallback and are
reported in the coverage disclosure.

## Verification

Run `node --test tests/test_power_xray.cjs` (Node and Python stdlib only).
Tests cover all published team ratings, every neutral matchup against the Python
runtime, additive reconciliation, no mutation, reciprocal probabilities, missing
and empty PFF/WAR payloads, legacy stacks, and insufficient form games.
Run `python3 scripts/publish.py` to build the existing GitHub Pages site.
