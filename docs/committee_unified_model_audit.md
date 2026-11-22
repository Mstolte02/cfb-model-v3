# Unified committee model, November 21, 2026

The deployed championship-context model and the separate nonlinear experiment had diverged. Revision `unified-nonlinear-opponent-margin-2026-11-21` combines the quadratic prior-rank term with all existing championship features and the existing 0.25 head-to-head comparison-pool adjustment. Python training, browser projection and R simulation use the same features.

Regular-season margin is signed, clipped to ±35, divided by seven, and multiplied by `(30 - opponent_prior_rank)/30`. Unranked opponents use rank 30 and receive zero margin weight. Only results since the previous poll contribute to this term. This rewards margin against opponents the prior committee ranked; it intentionally does not reward running up the score against unranked teams. Championship margins retain their separate feature and cannot double-count here.

## Chronological evaluation

Every yearly fit uses earlier seasons only, excludes 2020, and uses ridge alpha 30. The comparison uses all available candidate teams, the same head-to-head adjustment, and the same weekly Top 25 ordering metric. Choose among feature forms using 2018–23 development results; report 2024–25 separately. These later years have been inspected in prior work and are not a pristine new holdout. 2026 never enters training or model selection.

| Features | Development rank MAE | 2024–25 rank MAE |
|---|---:|---:|
| Current main: championship context | 1.2736 | 1.1760 |
| Main + quadratic prior rank | 1.2272 | 1.1480 |
| Main + unweighted margin | 1.3232 | 1.1920 |
| Main + quadratic + unweighted margin | 1.2784 | 1.0600 |
| Quadratic + capped-at-14 margin | 1.2704 | 1.0680 |
| Quadratic + capped-at-21 margin | 1.2864 | 1.0520 |
| Quadratic + signed logarithmic margin | 1.2816 | 1.0480 |
| Quadratic + winning margin only | 1.2320 | 1.0800 |
| **Quadratic + opponent-weighted margin (selected)** | **1.2208** | **1.1080** |
| Quadratic + close-win indicator | 1.2272 | 1.1160 |

Selected model reduces development MAE 4.15% and evaluation MAE 5.78% versus current main. Top-12 overlap remains 11.5/12 in 2024–25. The small evaluation sample comprises ten poll transitions; this is not a guarantee of future accuracy. Reproduce feature comparisons with `python -m scripts.audit_committee_features`; regenerate shipped model/backtests with `python scripts/train_committee_timeline.py`.

No manual ranking swaps or protected ranking cutoffs are introduced. In particular, beating an unranked opponent by a large margin does not automatically lift a team over another winner. Week-in-progress forecasts remain separate local scenarios; official site scheduling still holds the published poll Tuesday–Saturday.
