# Most Deserving ranking

The Colley-matrix ranking that this file described was replaced on 2026-11-08.

The current method, its calibration, historical validation, sensitivity tests and
limitations are documented in [MOST_DESERVING_V2.md](MOST_DESERVING_V2.md).

In one line: résumé score = −(FBS losses) + the number of games a playoff-caliber
team would be expected to lose on the same schedule, with opponent strength fitted
from this season's results with each team's own games removed.

## Stock Watch

Stock Watch compares the latest two published `ratings.history` snapshots for the
weekly view and the explicitly labeled preseason snapshot for the season view.
Missing baselines produce an empty state. Teams sort by rank places gained or lost,
with neutral-win-rate percentage-point changes shown alongside rank movement.
