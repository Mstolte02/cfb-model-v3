# Notre Dame and conference title game losers: historical experiment

Date: 2026-11-25. Research only; production committee weights and bracket are unchanged.

## Hypothesis

A team ranked ahead of Notre Dame before a conference championship remains ahead of Notre Dame after losing that title game, because Notre Dame did not play an additional game.

## Data and design

Use `data/committee_history.json` (ESPN dated CFP polls and completed games) and the existing weekly timeline features in `scripts/train_committee_timeline.py`. Examine the last pre-championship poll and selection poll in each year. Count title game losers previously above Notre Dame when Notre Dame appeared in the pre-championship Top 25. For model validation, fit the existing Ridge(alpha=30) on earlier seasons only and use its six-team-pool ranking; test 2018–2025, excluding the pandemic 2020 season. The proposed hard clause moves an otherwise overtaking Notre Dame back behind any such title loser, with variants restricted to prior-rank gaps of 1, 2, 3, or unrestricted. It does not alter the football win model.

## Observations

Seventeen title game losers started above Notre Dame in the seven usable held-out final weekends. Fifteen stayed ahead; two fell below. With a one-rank gap, the split is two kept ahead and two overtaken:

| Season | Prior order | Final order | Existing model's final order | Hard clause |
| --- | --- | --- | --- | --- |
| 2021 | Oklahoma State 5, Notre Dame 6 | Notre Dame 5, Oklahoma State 9 | Oklahoma State 5, Notre Dame 6 | Does not correct the model miss |
| 2023 | Iowa 16, Notre Dame 17 | Notre Dame 16, Iowa 17 | Notre Dame 16, Iowa 17 | Creates a wrong ordering |
| 2024 | Penn State 3, Notre Dame 4 | Penn State 4, Notre Dame 5 | Penn State 4, Notre Dame 5 | No change |
| 2025 | Alabama 9, Notre Dame 10 | Alabama 9, Notre Dame 11 | Notre Dame 9, Alabama 10 | Corrects one pair ordering |

In 2024, Texas entered No. 2, lost the SEC title game, and remained No. 3, ahead of Notre Dame at No. 5. This is close to the 2026 Texas concern, but Texas entered two places ahead that year. The 2021 Oklahoma State and 2023 Iowa cases show a title game loss can cause a team to fall past Notre Dame.

## Held-out comparison

On the seven final releases, the baseline averaged 0.777 rank absolute error across actual Top 25 teams, and 11.571 of the Top 12 overlapped. All tested hard-clause rank-gap thresholds (1, 2, 3, unrestricted) yielded the same aggregate results: 0.777 and 11.571. The one-rank clause improved 2025's rank error from 0.56 to 0.48, but worsened 2023's from 0.88 to 0.96. It did not repair the incorrect 2021 Oklahoma State ordering.

This is a small, selected set of examples. No claim that the unchanged aggregate metric proves equivalence outside this sample is warranted. The unconditional clause has a direct historical counterexample and did not improve the held-out ranking metric; do not add it to production.

## Current bracket interpretation

The 2026 November 24 baseline has Texas No. 3 and Notre Dame No. 4. The bracket orders teams by all-draw mean final committee score, not by one conditional outcome. It places Notre Dame ahead because its mean simulated rank is 3.61 versus Texas's 4.93. These averages do not establish what the model predicts conditional on both teams winning their final regular-season games and Texas losing an SEC championship. Testing that particular claim requires retaining joint simulation outcomes (Texas finale, Notre Dame finale, title-game appearance/result, and relative final rank) rather than using the published marginal summaries.
