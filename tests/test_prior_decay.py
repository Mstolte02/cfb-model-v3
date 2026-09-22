from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.prior_decay_backtest import DEF, OFF, _ewma, todate_od


def row(week, team, opponent, value):
    result = {"week": week, "team": team, "opponent": opponent}
    result.update({column: value for column in OFF})
    result.update({column: -value for column in DEF})
    return result


class PriorDecayTests(unittest.TestCase):
    def test_future_games_cannot_change_a_prior_week_state(self):
        history = pd.DataFrame([
            row(1, "A", "B", 1.0), row(1, "B", "A", -1.0),
            row(2, "A", "B", 0.5), row(2, "B", "A", -0.5),
        ])
        before = todate_od(history, week=3, halflife=4.0)
        future = pd.concat([
            history,
            pd.DataFrame([row(10, "A", "B", 100.0),
                          row(10, "B", "A", -100.0)]),
        ], ignore_index=True)
        after = todate_od(future, week=3, halflife=4.0)
        pd.testing.assert_frame_equal(before, after)

    def test_ewma_weights_recent_games_more_than_old_games(self):
        rising = _ewma(np.array([0.0, 1.0]), halflife=1.0)
        falling = _ewma(np.array([1.0, 0.0]), halflife=1.0)
        self.assertGreater(rising, falling)
        self.assertAlmostEqual(_ewma(np.array([0.0, 1.0]), np.inf), .5)


if __name__ == "__main__":
    unittest.main()
