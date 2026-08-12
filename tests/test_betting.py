import unittest

import pandas as pd

from scripts.betting_backtest import implied, settle_games


class BettingBacktestTests(unittest.TestCase):
    def test_american_implied_probability(self):
        self.assertAlmostEqual(implied(-110), 110 / 210)
        self.assertAlmostEqual(implied(150), 100 / 250)

    def test_spread_settlement_uses_home_line_sign(self):
        games = pd.DataFrame([
            {"spread": -3.5, "spread_gap": 2.0, "actual_margin": 7},
            {"spread": -3.5, "spread_gap": -2.0, "actual_margin": 7},
        ])
        result = settle_games(games, "spread", 1.0)
        self.assertEqual(result.won.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
