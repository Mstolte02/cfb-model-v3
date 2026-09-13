import unittest

import pandas as pd

from scripts.betting_backtest import implied, select_line, settle_games


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

    def test_historical_market_uses_draftkings_only(self):
        game = {"lines": [
            {"provider": "Bovada", "spread": -7},
            {"provider": "Draft Kings", "spread": -3},
        ]}
        book, line = select_line(game)
        self.assertEqual(book, "DraftKings")
        self.assertEqual(line["spread"], -3)
        self.assertEqual(select_line({"lines": [game["lines"][0]]}), (None, None))

    def test_game_total_is_not_a_supported_historical_market(self):
        with self.assertRaises(ValueError):
            settle_games(pd.DataFrame(), "total", 2.0)


if __name__ == "__main__":
    unittest.main()
