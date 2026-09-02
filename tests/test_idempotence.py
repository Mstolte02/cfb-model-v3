import unittest

import numpy as np
import pandas as pd

from scripts.idempotence_backtest import fixed_point


def games(rows):
    return pd.DataFrame(rows, columns=[
        "home_team", "away_team", "home_points", "away_points", "neutral_site"])


class FixedPointTests(unittest.TestCase):
    def test_balanced_cycle_centers_at_zero(self):
        d = games([
            ("A", "B", 21, 14, True),
            ("B", "C", 21, 14, True),
            ("C", "A", 21, 14, True),
        ])
        fp = fixed_point(d)
        np.testing.assert_allclose(fp.fixed, 0.0, atol=1e-12)

    def test_reversing_every_result_negates_strength(self):
        d = games([
            ("A", "B", 28, 14, True),
            ("A", "C", 24, 21, True),
            ("B", "C", 17, 10, True),
        ])
        reverse = d.copy()
        reverse[["home_points", "away_points"]] = d[["away_points", "home_points"]]
        left, right = fixed_point(d), fixed_point(reverse)
        a = left.series(left.fixed).sort_index()
        b = right.series(right.fixed).sort_index()
        np.testing.assert_allclose(a, -b, atol=1e-12)

    def test_second_pass_moves_a_schedule_aided_result_down(self):
        # A's unbeaten record came against the two teams that lost every game.
        d = games([
            ("A", "C", 21, 14, True),
            ("A", "D", 21, 14, True),
            ("B", "E", 21, 14, True),
            ("B", "F", 21, 14, True),
            ("G", "C", 35, 7, True),
            ("G", "D", 35, 7, True),
            ("E", "G", 21, 14, True),
            ("F", "G", 21, 14, True),
        ])
        fp = fixed_point(d)
        first, second = fp.series(fp.first), fp.series(fp.second)
        self.assertLess(second["A"] - first["A"],
                        second["B"] - first["B"])


if __name__ == "__main__":
    unittest.main()
