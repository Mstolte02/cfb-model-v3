import unittest

import numpy as np
import pandas as pd

from scripts.spread_cover_research import fit, paper


class SpreadCoverResearchTests(unittest.TestCase):
    def test_fit_keeps_positive_market_relative_direction(self):
        gaps = np.tile(np.array([-12.0, -6.0, 6.0, 12.0]), 30)
        outcomes = np.tile(np.array([0, 0, 1, 1]), 30)
        beta = fit(pd.DataFrame({"margin_gap": gaps, "cover_home": outcomes}))
        self.assertGreater(beta[1], 0)

    def test_paper_gate_is_fixed_and_scores_both_sides(self):
        frame = pd.DataFrame({"p_cover": [.56, .44, .54],
                              "cover_home": [1, 0, 1]})
        result = paper(frame)
        self.assertEqual((result["n"], result["wins"]), (2, 2))
        self.assertAlmostEqual(result["units_assuming_minus_110"], 20/11)


if __name__ == "__main__":
    unittest.main()
