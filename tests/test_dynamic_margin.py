"""The published margin and the published win probability describe one model.

They did not always. The probability came from the static/dynamic blend and the margin
from the preseason ridge, so with `dynamic_blend = 1.0` they were two models printed
side by side and disagreed about the winner on 15.3% of 2022-25 games. These tests pin
the property that made that possible, so it cannot come back quietly.
"""
import unittest

import numpy as np
from scipy.special import ndtri

from src import v4 as V4
from src.dynamic import MARGIN_P_CLIP, WeeklyRatingState, implied_margin

from tests.test_v4 import synthetic_frame, synthetic_model


def state_with(model, frame, ratings, blend=1.0):
    state = WeeklyRatingState.initialize(model, frame, season=2026,
                                         dynamic_blend=blend)
    state.ratings.update(ratings)
    return state


class ImpliedMarginTests(unittest.TestCase):
    def test_inverts_the_link_the_ensemble_reads_a_margin_with(self):
        """sigma * Phi^-1(Phi(m / sigma)) is m, for any margin the sport produces."""
        sigma = 16.5
        for margin in (-38.0, -12.5, -0.4, 0.0, 3.5, 21.0, 45.0):
            p = float(V4._norm_cdf(margin / sigma))
            self.assertAlmostEqual(implied_margin(p, sigma), margin, places=6)

    def test_is_antisymmetric(self):
        for p in (.02, .3, .5, .61, .97):
            self.assertAlmostEqual(implied_margin(p, 17.0),
                                   -implied_margin(1 - p, 17.0), places=12)

    def test_even_money_is_a_pick(self):
        self.assertAlmostEqual(implied_margin(.5, 17.0), 0.0, places=12)

    def test_certainty_is_capped_rather_than_infinite(self):
        capped = 17.0 * float(ndtri(1 - MARGIN_P_CLIP))
        self.assertAlmostEqual(implied_margin(1.0, 17.0), capped, places=9)
        self.assertAlmostEqual(implied_margin(0.0, 17.0), -capped, places=9)
        self.assertTrue(np.isfinite(implied_margin(1.0, 17.0)))


class PublishedPredictionTests(unittest.TestCase):
    def setUp(self):
        self.frame, self.model = synthetic_frame(), synthetic_model()

    def test_margin_and_probability_never_name_different_winners(self):
        """The defect itself: 58% for one team beside a spread on the other.

        The rating gap is set well past anything the static model would produce, which
        is exactly the situation the old code got wrong.
        """
        for gap in (-3.0, -1.2, -.4, 0.0, .4, 1.2, 3.0):
            state = state_with(self.model, self.frame, {"A": gap, "B": 0.0, "C": 0.0})
            pred = state.predict(self.model, self.frame, "A", "B")
            self.assertGreaterEqual((pred["p_home"] - .5) * pred["pred_margin"], 0.0,
                                    f"probability and margin disagree at gap {gap}")

    def test_margin_follows_the_in_season_rating_not_the_preseason_ridge(self):
        flat = state_with(self.model, self.frame, {"A": 0.0, "B": 0.0, "C": 0.0})
        moved = state_with(self.model, self.frame, {"A": 1.5, "B": 0.0, "C": 0.0})
        before = flat.predict(self.model, self.frame, "A", "B")
        after = moved.predict(self.model, self.frame, "A", "B")
        self.assertGreater(after["pred_margin"], before["pred_margin"] + 1.0)
        # The preseason number is still reported, and is precisely what did not move.
        self.assertAlmostEqual(after["pred_margin_static"],
                               before["pred_margin_static"], places=12)

    def test_margin_equals_the_link_applied_to_the_published_probability(self):
        state = state_with(self.model, self.frame, {"A": .8, "B": -.3, "C": 0.0},
                           blend=.75)
        pred = state.predict(self.model, self.frame, "A", "B")
        self.assertAlmostEqual(
            pred["pred_margin"],
            implied_margin(pred["p_home"], self.model.margin_sigma), places=12)

    def test_neutral_swap_negates_the_margin_exactly(self):
        state = state_with(self.model, self.frame, {"A": .8, "B": -.3, "C": 0.0})
        one = state.predict(self.model, self.frame, "A", "B", neutral_site=True)
        other = state.predict(self.model, self.frame, "B", "A", neutral_site=True)
        self.assertAlmostEqual(one["pred_margin"], -other["pred_margin"], places=12)


if __name__ == "__main__":
    unittest.main()
