import unittest

import numpy as np
import pandas as pd

from scripts.export_viz import legacy_cache_compat
from src.v4 import ReciprocalTeamModel, matchup_vector


class LegacyCacheCompatibilityTests(unittest.TestCase):
    def test_six_slot_shim_preserves_v4_logit_and_margin(self):
        frame = pd.DataFrame({
            "O": [1.2, -.4, .1], "D": [.7, -.2, .3],
            "talent": [.8, -.6, .2], "returning": [-.1, .4, .5],
        }, index=["A", "B", "C"])
        model = ReciprocalTeamModel(
            feature_names=["O", "D", "talent", "returning"],
            coef=np.array([.35, .32, .38, .18]), hfa_coef=.29,
            margin_coef=np.array([3.9, 3.4, 5.0, 1.5]), margin_hfa=3.0,
            margin_sigma=17.7, ensemble_weight=1.0, probability_scale=1.0)
        compat = legacy_cache_compat(
            model, frame, {"coef": [1., 1., 1.], "intercept": 27.,
                           "alpha": 1., "resid_sd": 10.})
        for home in frame.index:
            for away in frame.index:
                if home == away:
                    continue
                a, b = compat["teams"][home], compat["teams"][away]
                old_x = np.array([a[0]-b[1], a[1]-b[0], a[2]-b[2],
                                  a[3]-b[3], a[4]-b[4], a[5]-b[5]])
                new_x = matchup_vector(frame, home, away, model.feature_names)
                old_logit = np.dot(compat["logistic"]["coef"], old_x)
                old_margin = np.dot(compat["margin"]["coef"], old_x)
                self.assertAlmostEqual(old_logit, model.raw_logit(new_x), places=3)
                self.assertAlmostEqual(old_margin, model.pred_margin(new_x), places=3)
                self.assertTrue(np.isfinite(old_logit))
                self.assertTrue(np.isfinite(old_margin))


if __name__ == "__main__":
    unittest.main()
