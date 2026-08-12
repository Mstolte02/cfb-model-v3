import unittest

import numpy as np
import pandas as pd

from src.context import CONTEXT_FEATURES, OffsetLogit, _haversine_miles


class ContextTests(unittest.TestCase):
    def test_haversine_is_symmetric_and_zero_on_identity(self):
        self.assertAlmostEqual(_haversine_miles(40, -86, 40, -86), 0.0)
        a = _haversine_miles(40.0, -86.0, 34.0, -118.0)
        b = _haversine_miles(34.0, -118.0, 40.0, -86.0)
        self.assertAlmostEqual(a, b, places=10)

    def test_zero_context_preserves_base_probability(self):
        n = 40
        frame = pd.DataFrame({c: np.zeros(n) for c in CONTEXT_FEATURES})
        frame["p_dynamic"] = np.linspace(.2, .8, n)
        frame["y"] = (frame.p_dynamic > .5).astype(int)
        model = OffsetLogit(penalty=20).fit(frame)
        np.testing.assert_allclose(model.predict(frame), frame.p_dynamic, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
