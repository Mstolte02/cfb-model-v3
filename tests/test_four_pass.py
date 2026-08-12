import unittest

import numpy as np
import pandas as pd

from scripts.four_pass_initial_backtest import (
    FRAGILITY, FRAG_REV, REVERSIBILITY, STRUCTURAL,
    _add_structural_features, _direct_shape_features,
)
from src import v4 as V4


class FourPassReciprocityTests(unittest.TestCase):
    def test_shape_features_negate_when_teams_swap(self):
        sources = [
            "giveback", "lead_loss", "late_decline", "volatility",
            "comeback_gain", "comeback_win", "late_gain",
        ]
        row = {"home_team": "A", "away_team": "B"}
        for i, source in enumerate(sources, 1):
            row[f"home_{source}"] = float(i)
            row[f"away_{source}"] = float(i + 2)
        swapped = {"home_team": "B", "away_team": "A"}
        for source in sources:
            swapped[f"home_{source}"] = row[f"away_{source}"]
            swapped[f"away_{source}"] = row[f"home_{source}"]
        out = _direct_shape_features(pd.DataFrame([row, swapped]))
        columns = [*FRAGILITY, *REVERSIBILITY, *FRAG_REV]
        np.testing.assert_allclose(out.loc[1, columns], -out.loc[0, columns])

    def test_structural_features_negate_when_teams_swap(self):
        columns = {"O", "D"}
        for offense, defense in V4.MATCHUP_PAIRS.values():
            columns.update([offense, defense])
        frame = pd.DataFrame(index=["A", "B"], columns=sorted(columns), dtype=float)
        frame.loc["A"] = np.linspace(-1.0, 1.0, len(columns))
        frame.loc["B"] = np.linspace(.7, -.8, len(columns))
        games = pd.DataFrame([
            {"home_team": "A", "away_team": "B"},
            {"home_team": "B", "away_team": "A"},
        ])
        out = _add_structural_features(games, frame)
        np.testing.assert_allclose(out.loc[1, STRUCTURAL],
                                   -out.loc[0, STRUCTURAL], atol=1e-14)


if __name__ == "__main__":
    unittest.main()
