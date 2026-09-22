from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.data import pff_team


class PffTeamTests(unittest.TestCase):
    def _cache(self, root: Path):
        folder = root / "team_stats"
        folder.mkdir()
        pd.DataFrame([
            {"franchise_id": 1, "city": "Appalachian State"},
            {"franchise_id": 2, "city": "USF"},
        ]).to_csv(folder / "team_directory_2024.csv", index=False)
        for category in pff_team.CATEGORY_PREFIX:
            pd.DataFrame([
                {"team_id": 1, "metric": 1.0, "metric_rank": 2},
                {"team_id": 2, "metric": 3.0, "metric_rank": 1},
            ]).to_csv(folder / f"{category}_2024.csv", index=False)

    def test_load_maps_names_drops_ranks_and_standardizes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._cache(root)
            frame = pff_team.load_season(2024, root)
        self.assertEqual(set(frame.index), {"App State", "South Florida"})
        self.assertFalse(any(column.endswith("_rank") for column in frame))
        self.assertAlmostEqual(float(frame.mean().abs().max()), 0.0)

    def test_attach_uses_neutral_zero_and_records_coverage(self):
        base = pd.DataFrame({"O": [1.0, -1.0]}, index=["A", "B"])
        features = pd.DataFrame({"pff_x": [0.5]}, index=["A"])
        result = pff_team.attach(base, features)
        self.assertEqual(result.loc["B", "pff_x"], 0.0)
        self.assertEqual(result.attrs["pff_team_coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()
