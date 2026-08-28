import unittest

import numpy as np
import pandas as pd

from src import facet_matchup as FM
from src import v4 as V4


def _probe_frame(columns):
    frame = pd.DataFrame(index=["A", "B"], columns=sorted(columns), dtype=float)
    frame.loc["A"] = np.linspace(-1.3, 1.1, len(columns))
    frame.loc["B"] = np.linspace(.9, -1.6, len(columns))
    return frame


class UnitTaxonomyTests(unittest.TestCase):
    def test_every_cell_belongs_to_exactly_one_unit(self):
        seen = []
        for _, cells in FM.UNITS.values():
            seen.extend(cells)
        self.assertEqual(len(seen), len(set(seen)))

    def test_every_facet_carries_a_concept(self):
        mapping = FM.concept_map()
        raw = pd.read_csv(FM.PLAYER_FACET_WAR, nrows=1)
        facets = [c for c in raw.columns if c not in FM.META]
        missing = [f for f in facets if f not in mapping]
        self.assertEqual(missing, [])

    def test_units_partition_team_war(self):
        """The split has to be exact, or a unit model is not a model of WAR."""
        table = FM.realized_unit_war()
        self.assertAlmostEqual(table.attrs["coverage_share"], 1.0, places=9)

    def test_predeclared_pairs_name_real_units(self):
        for offense, defense in FM.UNIT_PAIRS.values():
            self.assertIn(offense, FM.OFF_UNITS)
            self.assertIn(defense, FM.DEF_UNITS)
        for offense, defense in FM.GROUP_PAIRS.values():
            self.assertIn(offense, FM.OFF_GROUPS)
            self.assertIn(defense, FM.DEF_GROUPS)


class ReciprocityTests(unittest.TestCase):
    """A matchup term that survives a team swap is not a matchup term.

    The v4 win probability is a complement and the margin an opposite only because
    every feature negates. These crosses are registered into the same vector, so the
    invariant is theirs to keep too.
    """

    def _assert_antisymmetric(self, names, prefix, pairs):
        original = dict(V4.MATCHUP_PAIRS)
        try:
            FM.register_pairs(pairs, prefix)
            columns = {f"{prefix}{u}" for pair in pairs.values() for u in pair}
            frame = _probe_frame(columns)
            forward = V4.matchup_vector(frame, "A", "B", names)
            reverse = V4.matchup_vector(frame, "B", "A", names)
            np.testing.assert_allclose(reverse, -forward, atol=1e-14)
        finally:
            V4.MATCHUP_PAIRS.clear()
            V4.MATCHUP_PAIRS.update(original)

    def test_predeclared_unit_crosses_negate(self):
        self._assert_antisymmetric(list(FM.UNIT_PAIRS), FM.UNIT_PREFIX, FM.UNIT_PAIRS)

    def test_predeclared_room_crosses_negate(self):
        self._assert_antisymmetric(list(FM.GROUP_PAIRS), FM.GROUP_PREFIX,
                                   FM.GROUP_PAIRS)

    def test_full_scan_crosses_negate(self):
        pairs = FM.all_unit_pairs()
        self._assert_antisymmetric(list(pairs), FM.UNIT_PREFIX, pairs)

    def test_level_units_produce_no_edge(self):
        """Two identical rooms are not a mismatch, whatever the level."""
        original = dict(V4.MATCHUP_PAIRS)
        try:
            FM.register_unit_pairs()
            columns = {f"{FM.UNIT_PREFIX}{u}" for u in FM.UNITS}
            frame = pd.DataFrame(2.5, index=["A", "B"], columns=sorted(columns),
                                 dtype=float)
            vector = V4.matchup_vector(frame, "A", "B", list(FM.UNIT_PAIRS))
            np.testing.assert_allclose(vector, 0.0, atol=1e-14)
        finally:
            V4.MATCHUP_PAIRS.clear()
            V4.MATCHUP_PAIRS.update(original)


class AttachTests(unittest.TestCase):
    def test_missing_source_is_neutral_and_recorded(self):
        frame = pd.DataFrame({"O": [1.0, 2.0]}, index=["A", "B"])
        FM.attach(frame, None, None)
        for column in FM.unit_columns() + FM.group_columns():
            self.assertIn(column, frame.columns)
            self.assertTrue((frame[column] == 0.0).all())
        self.assertEqual(frame.attrs[f"{FM.UNIT_PREFIX}coverage"], 0.0)

    def test_absent_team_is_neutral_not_dropped(self):
        frame = pd.DataFrame({"O": [1.0, 2.0]}, index=["A", "B"])
        units = pd.DataFrame(1.0, index=["A"], columns=list(FM.UNITS))
        FM.attach(frame, units, None)
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.loc["B", f"{FM.UNIT_PREFIX}o_pass_qb"], 0.0)
        self.assertAlmostEqual(frame.attrs[f"{FM.UNIT_PREFIX}coverage"], .5)


if __name__ == "__main__":
    unittest.main()
