import unittest
from unittest.mock import patch

import pandas as pd

from scripts.coach_effects import coach_spells, graph_components, mover_count
from src import features
from src.data import coaches


class CoachAttributionTests(unittest.TestCase):
    def test_dominant_coach_uses_games_and_preserves_change_flag(self):
        raw = pd.DataFrame([
            {"season": 2025, "team_id": 1, "team": "A", "coach_id": 10,
             "coach_name": "Full", "games": 9.0},
            {"season": 2025, "team_id": 1, "team": "A", "coach_id": 11,
             "coach_name": "Interim", "games": 4.0},
        ])
        with patch.object(coaches, "load_coach_seasons", return_value=raw):
            selected = coaches.dominant_head_coaches(2025, 2025)
        self.assertEqual(int(selected.iloc[0].coach_id), 10)
        self.assertTrue(bool(selected.iloc[0].midseason_change))
        self.assertAlmostEqual(float(selected.iloc[0].coach_share), 9 / 13)

    def test_preseason_assignment_keeps_incumbent_without_using_games(self):
        raw = pd.DataFrame([
            {"season": 2024, "team_id": 1, "team": "A", "coach_id": 10,
             "coach_name": "Incumbent", "games": 12., "hire_date": "2020-01-01"},
            {"season": 2025, "team_id": 1, "team": "A", "coach_id": 10,
             "coach_name": "Incumbent", "games": 3., "hire_date": "2020-01-01"},
            {"season": 2025, "team_id": 1, "team": "A", "coach_id": 11,
             "coach_name": "Interim", "games": 9., "hire_date": "2025-09-20"},
        ])
        membership = pd.DataFrame([
            {"season": year, "team_id": 1, "team": "A"}
            for year in (2024, 2025)])
        with (patch.object(coaches, "load_coach_seasons", return_value=raw),
              patch.object(coaches, "_fbs_membership", return_value=membership)):
            selected = coaches.preseason_head_coaches(2024, 2025)
        row = selected[selected.season == 2025].iloc[0]
        self.assertEqual(int(row.coach_id), 10)
        self.assertEqual(row.assignment_rule, "returning_incumbent")
        self.assertEqual(int(row.hc_change), 0)

    def test_target_season_outcome_cannot_change_coach_feature(self):
        outcomes = pd.DataFrame([
            {"season": season, "team_id": team_id, "coach_id": coach_id,
             "rating_overall": value, "rating_offense": value / 2,
             "rating_defense": value / 2, "prior_offense": 0.,
             "prior_defense": 0., "talent": 0., "returning": 0.}
            for season, team_id, coach_id, value in
            [(2021, 1, 10, 1.), (2021, 2, 20, -1.),
             (2022, 1, 10, 100.), (2022, 2, 20, -100.)]
        ])
        assignment = pd.DataFrame([
            {"season": 2022, "team": "A", "coach_id": 10,
             "prior_coach_id": 10, "hc_change": 0, "hc_tenure_year": 2,
             "hc_first_year": 0},
            {"season": 2022, "team": "B", "coach_id": 20,
             "prior_coach_id": 20, "hc_change": 0, "hc_tenure_year": 2,
             "hc_first_year": 0},
        ])
        first = features.leakage_safe_coach_features(outcomes, assignment, [2022])[2022]
        changed = outcomes.copy()
        changed.loc[changed.season == 2022, "rating_overall"] *= 1000
        second = features.leakage_safe_coach_features(changed, assignment, [2022])[2022]
        pd.testing.assert_series_equal(first.hc_prior_effect,
                                       second.hc_prior_effect)
        self.assertEqual(first.attrs["max_outcome_season"], 2021)


class MoverGraphTests(unittest.TestCase):
    def test_spells_split_on_missing_year(self):
        frame = pd.DataFrame([
            {"coach_id": 1, "coach_name": "C", "team_id": 10,
             "team": "A", "season": year}
            for year in (2018, 2019, 2021)
        ])
        spells = coach_spells(frame)
        self.assertEqual(spells.seasons.tolist(), [2, 1])

    def test_mover_threshold_and_connected_component(self):
        rows = []
        for team_id, team, years in [(10, "A", (2018, 2019)),
                                     (11, "B", (2020, 2021))]:
            rows.extend({"coach_id": 1, "coach_name": "Mover",
                         "team_id": team_id, "team": team, "season": year}
                        for year in years)
        rows.append({"coach_id": 2, "coach_name": "Other", "team_id": 11,
                     "team": "B", "season": 2022})
        frame = pd.DataFrame(rows)
        self.assertEqual(mover_count(frame, 2), 1)
        self.assertEqual(mover_count(frame, 3), 0)
        components, _ = graph_components(frame)
        self.assertEqual(len(components), 1)
        self.assertEqual(len(components[0]), 4)  # two coaches, two schools


if __name__ == "__main__":
    unittest.main()
