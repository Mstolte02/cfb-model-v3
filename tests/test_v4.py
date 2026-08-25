import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import v4_backtest as BT
from scripts.train import projection_returning_raw
from src import v4 as V4
from src.data import pff, war
from war_model import player_production_forecast as PPF
from src.dynamic import WeeklyRatingState


def synthetic_frame():
    return pd.DataFrame({
        "O": [1.1, .2, -.7], "D": [.8, -.1, -.4],
        "talent": [.7, .1, -.8], "returning": [.2, -.3, .4],
        "off_success_rate": [1.0, -.2, -.6], "def_ppa": [.5, .1, -.7],
    }, index=["A", "B", "C"])


def synthetic_model():
    return V4.ReciprocalTeamModel(
        feature_names=["O", "D", "talent", "returning", "match_success"],
        coef=np.array([.5, .4, .25, .1, .08]), hfa_coef=.22,
        margin_coef=np.array([3.0, 2.5, 1.0, .5, .3]), margin_hfa=2.2,
        margin_sigma=15.0, ensemble_weight=.5, probability_scale=.9)


class ReciprocalInvariantTests(unittest.TestCase):
    def test_neutral_swap_is_exact_complement(self):
        frame, model = synthetic_frame(), synthetic_model()
        for home, away in [("A", "B"), ("A", "C"), ("B", "C")]:
            x = V4.matchup_vector(frame, home, away, model.feature_names)
            xr = V4.matchup_vector(frame, away, home, model.feature_names)
            np.testing.assert_allclose(xr, -x, atol=1e-14)
            self.assertAlmostEqual(model.win_prob(x, 0) + model.win_prob(xr, 0),
                                   1.0, places=13)
            self.assertAlmostEqual(model.pred_margin(x, 0) +
                                   model.pred_margin(xr, 0), 0.0, places=13)

    def test_average_strength_supports_interactions(self):
        value = synthetic_model().team_logit_strength(synthetic_frame(), "A")
        self.assertTrue(np.isfinite(value))


class WeeklyUpdateTests(unittest.TestCase):
    def test_same_week_order_does_not_change_predictions_or_state(self):
        frame, model = synthetic_frame(), synthetic_model()
        games = pd.DataFrame([
            {"week": 1, "home_team": "A", "away_team": "B",
             "neutral_site": False, "home_points": 31, "away_points": 20},
            {"week": 1, "home_team": "C", "away_team": "A",
             "neutral_site": True, "home_points": 17, "away_points": 24},
        ])
        one = WeeklyRatingState.initialize(model, frame, 2026, .2, .75)
        two = WeeklyRatingState.initialize(model, frame, 2026, .2, .75)
        p1 = one.update_week(model, frame, games).set_index(["home_team", "away_team"])
        p2 = two.update_week(model, frame, games.iloc[::-1]).set_index(
            ["home_team", "away_team"])
        np.testing.assert_allclose(p1.sort_index().p_home,
                                   p2.sort_index().p_home, atol=1e-14)
        for team in frame.index:
            self.assertAlmostEqual(one.ratings[team], two.ratings[team], places=13)


class TemporalFeatureTests(unittest.TestCase):
    def test_live_returning_uses_training_feature_definition(self):
        live = projection_returning_raw()
        self.assertEqual(len(live), 136)
        self.assertFalse(live.index.duplicated().any())
        # This snapshot is CFBD percentPPA, the same field returned by the historical
        # loader.  The older Connelly/ESPN file has Vanderbilt at 0.44 and must not be
        # silently substituted for the trained feature.
        self.assertAlmostEqual(float(live.loc["Vanderbilt"]), .354, places=6)

    def test_pff_entering_year_ignores_that_year_rows(self):
        base = pd.DataFrame([
            {"season": 2024, "team": "A", "group": "QB", "grade": 80., "snaps": 100},
            {"season": 2024, "team": "B", "group": "QB", "grade": 60., "snaps": 100},
            {"season": 2025, "team": "A", "group": "QB", "grade": 10., "snaps": 100},
            {"season": 2025, "team": "B", "group": "QB", "grade": 99., "snaps": 100},
        ])
        changed = base.copy()
        changed.loc[changed.season == 2025, "grade"] += 1000
        with patch.object(pff, "load_player_grades", return_value=base):
            first = pff.build_lagged_team_talent({"QB": 1.0})
        with patch.object(pff, "load_player_grades", return_value=changed):
            second = pff.build_lagged_team_talent({"QB": 1.0})
        pd.testing.assert_series_equal(first[2025], second[2025])

    def test_war_entering_year_ignores_that_year_rows(self):
        base = pd.DataFrame({"season": [2024, 2024, 2025, 2025],
                             "team": ["A", "B", "A", "B"],
                             "war": [2., 1., 0., 9.]})
        changed = base.copy()
        changed.loc[changed.season == 2025, "war"] += 1000
        index = {2025: pd.Index(["A", "B"])}
        with patch.object(war, "_load", return_value=base):
            first = war.lagged_team_talent(index)
        with patch.object(war, "_load", return_value=changed):
            second = war.lagged_team_talent(index)
        pd.testing.assert_series_equal(first[2025], second[2025])

    def test_player_production_lag_ignores_target_season_outcome(self):
        base = pd.DataFrame({
            "season": [2025], "target_season": [2025],
            "player_id": ["10"], "team": ["A"], "group": ["QB"],
            "player": ["Test Player"], "key": ["test player"],
        })
        history = pd.DataFrame({
            "season": [2024, 2025], "player_id": [10, 10],
            "player": ["Test Player", "Test Player"],
            "key": ["test player", "test player"], "team": ["A", "A"],
            "group": ["QB", "QB"],
            **{name: [100.0, 200.0] for name in PPF.MARKETS},
        })
        changed = history.copy()
        changed.loc[changed.season == 2025, list(PPF.MARKETS)] = 99999.0
        first = PPF.attach_market_targets(base, history)
        second = PPF.attach_market_targets(base, changed)
        lag_columns = [f"{name}_lag1" for name in PPF.MARKETS]
        pd.testing.assert_frame_equal(first[lag_columns], second[lag_columns])


class SelectionPolicyTests(unittest.TestCase):
    def test_tiny_extension_gain_is_rejected(self):
        scores = {name: .210 for name in BT.CANDIDATES}
        scores["clean_core"] = .2000
        scores["core_war_lag"] = .1998
        with patch.object(BT, "forward_score",
                          side_effect=lambda part, *_: scores[part]):
            selected, _ = BT.choose_candidate(
                {name: name for name in BT.CANDIDATES}, [2021, 2022])
        self.assertEqual(selected, "clean_core")

    def test_material_extension_gain_is_accepted(self):
        scores = {name: .210 for name in BT.CANDIDATES}
        scores["clean_core"] = .2000
        scores["core_war_lag"] = .1985
        with patch.object(BT, "forward_score",
                          side_effect=lambda part, *_: scores[part]):
            selected, _ = BT.choose_candidate(
                {name: name for name in BT.CANDIDATES}, [2021, 2022])
        self.assertEqual(selected, "core_war_lag")

    def test_research_candidate_map_uses_same_selection_policy(self):
        candidates = {"clean_core": ["O"], "new_component": ["O", "new"]}
        with patch.object(BT, "forward_score",
                          side_effect=lambda part, *_: {
                              "base": .2000, "new": .1989}[part]):
            selected, scores = BT.choose_candidate(
                {"clean_core": "base", "new_component": "new"},
                [2022, 2023], candidates)
        self.assertEqual(selected, "new_component")
        self.assertAlmostEqual(scores["new_component"], .1989)


if __name__ == "__main__":
    unittest.main()
