import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import tempo


class TempoReciprocityTests(unittest.TestCase):
    def test_drive_matchup_features_negate_on_swap(self):
        home = {
            "off_spp": 23., "def_spp": 27., "plays_drive": 7.,
            "script_ypp": 6., "script_points": 4., "middle8_net": 2.,
            "q4_net": 1., "trail_win": .4, "lead_hold": .8,
            "pace_control": .3, "fast_win": .7, "slow_win": .4,
        }
        away = {
            "off_spp": 29., "def_spp": 25., "plays_drive": 5.,
            "script_ypp": 4., "script_points": 2., "middle8_net": -1.,
            "q4_net": -2., "trail_win": .2, "lead_hold": .6,
            "pace_control": -.1, "fast_win": .3, "slow_win": .7,
        }
        one = tempo._matchup_row((2025, 1, "A", "B"), home, away)
        two = tempo._matchup_row((2025, 1, "B", "A"), away, home)
        cols = [*tempo.PACE_IDENTITY, *tempo.SCRIPT_WINDOWS,
                *tempo.STATE_CONTROL, *tempo.PACE_MATCHUP]
        np.testing.assert_allclose([two[c] for c in cols],
                                   -np.asarray([one[c] for c in cols]), atol=1e-14)

    def test_quick_pressure_features_negate_on_swap(self):
        names = ["qb_quickness", "press_gen", "blitz_pg", "early_pass_rate",
                 "press_allowed", "qb_adot", "qb_pressure_sack",
                 "qb_positive_epa"]
        home = pd.Series(dict(zip(names, np.linspace(-1., 1., len(names)))))
        away = pd.Series(dict(zip(names, np.linspace(.8, -.7, len(names)))))
        one = tempo._quick_pressure_row(home, away)
        two = tempo._quick_pressure_row(away, home)
        np.testing.assert_allclose([two[c] for c in tempo.QUICK_PRESSURE],
                                   -np.asarray([one[c] for c in tempo.QUICK_PRESSURE]),
                                   atol=1e-14)

    def test_corrupt_drive_elapsed_is_excluded(self):
        valid = {"plays": 10, "elapsed": {"minutes": 4, "seconds": 0}}
        corrupt = {"plays": 3, "elapsed": {"minutes": 35, "seconds": 0}}
        seconds, plays = tempo._pace_totals([valid, corrupt])
        self.assertEqual((seconds, plays), (240., 10.))

    def test_same_week_games_share_start_of_week_history(self):
        def game(spp):
            return {
                "off_spp": spp, "opp_off_spp": 26.5, "plays_drive": 6.,
                "script_ypp": 5., "script_points": 3., "middle8_points": 0.,
                "middle8_net": 0., "q4_points": 0., "q4_net": 0.,
                "trail_ppd": 0., "lead_ppd": 0., "q4_trail_op": 0.,
                "q4_trail_win": 0., "q4_lead_op": 0., "q4_lead_win": 0.,
                "won": 1., "game_spp": spp,
            }

        metrics = {
            (2025, 1, "A", "B"): {"home": game(20.), "away": game(28.)},
            (2025, 1, "A", "C"): {"home": game(21.), "away": game(27.)},
            (2025, 2, "A", "D"): {"home": game(22.), "away": game(26.)},
        }
        with patch.object(tempo, "build_team_game_metrics", return_value=metrics):
            out = tempo.build_rolling_drive_features([2025]).set_index(
                ["week", "home_team", "away_team"])
        self.assertEqual(out.loc[(1, "A", "B"), "pace_off_diff"], 0.)
        self.assertEqual(out.loc[(1, "A", "C"), "pace_off_diff"], 0.)
        self.assertNotEqual(out.loc[(2, "A", "D"), "pace_off_diff"], 0.)


if __name__ == "__main__":
    unittest.main()
