import json
import re
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import totals as TT
from src.data import plays as PL


ROOT = Path(__file__).resolve().parents[1]


def frame():
    return pd.DataFrame({
        "O": [1.2, -.4], "D": [.7, -.2],
        "points_for": [34.0, 21.0], "points_against": [17.0, 31.0],
        "off_plays": [78.0, 64.0], "def_plays": [70.0, 75.0],
    }, index=["A", "B"])


class PointsDesignTests(unittest.TestCase):
    def test_side_row_matches_declared_feature_order(self):
        row = TT.side_row(frame().loc["A"], frame().loc["B"], 1.0)
        self.assertEqual(sorted(row), sorted(TT.POINTS_FEATURES))

    def test_side_row_reads_scorer_offense_and_opponent_defense(self):
        f = frame()
        row = TT.side_row(f.loc["A"], f.loc["B"], 1.0)
        self.assertEqual(row["O_scorer"], f.loc["A", "O"])
        self.assertEqual(row["D_opponent"], f.loc["B", "D"])
        self.assertEqual(row["pf_scorer"], f.loc["A", "points_for"])
        self.assertEqual(row["pa_opponent"], f.loc["B", "points_against"])

    def test_pace_interaction_is_level_scaled_by_relative_pace(self):
        f = frame()
        row = TT.side_row(f.loc["A"], f.loc["B"], 1.0)
        self.assertAlmostEqual(
            row["pf_x_pace"],
            f.loc["A", "points_for"] * f.loc["A", "off_plays"] /
            TT.DEFAULTS["off_plays"])

    def test_a_league_average_team_keeps_its_level_through_the_interaction(self):
        f = frame()
        f.loc["A", "off_plays"] = TT.DEFAULTS["off_plays"]
        row = TT.side_row(f.loc["A"], f.loc["B"], 1.0)
        self.assertAlmostEqual(row["pf_x_pace"], f.loc["A", "points_for"])

    def test_attach_defaults_a_team_with_no_prior_season_to_league_average(self):
        out = TT.attach(pd.DataFrame({"O": [.1], "D": [.2]}, index=["New"]),
                        pd.DataFrame(columns=TT.PROFILE_COLUMNS))
        for column in TT.PROFILE_COLUMNS:
            self.assertEqual(out.loc["New", column], TT.DEFAULTS[column])

    def test_home_flag_is_the_only_asymmetry_between_mirrored_sides(self):
        f = frame()
        home = TT.side_row(f.loc["A"], f.loc["B"], 1.0)
        away = TT.side_row(f.loc["A"], f.loc["B"], 0.0)
        differing = [k for k in home if home[k] != away[k]]
        self.assertEqual(differing, ["home"])


class JavascriptParityTests(unittest.TestCase):
    """The client recomputes the points design by hand; the two must not drift.

    A mismatch here is silent in production - the site would simply publish a
    different total from the one the backtest measured - so the feature order the
    exporter writes is pinned against the order `viz/app.js` indexes.
    """

    def test_app_js_builds_the_features_in_the_exported_order(self):
        source = (ROOT / "viz" / "app.js").read_text(encoding="utf-8")
        match = re.search(r"const x = \[(.*?)\];", source, re.S)
        self.assertIsNotNone(match, "sidePoints design vector not found in app.js")
        terms = [t.strip() for t in match.group(1).split(",")]
        # Nine terms in the same order as POINTS_FEATURES, with the two interaction
        # terms last and built from the team-input array rather than the z-scores.
        self.assertEqual(len(terms), len(TT.POINTS_FEATURES))
        self.assertEqual(terms[0], "S[0]")          # scorer offense
        self.assertEqual(terms[1], "O[1]")          # opponent defense
        self.assertEqual(terms[2], "isHome")
        self.assertEqual(terms[3], "si[0]")         # scorer points_for
        self.assertEqual(terms[4], "oi[1]")         # opponent points_against
        self.assertEqual(terms[5], "si[2]")         # scorer off_plays
        self.assertEqual(terms[6], "oi[3]")         # opponent def_plays
        self.assertIn("si[0] * si[2]", terms[7])
        self.assertIn("oi[1] * oi[3]", terms[8])

    def test_exported_payload_declares_the_same_features(self):
        path = ROOT / "viz" / "data" / "model_v4.json"
        if not path.exists():
            self.skipTest("model_v4.json not exported")
        payload = json.loads(path.read_text())
        block = payload.get("points_v2")
        self.assertIsNotNone(block, "points_v2 missing from the exported model")
        self.assertEqual(block["features"], TT.POINTS_FEATURES)
        self.assertEqual(block["team_input_order"], TT.PROFILE_COLUMNS)
        self.assertEqual(len(block["coef"]), len(TT.POINTS_FEATURES))


class PlayTypeVocabularyTests(unittest.TestCase):
    def test_relabelled_pass_and_punt_types_are_classified(self):
        # CFBD introduced these names in 2025 only; treating them as unknown
        # silently deleted 1,431 snaps from the holdout season.
        self.assertEqual(PL._kind("Pass Completion", ""), "pass")
        self.assertEqual(PL._kind("Pass Reception", ""), "pass")
        self.assertEqual(PL._kind("Punt Return", ""), "punt")
        self.assertEqual(PL._kind("Punt", ""), "punt")

    def test_sack_counts_as_a_dropback(self):
        self.assertEqual(PL._kind("Sack", ""), "pass")

    def test_fumble_rows_fall_back_to_play_text(self):
        self.assertEqual(PL._kind("Fumble", "Smith pass complete to Jones"), "pass")
        self.assertEqual(PL._kind("Fumble", "Smith run for 3 yards"), "rush")
        self.assertEqual(PL._kind("Fumble", ""), "other")

    def test_clock_conversion_counts_down_across_regulation(self):
        self.assertEqual(PL._seconds_left(1, {"minutes": 15, "seconds": 0}), 3600.0)
        self.assertEqual(PL._seconds_left(4, {"minutes": 0, "seconds": 0}), 0.0)
        self.assertEqual(PL._seconds_left(5, {"minutes": 15, "seconds": 0}), 0.0)


if __name__ == "__main__":
    unittest.main()
