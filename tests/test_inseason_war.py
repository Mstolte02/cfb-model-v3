import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from config import ROOT
from src import inseason_war as IW


class InseasonWarTests(unittest.TestCase):
    def test_failed_refresh_keeps_existing_window_and_published_payload(self):
        from scripts import sync_pff_war_windows as SW
        from scripts import update_inseason_war as U

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            window = root / "2026_w01-04"
            window.mkdir()
            cached = window / "rushing.csv"
            cached.write_text("verified old report")
            params = root / "params.json"
            params.write_text("{}")
            published = root / "players_inseason.json"
            published.write_text("verified old WAR")
            with patch.object(SW, "WINDOW_DIR", root), \
                 patch.object(SW, "main", return_value=None), \
                 patch.object(U, "ensure_key", return_value=None), \
                 patch.object(IW, "PARAMS", params), \
                 patch.object(IW, "OUT", published):
                with self.assertRaisesRegex(SystemExit, "PFF refresh incomplete"):
                    U.main(week=4, refit=False, skip_pull=False)
            self.assertEqual(cached.read_text(), "verified old report")
            self.assertEqual(published.read_text(), "verified old WAR")

    def test_pick_cut_nearest_with_ties_to_earlier(self):
        g = {"3": {}, "6": {}, "9": {}}
        self.assertEqual(IW.pick_cut(g, 1), "3")
        self.assertEqual(IW.pick_cut(g, 4), "3")
        self.assertEqual(IW.pick_cut(g, 5), "6")
        self.assertEqual(IW.pick_cut(g, 7), "6")      # 7 is nearer 6 than 9
        self.assertEqual(IW.pick_cut(g, 12), "9")

    def test_new_fbs_teams_map_without_touching_production_map(self):
        tm = json.load(open(ROOT / "war_model" / "team_map.json"))
        self.assertNotIn("SACRAMENTO", tm)            # production map unchanged
        self.assertEqual(IW.canonical_team(["SACRAMENTO"], tm), "Sacramento State")
        self.assertEqual(IW.canonical_team(["N DAK ST"], tm), "North Dakota State")
        self.assertEqual(IW.canonical_team(["AIR FORCE", "Air Force"], tm), "Air Force")

    def test_fit_rule_recovers_a_known_sample_weighted_update(self):
        # Truth: each player's season-to-date counts by his own precision, so thin
        # records (large P) and big samples (many snaps) move further.
        rng = np.random.default_rng(0)
        n = 6000
        m = rng.normal(0, 1, n)
        P = rng.choice([.05, .5, 2.0], n)
        snaps = rng.choice([30.0, 150.0, 400.0], n)
        obs = rng.normal(0, 1, n)
        K = P / (P + 30.0 / snaps)
        target = m + K * (obs - m) + rng.normal(0, .05, n)
        fr = pd.DataFrame({"group": "QB", "cut": 3, "m": m, "obs": obs, "P": P,
                           "snaps_w": snaps, "target": target, "snaps_t": 100.0})
        c = IW.fit_rule(fr)["QB"]["3"]
        self.assertAlmostEqual(c["g"], 1.0, delta=0.15)
        self.assertAlmostEqual(c["b"], 1.0, delta=0.1)
        # The rule moves a thin-record player further than a veteran on the same data.
        thin = IW.updated_rate(c, 0.0, 2.0, 1.0, 150.0)
        known = IW.updated_rate(c, 0.0, 0.05, 1.0, 150.0)
        self.assertGreater(thin - c["a"], known - c["a"])

    def test_published_payload_shape(self):
        path = IW.OUT
        if not path.exists():
            self.skipTest("players_inseason.json not built")
        d = json.loads(path.read_text())
        self.assertEqual(d["season"], 2026)
        self.assertGreaterEqual(d["through_week"], 1)
        row = next(iter(next(iter(d["players"].values())).values()))
        rows = [r for team in d["players"].values() for r in team.values()]
        for r in rows:   # "inj" appears only for players on an injury report
            self.assertEqual(set(r) - {"inj", "new", "g", "al"}, {"sn", "war", "d", "st", "out", "sh", "base"})

    def test_availability_delta_does_not_double_count_base_absence(self):
        roster = pd.DataFrame([
            {"team": "A", "player": "New Injury", "proj_war": .12, "available": True},
            {"team": "A", "player": "Already Out", "proj_war": 0, "available": False},
        ])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "availability.csv"
            path.write_text("team,player,status,note\nA,New Injury,out,x\n"
                            "A,Already Out,out,x\n")
            self.assertEqual(IW.availability_team_deltas(roster, path),
                             {"A": -.12 / IW.SEASON_GAMES})

    def test_availability_delta_is_per_week_and_scales_by_report_status(self):
        # D is WAR per week; a season proj_war must be spread over the season, or
        # one injured starter would outweigh every in-season performance change.
        roster = pd.DataFrame([
            {"team": "A", "player": "Questionable Guy", "proj_war": .6, "available": True},
            {"team": "A", "player": "Doubtful Guy", "proj_war": .4, "available": True},
            {"team": "B", "player": "Healthy Guy", "proj_war": 1.0, "available": True},
        ])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "availability.csv"
            path.write_text("team,player,status,note\nA,Questionable Guy,questionable,x\n"
                            "A,Doubtful Guy,doubtful,x\n")
            got = IW.availability_team_deltas(roster, path)
        self.assertEqual(set(got), {"A"})
        self.assertAlmostEqual(got["A"], -(.5 * .6 + .75 * .4) / IW.SEASON_GAMES)

    def test_injury_report_status_reads_headline_not_prose(self):
        from scripts import sync_injury_reports as S
        self.assertEqual(S.norm_status("Out"), "out")
        self.assertEqual(S.norm_status("IR - Knee"), "out")
        self.assertEqual(S.norm_status("Questionable - Undisclosed"), "questionable")
        self.assertIsNone(S.norm_status("Without a timetable"))
        # Two sources that disagree average to the status between them.
        self.assertEqual(S.status_of_share((S.SHARE["out"] + S.SHARE["questionable"]) / 2),
                         "doubtful")
        self.assertIsNone(S.status_of_share(1.0))

    def test_current_starters_follow_usage_and_remove_absences(self):
        frame = pd.DataFrame([
            {"team": "A", "broad_group": "QB", "is_starter": True,
             "available_now": False, "snaps": 80, "proj_war": .2},
            {"team": "A", "broad_group": "QB", "is_starter": False,
             "available_now": True, "snaps": 60, "proj_war": .1},
        ])
        self.assertEqual(IW.current_starters(frame).tolist(), [False, True])


class WarRuntimeTests(unittest.TestCase):
    """scripts/ensemble_replay's v5.2 WAR column: cut choice and stack fallback."""

    def setUp(self):
        from scripts import ensemble_replay as ER
        self.ER = ER
        self.payload = {"cutoffs": {"3": {"A": .02, "B": -.01}, "6": {"A": .03}}}

    def test_war_table_reads_latest_cut_strictly_before_the_slate(self):
        wt = self.ER.war_table
        self.assertIsNone(wt(None, 5))
        self.assertEqual(wt(self.payload, 1), {})      # before the first cut: zeros
        self.assertEqual(wt(self.payload, 3), {})      # cut 3 is not before week 3
        self.assertEqual(wt(self.payload, 4), {"A": .02, "B": -.01})
        self.assertEqual(wt(self.payload, 7), {"A": .03})
        self.assertEqual(wt(self.payload, self.ER.POSTSEASON_OFFSET + 1), {"A": .03})

    def test_current_week_cut_applies_to_next_week_only(self):
        payload = {"cutoffs": {"3": {"A": .02}, "5": {"A": -.04}}}
        self.assertEqual(self.ER.war_table(payload, 5), {"A": .02})
        self.assertEqual(self.ER.war_table(payload, 6), {"A": -.04})

    def _member(self):
        base = ["prior_level", "elo_change", "dO", "dD", "hfa"]
        return {"initial": {"A": .5, "B": .1},
                "columns": base, "scale": [1] * 5, "coef": [1, 1, 0, 0, .1],
                "stack_pff": {"columns": base + ["pff_O_diff", "pff_D_diff"],
                              "scale": [1] * 7, "coef": [1, 1, 0, 0, .1, .2, .2]},
                "stack_war": {"columns": base + ["pff_O_diff", "pff_D_diff",
                                                 "war_delta_diff"],
                              "scale": [1] * 8, "coef": [1, 1, 0, 0, .1, .2, .2, 5.0]}}

    def test_stack_fallback_and_war_effect(self):
        ER, m = self.ER, self._member()
        ratings = {"A": .5, "B": .1}
        pff = {"A": [.1, 0], "B": [0, 0]}
        war = {"A": .02, "B": -.01}
        base = ER.member_probability(m, ratings, None, "A", "B", 1.0)
        p_pff = ER.member_probability(m, ratings, None, "A", "B", 1.0, pff)
        p_nowar = ER.member_probability(m, ratings, None, "A", "B", 1.0, pff, None)
        p_war = ER.member_probability(m, ratings, None, "A", "B", 1.0, pff, war)
        p_zero = ER.member_probability(m, ratings, None, "A", "B", 1.0, pff, {})
        self.assertEqual(p_pff, p_nowar)              # no WAR payload = exactly v5.1
        self.assertNotEqual(base, p_pff)
        self.assertGreater(p_war, p_zero)             # A's WAR rose, B's fell
        # WAR without PFF never switches stacks: base stack
        self.assertEqual(ER.member_probability(m, ratings, None, "A", "B", 1.0, None, war),
                         base)


if __name__ == "__main__":
    unittest.main()
