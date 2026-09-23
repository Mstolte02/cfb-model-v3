import json
import unittest

import numpy as np
import pandas as pd

from config import ROOT
from src import inseason_war as IW


class InseasonWarTests(unittest.TestCase):
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

    def test_fit_rule_recovers_a_known_blend(self):
        rng = np.random.default_rng(0)
        n = 4000
        m = rng.normal(0, 1, n)
        obs = rng.normal(0, 1, n)
        target = 0.7 * m + 0.3 * obs + rng.normal(0, .05, n)
        fr = pd.DataFrame({"group": "QB", "cut": 3, "m": m, "obs": obs,
                           "target": target, "snaps_t": 100.0})
        c = IW.fit_rule(fr)["QB"]["3"]
        self.assertAlmostEqual(c["lam"], 0.3, delta=0.05)
        self.assertAlmostEqual(c["b"], 1.0, delta=0.1)

    def test_published_payload_shape(self):
        path = IW.OUT
        if not path.exists():
            self.skipTest("players_inseason.json not built")
        d = json.loads(path.read_text())
        self.assertEqual(d["season"], 2026)
        self.assertGreaterEqual(d["through_week"], 1)
        row = next(iter(next(iter(d["players"].values())).values()))
        self.assertEqual(set(row), {"sn", "war", "d"})


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
