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


if __name__ == "__main__":
    unittest.main()
