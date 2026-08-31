import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from scripts.capture_market_snapshot import (flatten, implied, latest_quotes,
                                             model_probability, publish_finals,
                                             quote_key, quote_value,
                                             replay_published_results)
from war_model.materialize_availability import current_rows


class MarketTrackingTests(unittest.TestCase):
    def test_quote_change_identity_ignores_capture_time(self):
        raw = [{"id": 1, "week": 1, "startDate": "2026-09-01T00:00:00Z",
                "homeTeam": "A", "awayTeam": "B", "lines": [{
                    "provider": "Draft Kings", "spread": -3,
                    "homeMoneyline": -150, "awayMoneyline": 130}]}]
        a = flatten(raw, "2026-08-01T00:00:00Z")[0]
        b = flatten(raw, "2026-08-01T06:00:00Z")[0]
        self.assertEqual(quote_key(a), (1, "DraftKings"))
        self.assertEqual(quote_value(a), quote_value(b))
        self.assertEqual(latest_quotes([a, b])[(1, "DraftKings")]["captured_at"],
                         "2026-08-01T06:00:00Z")

    def test_removed_quote_tombstone_does_not_remain_current(self):
        row = {"game_id": 1, "provider": "Bovada", "captured_at": "2026-08-01T00:00:00Z"}
        row.update({k: None for k in ("spread", "spreadOpen", "overUnder",
                                      "overUnderOpen", "homeMoneyline", "awayMoneyline")})
        live = {**row, "spread": -3}
        gone = {**row, "captured_at": "2026-08-02T00:00:00Z", "removed": True}
        self.assertEqual(latest_quotes([live, gone]), {})
        self.assertTrue(latest_quotes([live, gone], include_removed=True)[
            (1, "Bovada")]["removed"])

    def test_american_implied_probability(self):
        self.assertAlmostEqual(implied(200), 1/3)
        self.assertAlmostEqual(implied(-200), 2/3)

    def test_model_probability_is_reciprocal_on_neutral_field(self):
        model = {"teams": {"A": [1.0], "B": [-1.0]},
                 "logistic": {"coef": [.5], "hfa": .2, "intercept": 0},
                 "margin": {"coef": [3.0], "hfa": 2.0, "intercept": 0, "sigma": 10},
                 "ens_w": .5, "probability_scale": 1.0, "dynamic": {}}
        p = model_probability(model, "A", "B", True)
        q = model_probability(model, "B", "A", True)
        self.assertAlmostEqual(p+q, 1.0, places=12)

    def test_publish_finals_only_writes_completed_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            schedule = Path(tmp) / "schedule.json"
            schedule.write_text(json.dumps([
                {"id": 1, "h": "UNLV", "a": "Memphis"},
                {"id": 2, "h": "A", "a": "B"},
            ]))
            changed = publish_finals([
                {"id": 1, "completed": True, "homePoints": 21, "awayPoints": 27},
                {"id": 2, "completed": False, "homePoints": 7, "awayPoints": 3},
            ], schedule)
            rows = json.loads(schedule.read_text())
            self.assertEqual(changed, 1)
            self.assertEqual(rows[0], {
                "id": 1, "h": "UNLV", "a": "Memphis", "f": 1, "hp": 21, "ap": 27})
            self.assertNotIn("f", rows[1])

    def test_completed_results_replay_is_idempotent_and_exports_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            schedule = root / "schedule.json"
            model_path = root / "model.json"
            ratings_path = root / "ratings.json"
            schedule.write_text(json.dumps([
                {"h": "A", "a": "C", "w": 1, "n": 0, "f": 1, "hp": 7, "ap": 35},
                {"h": "B", "a": "A", "w": 1, "n": 0},
            ]))
            model_path.write_text(json.dumps({
                "features": ["x"], "teams": {"A": [1.0], "B": [0.0], "C": [-1.0]},
                "logistic": {"coef": [1.0], "hfa": 0.2, "intercept": 0.0},
                "margin": {"coef": [4.0], "hfa": 2.0, "intercept": 0.0, "sigma": 10.0},
                "ens_w": 0.5, "probability_scale": 1.0,
                "dynamic": {"blend": 1.0, "k": 2.0,
                            "ratings": {"A": 1.0, "B": 0.0, "C": -1.0}},
            }))
            ratings_path.write_text(json.dumps({"season": 2026, "teams": [
                {"rank": 1, "team": "A", "power": .7, "vs_average": .7},
                {"rank": 2, "team": "B", "power": .5, "vs_average": .5},
                {"rank": 3, "team": "C", "power": .3, "vs_average": .3},
            ]}))

            self.assertEqual(replay_published_results(
                schedule, model_path, ratings_path), 1)
            first_model = model_path.read_text()
            first_ratings = ratings_path.read_text()
            payload = json.loads(first_ratings)
            self.assertEqual([s["label"] for s in payload["history"]],
                             ["Preseason", "Week 1 to date"])
            self.assertEqual(payload["history"][-1]["completed_games"], 1)
            self.assertGreater(json.loads(first_model)["dynamic"]["ratings"]["C"], -1.0)

            self.assertEqual(replay_published_results(
                schedule, model_path, ratings_path), 1)
            self.assertEqual(model_path.read_text(), first_model)
            self.assertEqual(ratings_path.read_text(), first_ratings)

    def test_availability_events_materialize_latest_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.csv"
            path.write_text("event_id,observed_at,team,player,status,note\n"
                            "1,2026-08-01T00:00:00Z,A,P,out,hurt\n"
                            "2,2026-08-02T00:00:00Z,A,P,clear,returned\n")
            self.assertEqual(current_rows(path), [])


if __name__ == "__main__":
    unittest.main()
