import json
import tempfile
import unittest
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from scripts.capture_market_snapshot import (fetch_cfbd, flatten, implied, latest_quotes,
                                             model_probability,
                                             moneyline_research_candidate, publish_finals,
                                             quote_key, quote_payload_hash, quote_value,
                                             replay_published_results,
                                             update_weekly_board, weekly_payload)
from war_model.materialize_availability import current_rows


class MarketTrackingTests(unittest.TestCase):
    def test_newcomer_week_zero_games_are_excluded_from_bets(self):
        rows = weekly_payload({
            (401864577, "Book"): {"game_id": 401864577, "provider": "Book",
                "week": 1, "start": "2026-08-29T21:30:00Z",
                "home": "North Dakota State", "away": "Jacksonville State"},
            (401866408, "Book"): {"game_id": 401866408, "provider": "Book",
                "week": 1, "start": "2026-08-29T22:30:00Z",
                "home": "Eastern Michigan", "away": "Sacramento State"},
            (99, "Book"): {"game_id": 99, "provider": "Book", "week": 1,
                "start": "2026-09-05T00:00:00Z", "home": "A", "away": "B"},
        })
        by_id = {row["id"]: row for row in rows}
        self.assertTrue(by_id[401864577]["bettingExcluded"])
        self.assertTrue(by_id[401866408]["bettingExcluded"])
        self.assertNotIn("bettingExcluded", by_id[99])

    def test_published_ratings_cover_both_2026_fbs_newcomers(self):
        root = Path(__file__).resolve().parents[1]
        ratings = json.loads((root / "viz/data/ratings.json").read_text())
        model = json.loads((root / "viz/data/model_v4.json").read_text())
        rated = {row["team"] for row in ratings["teams"]}
        newcomers = {"Sacramento State", "North Dakota State"}
        self.assertEqual(rated, set(model["teams"]))
        self.assertTrue(newcomers <= rated)
        for snapshot in ratings.get("history", []):
            self.assertTrue(newcomers <= {row["team"] for row in snapshot["teams"]})

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

    def test_duplicate_provider_aliases_are_collapsed_before_hashing(self):
        raw = [{"id": 1, "week": 1, "startDate": "2026-09-01T00:00:00Z",
                "homeTeam": "A", "awayTeam": "B", "lines": [
                    {"provider": "Draft Kings", "spread": None,
                     "overUnder": 52.5, "homeMoneyline": None,
                     "awayMoneyline": 130},
                    {"provider": "DraftKings", "spread": -3.5,
                     "overUnder": 52.5, "homeMoneyline": -150,
                     "awayMoneyline": 130},
                ]}]
        rows = flatten(raw, "2026-08-01T00:00:00Z")
        self.assertEqual(len(rows), 1)
        self.assertEqual(quote_key(rows[0]), (1, "DraftKings"))
        self.assertEqual(rows[0]["spread"], -3.5)
        self.assertEqual(len(quote_payload_hash(rows)), 64)

    def test_payload_hash_handles_duplicate_keys_with_null_and_numeric_values(self):
        rows = [
            {"game_id": 1, "provider": "DraftKings", "spread": None},
            {"game_id": 1, "provider": "DraftKings", "spread": -3.5},
        ]
        self.assertEqual(quote_payload_hash(rows), quote_payload_hash(list(reversed(rows))))

    def test_cfbd_fetch_retries_transient_server_error(self):
        error = urllib.error.HTTPError("https://example.test", 502, "Bad Gateway", {}, None)
        with patch("scripts.capture_market_snapshot._fetch_cfbd_once",
                   side_effect=[error, [{"id": 1}]]) as request, \
             patch("scripts.capture_market_snapshot.time.sleep") as sleep:
            self.assertEqual(fetch_cfbd("key", "/lines", {"year": 2026}), [{"id": 1}])
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_weekly_board_stays_frozen_between_monday_locks(self):
        old = [{"id": 1, "books": {"Book": {"spread": -3}}}]
        odds = {"weekly": old, "sources": {"cfbd_lines": {"as_of": "old"}}}
        quotes = {(1, "Book"): {"game_id": 1, "provider": "Book", "week": 1,
            "start": "2026-09-05T00:00:00Z", "home": "A", "away": "B",
            "spread": -7}}
        self.assertFalse(update_weekly_board(
            odds, quotes, "2026-09-03T16:30:00Z", False))
        self.assertEqual(odds["weekly"], old)
        self.assertEqual(odds["sources"]["cfbd_lines"]["as_of"], "old")
        self.assertEqual(odds["weekly_lock"]["locked_at"], "old")
        self.assertEqual(odds["weekly_lock"]["reason"], "pre-lock baseline")

    def test_monday_lock_replaces_board_and_records_timestamp(self):
        odds = {"weekly": [{"id": 99}], "sources": {}}
        quotes = {(1, "Book"): {"game_id": 1, "provider": "Book", "week": 1,
            "start": "2026-09-05T00:00:00Z", "home": "A", "away": "B",
            "spread": -7}}
        self.assertTrue(update_weekly_board(
            odds, quotes, "2026-09-07T16:30:00Z", True))
        self.assertEqual([row["id"] for row in odds["weekly"]], [1])
        self.assertEqual(odds["weekly_lock"]["locked_at"], "2026-09-07T16:30:00Z")
        self.assertEqual(odds["weekly_lock"]["cadence"], "Monday 12:30 PM ET")

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

    def test_moneyline_candidate_must_be_more_likely_than_not(self):
        self.assertFalse(moneyline_research_candidate(.48, .25))
        self.assertFalse(moneyline_research_candidate(.50, .25))
        self.assertTrue(moneyline_research_candidate(.51, .30))

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
            teams_path = root / "teams.json"
            playoff_path = root / "playoff.json"
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
            ]}))
            teams_path.write_text(json.dumps({"C": {"conference": "New League"}}))
            playoff_path.write_text(json.dumps({"teams": [{
                "team": "C", "conference": "New League", "avg_wins": 4.5,
                "avg_losses": 7.5, "conf_champ": .01, "playoff": 0.0,
            }]}))

            self.assertEqual(replay_published_results(
                schedule, model_path, ratings_path, teams_path, playoff_path), 1)
            first_model = model_path.read_text()
            first_ratings = ratings_path.read_text()
            payload = json.loads(first_ratings)
            self.assertEqual([s["label"] for s in payload["history"]],
                             ["Preseason", "Week 1 to date"])
            self.assertEqual(payload["history"][-1]["completed_games"], 1)
            self.assertEqual(len(payload["teams"]), 3)
            newcomer = next(row for row in payload["teams"] if row["team"] == "C")
            self.assertEqual(newcomer["conference"], "New League")
            self.assertEqual(newcomer["avg_wins"], 4.5)
            self.assertGreater(json.loads(first_model)["dynamic"]["ratings"]["C"], -1.0)
            self.assertAlmostEqual(
                json.loads(first_model)["dynamic"]["ratings"]["C"], 4.0)
            self.assertEqual(json.loads(first_model)["dynamic"]["update_rule"],
                             "robust_margin_residual_v1")

            self.assertEqual(replay_published_results(
                schedule, model_path, ratings_path, teams_path, playoff_path), 1)
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
