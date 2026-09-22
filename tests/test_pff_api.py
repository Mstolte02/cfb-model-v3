from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.data.pff_api import PffApiError, PffClient


class Response:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code = status
        self._payload = payload or {}
        self.headers = headers or {}
        self.ok = 200 <= status < 300

    def json(self):
        return self._payload


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def passing_row():
    return {"player_id": 1, "player": "Test QB", "position": "QB",
            "team_name": "TEST", "player_game_count": 1,
            "grades_offense": 70.0, "passing_snaps": 20}


class PffApiTests(unittest.TestCase):
    def test_facet_report_authenticates_and_validates_schema(self):
        session = Session([Response(payload={"passing_summary": [passing_row()]},
                                    headers={"x-ratelimit-limit": "60",
                                             "x-ratelimit-remaining": "59"})])
        client = PffClient("secret", session=session)
        frame = client.facet_report("passing", 2025)

        self.assertEqual(list(frame.player), ["Test QB"])
        url, request = session.calls[0]
        self.assertTrue(url.endswith("/v1/facet/passing/summary"))
        self.assertEqual(request["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(request["params"],
                         {"league": "ncaa", "season": 2025,
                          "division": "fbs"})
        self.assertEqual(client.rate_limit.remaining, 59)

    def test_missing_schema_fails_before_a_bad_export_is_written(self):
        session = Session([
            Response(payload={"passing_summary": [{"player_id": 1}]})])
        client = PffClient("secret", session=session)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(PffApiError, "missing required columns"):
                client.sync_facet_report("passing", 2025, path)
            self.assertEqual(list(path.iterdir()), [])

    def test_sync_refuses_to_replace_without_force(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "passing_2025.csv"
            destination.write_text("sentinel\n", encoding="utf-8")
            client = PffClient("secret", session=Session([]))
            with self.assertRaises(FileExistsError):
                client.sync_facet_report("passing", 2025, destination.parent)
            self.assertEqual(destination.read_text(encoding="utf-8"), "sentinel\n")

    def test_v2_team_stats_are_normalized_to_snake_case(self):
        payload = {"rows": [{"teamId": 1, "epaPerPlay": 0.2}]}
        client = PffClient("secret", session=Session([Response(payload=payload)]))
        frame = client.team_stats(2025, "offense-overall-success")
        self.assertEqual(list(frame.columns), ["team_id", "epa_per_play"])

    def test_team_directory_is_normalized(self):
        payload = {"rows": [{"franchiseId": 260, "city": "Ohio State",
                              "groupIds": "11;18"}]}
        client = PffClient("secret", session=Session([Response(payload=payload)]))
        frame = client.team_directory(2025)
        self.assertEqual(frame.to_dict("records"),
                         [{"franchise_id": 260, "city": "Ohio State",
                           "group_ids": "11;18"}])

    def test_position_report_can_filter_and_canonicalize_fbs(self):
        report = {"rows": [
            {"playerId": 1, "player": "A", "position": "T",
             "franchiseId": 260, "teamAbbreviation": "OSU", "gamesPlayed": 2},
            {"playerId": 2, "player": "B", "position": "T",
             "franchiseId": 999, "teamAbbreviation": "LOW", "gamesPlayed": 2},
        ]}
        directory = {"rows": [
            {"franchiseId": 260, "city": "Ohio State", "groupIds": "11;18"},
            {"franchiseId": 999, "city": "Lower", "groupIds": "13"},
        ]}
        client = PffClient(
            "secret", session=Session([Response(payload=report),
                                       Response(payload=directory)]))
        frame = client.position_report(2025, "pass-blocking", division="fbs")
        self.assertEqual(frame[["player_id", "team_name", "player_game_count"]]
                         .to_dict("records"),
                         [{"player_id": 1, "team_name": "Ohio State",
                           "player_game_count": 2}])

    def test_weekly_team_stats_request_pregame_weeks_and_resume(self):
        from scripts.sync_pff_api import pregame_week_ids, sync_team_stats_weekly
        self.assertEqual(pregame_week_ids(2024, 1), [0, 1])
        self.assertEqual(pregame_week_ids(2024, 14), list(range(15)))
        # 2021-2023 conference championships (PFF week 17) are CFBD week 14.
        self.assertEqual(pregame_week_ids(2023, 14), [*range(15), 17])
        payload = {"weekIds": "0,1", "rows": [{"teamId": 1, "name": "A",
                                               "epaPerPassPlay": 0.1}]}
        session = Session([Response(payload=payload)])
        client = PffClient("secret", session=session)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "team_stats_weekly" / "offense_passing_2024_thru_w02.csv"
            existing.parent.mkdir()
            existing.write_text("sentinel\n", encoding="utf-8")
            result = sync_team_stats_weekly(client, [2024], root,
                                            categories=["offense-passing"],
                                            first_week=2, last_week=3, pause=0)
            self.assertEqual([p.name for p in result["written"]],
                             ["offense_passing_2024_thru_w01.csv"])
            self.assertEqual([p.name for p in result["skipped"]], [existing.name])
            self.assertEqual(existing.read_text(encoding="utf-8"), "sentinel\n")
            self.assertEqual(len(session.calls), 1)
            self.assertEqual(session.calls[0][1]["params"],
                             {"season": 2024, "category": "offense-passing",
                              "weekIds": "0,1"})
            written = (root / "team_stats_weekly" /
                       "offense_passing_2024_thru_w01.csv").read_text(encoding="utf-8")
            self.assertTrue(written.startswith("team_id,name,epa_per_pass_play"))

    def test_errors_never_include_the_credential(self):
        payload = {"error": {"code": "forbidden",
                             "message": "do-not-print is not entitled",
                             "request_id": "request-1"}}
        client = PffClient(
            "do-not-print", session=Session([Response(403, payload)]))
        with self.assertRaises(PffApiError) as caught:
            client.whoami()
        self.assertNotIn("do-not-print", str(caught.exception))
        self.assertIn("request-1", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
