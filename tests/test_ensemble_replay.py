"""The stdlib live runtime must reproduce the backtest's own arithmetic.

scripts/ensemble_replay.py is what the scheduled capture runs. These tests compare it
with the functions scripts/prior_decay_backtest.py scored the ensemble with - the
pandas EWMA form builder and the season walk - rather than with itself.
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import ensemble_replay as ER
from scripts import prior_decay_backtest as PD
from scripts.capture_market_snapshot import (form_missing_finals,
                                             freeze_weekly_model_snapshots,
                                             replay_published_results)

TEAMS = ["A", "B", "C", "D", "E", "F"]


def synthetic_season(seed=7, weeks=6):
    rng = np.random.default_rng(seed)
    games, stats = [], []
    for week in range(1, weeks + 1):
        order = rng.permutation(TEAMS)
        for home, away in zip(order[::2], order[1::2]):
            margin = int(rng.integers(-28, 29))
            games.append({"week": week, "home_team": home, "away_team": away,
                          "neutral_site": bool(rng.random() < .2), "margin": margin})
            for team, opp in ((home, away), (away, home)):
                row = {"week": week, "team": team, "opponent": opp}
                row.update({c: float(rng.normal()) for c in (*ER.OFF, *ER.DEF)})
                stats.append(row)
    return pd.DataFrame(games), pd.DataFrame(stats)


def synthetic_ensemble(seed=11):
    rng = np.random.default_rng(seed)
    members = []
    for index, half in enumerate((2.0, 4.0, None, 8.0)):
        members.append({
            "name": f"m{index}", "weight": .25,
            "columns": PD.ARM_COLUMNS["ewma_elo_nodecay"],
            "scale": rng.uniform(.5, 2.0, 5).tolist(),
            "coef": rng.normal(0, .6, 5).tolist(),
            "score_k": float(rng.choice(PD.K_GRID)), "current_halflife": half,
            "hfa_coef": float(rng.uniform(.1, .4)),
            "margin_sigma": float(rng.uniform(12, 17)),
            "initial": {t: float(v) for t, v in zip(TEAMS, rng.normal(0, .8, len(TEAMS)))},
        })
    return {"members": members, "min_form_games": 2,
            "margin_sigma": float(np.mean([m["margin_sigma"] for m in members]))}


class _StubModel:
    """Enough of ReciprocalTeamModel for PD.season_design."""

    def __init__(self, member):
        self.member = member
        self.hfa_coef = member["hfa_coef"]
        self.margin_sigma = member["margin_sigma"]

    def team_logit_strength(self, frame, team):
        return self.member["initial"][team]

    def win_prob(self, x, hfa):
        return .5


class FormParity(unittest.TestCase):
    def test_form_matches_backtest_todate_od(self):
        _, stats = synthetic_season()
        rows = stats.to_dict("records")
        for week in (2, 3, 5, 7):
            for half in (2.0, 4.0, math.inf):
                want = PD.todate_od(stats, float(week), half)
                got = ER.form_before(rows, week, half)
                self.assertEqual(set(got), set(want.index))
                for team, (o, d, n) in got.items():
                    self.assertAlmostEqual(o, want.at[team, "O"], places=12)
                    self.assertAlmostEqual(d, want.at[team, "D"], places=12)
                    self.assertEqual(n, int(want.at[team, "n"]))

    def test_no_form_before_first_week(self):
        _, stats = synthetic_season()
        self.assertIsNone(ER.form_before(stats.to_dict("records"), 1, 4.0))


class ReplayParity(unittest.TestCase):
    def test_replay_matches_backtest_season_walk(self):
        games, stats = synthetic_season()
        ensemble = synthetic_ensemble()
        finals = [{"week": g.week, "season_type": "regular", "home": g.home_team,
                   "away": g.away_team, "neutral": g.neutral_site,
                   "home_score": g.margin, "away_score": 0}
                  for g in games.itertuples()]
        events = ER.replay(ensemble, finals, stats.to_dict("records"))["events"]
        got = np.asarray([e["p_home"] for e in events])

        meta = games[["week", "home_team", "away_team", "neutral_site"]].reset_index(drop=True)
        home_flag = (~games.neutral_site).astype(float).to_numpy()
        part = (np.zeros((len(games), 1)), np.zeros(len(games)), home_flag,
                games.margin.to_numpy(float), meta)
        frame = pd.DataFrame(index=TEAMS)
        want = np.zeros(len(games))
        for member in ensemble["members"]:
            half = math.inf if member["current_halflife"] is None else member["current_halflife"]
            design = PD.season_design(_StubModel(member), frame, part, stats, half,
                                      PD.DEFAULT_PRIOR_HALFLIFE, member["score_k"])
            x = design[member["columns"]].to_numpy(float) / np.asarray(member["scale"])
            want += 1 / (1 + np.exp(-(x @ np.asarray(member["coef"]))))
        want /= len(ensemble["members"])
        # ER sorts by slate; the backtest keeps the original row order within weeks.
        order = np.argsort(meta.week.to_numpy(), kind="stable")
        np.testing.assert_allclose(got, want[order], rtol=0, atol=1e-12)

    def test_postseason_slates_sort_after_the_regular_season(self):
        self.assertGreater(ER.slate_key(1, "postseason"), ER.slate_key(15, "regular"))

    def test_neutral_probabilities_are_complements(self):
        _, stats = synthetic_season()
        ensemble = synthetic_ensemble()
        state = ER.replay(ensemble, [], stats.to_dict("records"))["state"]
        state["form"] = {m["name"]: ER.form_before(stats.to_dict("records"), 99,
                                                   4.0) for m in ensemble["members"]}
        p = ER.probability(ensemble, state, "A", "B", 0.0)
        q = ER.probability(ensemble, state, "B", "A", 0.0)
        self.assertAlmostEqual(p + q, 1.0, places=12)


class PublishedReplay(unittest.TestCase):
    def _write(self, root: Path):
        games, stats = synthetic_season(weeks=3)
        schedule = [{"id": i, "h": g.home_team, "a": g.away_team, "w": g.week,
                     "n": int(g.neutral_site), "f": 1, "hp": max(g.margin, 0),
                     "ap": max(-g.margin, 0), "st": "regular"}
                    for i, g in enumerate(games.itertuples(), 1)]
        schedule.append({"id": 99, "h": "A", "a": "B", "w": 4, "n": 0})
        (root / "schedule.json").write_text(json.dumps(schedule))
        model = {"features": ["x"], "teams": {t: [0.0] for t in TEAMS},
                 "logistic": {"coef": [0.0], "hfa": 0.0, "intercept": 0.0},
                 "margin": {"coef": [0.0], "hfa": 0.0, "intercept": 0.0, "sigma": 10.0},
                 "ens_w": 1.0, "probability_scale": 1.0,
                 "dynamic": {"blend": 1.0, "preseason_ratings": {t: 0.0 for t in TEAMS},
                             "ratings": {t: 0.0 for t in TEAMS}},
                 "ensemble": synthetic_ensemble()}
        (root / "model.json").write_text(json.dumps(model))
        (root / "ratings.json").write_text(json.dumps({"season": 2026, "teams": []}))
        payload = {"fields": list(ER.FORM_FIELDS),
                   "rows": stats[list(ER.FORM_FIELDS)].values.tolist()}
        (root / "form.json").write_text(json.dumps(payload))
        return schedule, model

    def test_replay_is_idempotent_and_publishes_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root)
            args = (root / "schedule.json", root / "model.json", root / "ratings.json",
                    None, None, root / "form.json")
            self.assertEqual(replay_published_results(*args), 9)
            first = ((root / "model.json").read_text(), (root / "ratings.json").read_text())
            self.assertEqual(replay_published_results(*args), 9)
            self.assertEqual(first, ((root / "model.json").read_text(),
                                     (root / "ratings.json").read_text()))
            model = json.loads(first[0])
            ratings = json.loads(first[1])
            self.assertEqual(model["ensemble"]["state"]["completed_games"], 9)
            self.assertEqual([h["label"] for h in ratings["history"]],
                             ["Preseason", "Week 1", "Week 2", "Week 3"])
            self.assertEqual(len(ratings["game_history"]), 9)
            # The frozen v4 blocks are untouched by the v5 replay.
            self.assertEqual(model["dynamic"]["ratings"], {t: 0.0 for t in TEAMS})

    def test_upcoming_snapshot_uses_start_of_week_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root)
            args = (root / "schedule.json", root / "model.json", root / "ratings.json",
                    None, None, root / "form.json")
            replay_published_results(*args)
            model = json.loads((root / "model.json").read_text())
            ratings = json.loads((root / "ratings.json").read_text())
            odds = {"weekly": [
                {"id": 99, "week": 4, "start": "2026-10-01T00:00:00Z", "home": "A",
                 "away": "B", "books": {"DraftKings": {"spread": -3}}},
                {"id": 98, "week": 2, "start": "2026-09-12T00:00:00Z", "home": "A",
                 "away": "C", "books": {"DraftKings": {"spread": -3}}}]}
            self.assertEqual(freeze_weekly_model_snapshots(
                odds, model, ratings, "2026-09-30T00:00:00Z",
                form_path=root / "form.json"), 2)
            week4, week2 = (odds["weekly"][0]["modelSnapshot"],
                            odds["weekly"][1]["modelSnapshot"])
            # Week 4 starts after every final, so it reads the published state.
            current = ER.probability(model["ensemble"], model["ensemble"]["state"],
                                     "A", "B", 1.0)
            self.assertAlmostEqual(week4["homeWinProbability"], current, places=6)
            # A week-2 row replays only week 1: ratings and form from before week 2.
            names = list(model["ensemble"]["members"][0]["initial"])
            rows = ER.form_rows(json.loads((root / "form.json").read_text()), names)
            finals = [{**e, "season_type": "regular"} for e in ratings["game_history"]]
            state = ER.replay(model["ensemble"], finals, rows, stop_before=2)["state"]
            self.assertAlmostEqual(week2["homeWinProbability"], ER.probability(
                model["ensemble"], state, "A", "C", 1.0), places=12)
            self.assertAlmostEqual(week2["homeMargin"], model["ensemble"]["margin_sigma"]
                                   * __import__("statistics").NormalDist().inv_cdf(
                                       week2["homeWinProbability"]), places=6)

class FormRefreshRule(unittest.TestCase):
    """The advanced-stats pull spends a monthly budget, so it runs only when needed."""

    def _payload(self, root, rows):
        path = root / "form.json"
        path.write_text(json.dumps({"fields": list(ER.FORM_FIELDS), "rows": rows}))
        return path

    def test_only_recent_uncovered_finals_trigger_a_pull(self):
        from datetime import datetime, timezone
        now = datetime(2026, 9, 22, 12, tzinfo=timezone.utc)
        stat = [0.1] * (len(ER.OFF) + len(ER.DEF))
        games = [
            {"id": 1, "completed": True, "week": 3, "homeTeam": "A", "awayTeam": "B",
             "startDate": "2026-09-19T20:00:00Z"},                     # covered
            {"id": 2, "completed": True, "week": 3, "homeTeam": "C", "awayTeam": "D",
             "startDate": "2026-09-19T23:00:00Z"},                     # missing, recent
            {"id": 3, "completed": True, "week": 1, "homeTeam": "E", "awayTeam": "F",
             "startDate": "2026-08-30T20:00:00Z"},                     # missing, old
            {"id": 4, "completed": False, "week": 4, "homeTeam": "A", "awayTeam": "C",
             "startDate": "2026-09-26T20:00:00Z"},                     # not played
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._payload(Path(tmp), [[3, "A", "B", *stat], [3, "B", "A", *stat]])
            self.assertEqual(form_missing_finals(games, now, path), [2])
            path = self._payload(Path(tmp), [[3, "A", "B", *stat], [3, "C", "D", *stat]])
            self.assertEqual(form_missing_finals(games, now, path), [])
            self.assertEqual(form_missing_finals(games, now, Path(tmp) / "absent.json"),
                             [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
