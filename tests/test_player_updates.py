import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.player_updates import estimate_players, game_rows
from scripts.update_player_values import complete_week


def fixture():
    return pd.DataFrame([dict(season=y, week=w, id=str(i), name=f'Player {i}',
                              team='New' if y == 2026 else 'Old', position=pos,
                              opponent=f'Opp {w % 3}', ppa=(i - 3) * .2 + w * .01)
                         for y in (2025, 2026) for pos in ('QB', 'WR', 'RB', 'TE')
                         for w in range(1, 9) for i in range(6)])


class PlayerUpdatesTest(unittest.TestCase):
    def test_policy_boundary(self):
        for week in (4, 5):
            result = estimate_players(fixture(), 2026, week)
            self.assertEqual(len(result), 24)
            for p in result:
                personal = p['position'] == 'WR' or (p['position'] == 'QB' and week == 4)
                self.assertEqual(p['shrinkage'], 'personal' if personal else 'league')
                self.assertEqual(p['games'], week)
                self.assertEqual(p['team'], 'New')

    def test_future_data_cannot_change_estimates(self):
        games = fixture()
        expected = estimate_players(games, 2026, 3)
        games.loc[(games.season == 2026) & (games.week > 3), 'ppa'] = 9999
        future = games.assign(season=2027, ppa=-9999)
        self.assertEqual(expected, estimate_players(pd.concat([games, future]), 2026, 3))

    def test_unseen_player_and_missing_prior(self):
        games = fixture()
        games.loc[(games.season == 2026) & (games.id == '0'), 'id'] = 'new'
        result = estimate_players(games, 2026, 4)
        self.assertTrue(all(p['shrinkage'] == 'league' for p in result if p['id'] == 'new'))
        result = estimate_players(games[games.season == 2026], 2026, 4)
        self.assertTrue(all(p['prior_value'] is None for p in result))

    def test_week_cut_and_empty(self):
        self.assertEqual(estimate_players(fixture(), 2026, 0), [])
        with self.assertRaises(ValueError):
            estimate_players(fixture(), 2026, 16)
        self.assertEqual(complete_week([dict(week=1, completed=True),
                                       dict(week=2, completed=False),
                                       dict(week=3, completed=True)]), 1)

    def test_completed_id_and_fbs_required(self):
        p = dict(gameId=1, id=2, position='WR', team='A', opponent='B', averagePPA={'all': .4})
        self.assertEqual(len(game_rows([p], 2026, 2, {'1'}, {'A', 'B'})), 1)
        self.assertEqual(game_rows([p], 2026, 2, {'2'}, {'A', 'B'}), [])
        self.assertEqual(game_rows([p], 2026, 2, {'1'}, {'A'}), [])

    def test_export_pipeline(self):
        from scripts import update_player_values as script
        def schedule(year, **kwargs):
            return [dict(id=w, week=w, completed=True) for w in range(1, 5)]
        def payload(year, week, **kwargs):
            return [dict(gameId=week, id=i, name=f'Player {i}', team='A',
                         opponent='B', position='WR', averagePPA={'all': i * .2})
                    for i in range(6)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'viz' / 'data').mkdir(parents=True)
            with patch.object(script, 'ROOT', root), patch.object(script.c, 'games', side_effect=schedule), patch.object(script.c, 'fbs_teams', return_value=[{'school': 'A'}, {'school': 'B'}]), patch.object(script.c, 'player_ppa_games', side_effect=payload), patch('sys.argv', ['update_player_values', '--season', '2026']):
                script.main()
            export = json.loads((root / 'viz/data/player_values.json').read_text())
            self.assertEqual(export['through_week'], 4)
            self.assertEqual(len(export['players']), 6)
            self.assertFalse(export['affects_team_predictions'])
            self.assertTrue(all(p['shrinkage'] == 'personal' for p in export['players']))

    def test_repeatable(self):
        self.assertEqual(estimate_players(fixture(), 2026, 3), estimate_players(fixture(), 2026, 3))


if __name__ == '__main__':
    unittest.main()
