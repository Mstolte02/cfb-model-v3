import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from scripts import update_inseason_war as refresh
from scripts import war_inseason_backtest as backtest


class RefreshHistoryTests(unittest.TestCase):
    def test_refresh_preserves_published_pregame_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            team = Path(tmp) / 'team.json'
            players = Path(tmp) / 'players.json'
            old = {'3': {'A': .1}, '6': {'A': .2}, '8': {'A': .3}}
            team.write_text(json.dumps({'cutoffs': old}))
            players.write_text(json.dumps({'team_availability': {'A': -.05}}))
            priors = pd.DataFrame({'group': ['QB'], 'mu': [0.]})
            deltas = pd.DataFrame({'team': ['A'], 'D': [.5]})
            with patch.object(refresh.IW, 'team_payload_path', return_value=team), \
                 patch.object(refresh.IW, 'OUT', players), \
                 patch.object(backtest, 'history', return_value=pd.DataFrame()), \
                 patch.object(backtest, 'priors', return_value=priors), \
                 patch.object(refresh.IW, 'window_rows', return_value=pd.DataFrame()) as rows, \
                 patch.object(refresh.IW, 'team_deltas', return_value=deltas):
                refresh.write_team_tables({'rule': {}, 'k': {}}, 10, False, True)
            got = json.loads(team.read_text())
            self.assertEqual(got['through_week'], 10)
            self.assertEqual({k: got['cutoffs'][k] for k in old}, old)
            self.assertNotIn('9', got['cutoffs'])
            self.assertEqual(got['cutoffs']['10'], {'A': .45})
            self.assertEqual(rows.call_count, 1)
            self.assertEqual(rows.call_args.args[1], 10)


if __name__ == '__main__':
    unittest.main()
