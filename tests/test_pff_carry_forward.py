import copy
import json
import unittest
from pathlib import Path
from scripts import ensemble_replay as ER

ROOT = Path(__file__).resolve().parents[1]

class CarryForward(unittest.TestCase):
    def setUp(self):
        self.payload = {'cutoffs': {'3': {'A': [1, 2]}, '5': {'A': [3, 4]},
                                    '7': {'A': [99, 99]}},
                        'fetched_at': {'5': '2026-10-06T15:37:56Z'}}

    def test_latest_eligible_without_same_week_or_future(self):
        self.assertEqual(ER.pff_table(self.payload, 7), {'A': [3, 4]})
        self.assertEqual(ER.pff_table(self.payload, 5), {'A': [1, 2]})
        self.assertEqual(ER.pff_table(self.payload, 8), {'A': [99, 99]})
        self.assertIsNone(ER.pff_table(self.payload, 3))
        self.assertIsNone(ER.pff_table(None, 7))
        self.assertEqual(ER.pff_table(self.payload, 1), {})

    def test_empty_new_cut_does_not_discard_last_usable_table(self):
        self.payload['cutoffs']['6'] = {}
        self.assertEqual(ER.pff_table(self.payload, 7), {'A': [3, 4]})
        self.assertEqual(ER.pff_cutoff(self.payload, 7), 5)

    def test_policy_boundary_and_provenance_survive_compaction(self):
        e = {'pff_carry_forward_from_slate': 7}
        self.assertIsNone(ER.pff_state(e, self.payload, 5)['pff'])
        s = {'ratings': {}, 'form': {}, 'war': {}, 'slate': 7,
             **ER.pff_state(e, self.payload, 7)}
        out = ER.compact_state(s)
        self.assertEqual(out['pff_cutoff'], 5)
        self.assertTrue(out['pff_carried_forward'])
        self.assertEqual(out['pff_fetched_at'], self.payload['fetched_at']['5'])
        self.assertEqual(out['slate'], 7)

    def test_live_replay_preserves_all_old_game_predictions(self):
        e = json.loads((ROOT/'viz/data/model_v4.json').read_text())['ensemble']
        schedule = json.loads((ROOT/'viz/data/schedule.json').read_text())
        names = set(e['members'][0]['initial'])
        finals = [{'id': g['id'], 'week': g['w'], 'season_type': g.get('st', 'regular'),
                   'home': g['h'], 'away': g['a'], 'neutral': bool(g.get('n')),
                   'home_score': g['hp'], 'away_score': g['ap']}
                  for g in schedule if g.get('f') and g['h'] in names and g['a'] in names]
        rows = ER.form_rows(json.loads((ROOT/'data/live/game_advanced_2026.json').read_text()), names)
        pff = json.loads((ROOT/'data/live/pff_form_2026.json').read_text())
        war = json.loads((ROOT/'data/live/inseason_war_team_2026.json').read_text())
        legacy = copy.deepcopy(e); legacy['pff_carry_forward_from_slate'] = 10000
        before = ER.replay(legacy, finals, rows, stop_before=7, pff=pff, war=war)
        after = ER.replay(e, finals, rows, stop_before=7, pff=pff, war=war)
        self.assertEqual(before['events'], after['events'])
        self.assertIsNotNone(after['state']['pff'])
        self.assertEqual(after['state']['pff_cutoff'], ER.pff_cutoff(pff, 7))
        for m in e['members']:
            self.assertIn('war_delta_diff', ER._stack(m, after['state']['pff'], after['state']['war'], 7)[0])

if __name__ == '__main__': unittest.main()
