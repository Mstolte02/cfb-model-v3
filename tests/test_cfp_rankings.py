import unittest
from scripts.update_cfp_rankings import parse

class CommitteeDataTests(unittest.TestCase):
    def payload(self, **changes):
        poll = {'shortName': 'CFP Rankings', 'date': '2026-11-03T08:00Z',
                'ranks': [{'current': i, 'team': {'location': f'T{i}'}} for i in range(1, 26)]}
        poll.update(changes)
        return {'rankings': [poll]}

    def test_real_release_cutoff(self):
        data = parse(self.payload(), 2026, {f'T{i}': {} for i in range(1,26)})
        self.assertEqual(data['games_through'], '2026-10-31')
        self.assertEqual(len(data['rankings']),25)

    def test_reject_ap_stale_missing_and_unknown(self):
        names = {f'T{i}': {} for i in range(1,26)}
        for payload, season, teams in [(self.payload(shortName='AP Poll'),2026,names),
                                      (self.payload(),2025,names),
                                      (self.payload(ranks=[]),2026,names),
                                      (self.payload(),2026,{})]:
            with self.assertRaises(ValueError):
                parse(payload,season,teams)

if __name__ == '__main__':
    unittest.main()
