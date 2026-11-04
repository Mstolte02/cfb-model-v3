"""Refresh actual CFP rankings from ESPN; never replace a valid snapshot with AP/proxy data."""
import json
import sys
import urllib.request
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://site.api.espn.com/apis/site/v2/sports/football/college-football/rankings'

def parse(payload, season, names):
    polls = [p for p in payload.get('rankings', []) if p.get('shortName') == 'CFP Rankings']
    if not polls:
        raise ValueError('No actual CFP poll available')
    poll = polls[0]
    released = date.fromisoformat(poll['date'][:10])
    if released.year != season or released > date.today():
        raise ValueError('CFP release is not current-season published data')
    aliases = {'Miami (FL)': 'Miami', 'Mississippi': 'Ole Miss', 'Southern California': 'USC'}
    rankings = []
    for row in poll['ranks']:
        name = row['team']['location']
        name = aliases.get(name, name)
        if name not in names:
            raise ValueError(f'Unknown team: {name}')
        rankings.append({'rank': int(row['current']), 'team': name})
    if len(rankings) != 25 or {r['rank'] for r in rankings} != set(range(1, 26)) or len({r['team'] for r in rankings}) != 25:
        raise ValueError('Incomplete or duplicate CFP rankings')
    # Weekly Tuesday release evaluates games through the preceding Saturday.
    through = released - timedelta(days=(released.weekday() - 5) % 7)
    return {'season': season, 'released': released.isoformat(), 'games_through': through.isoformat(),
            'source': 'CFP Selection Committee via ESPN', 'source_url': 'https://collegefootballplayoff.com/rankings',
            'provider_url': URL, 'rankings': sorted(rankings, key=lambda r: r['rank'])}

def main():
    try:
        season = json.loads((ROOT/'viz/data/ratings.json').read_text())['season']
        names = json.loads((ROOT/'viz/data/teams.json').read_text())
        with urllib.request.urlopen(URL, timeout=30) as response:
            data = parse(json.load(response), season, names)
        path = ROOT/'viz/data/cfp_rankings.json'
        if path.exists() and json.loads(path.read_text()).get('released', '') > data['released']:
            raise ValueError('Provider would roll back the CFP release')
        path.write_text(json.dumps(data, indent=2)+'\n')
        print(f"CFP: {len(data['rankings'])} teams, released {data['released']}")
    except Exception as exc:
        print(f'CFP refresh unavailable; retaining last verified snapshot: {exc}', file=sys.stderr)
        if '--strict' in sys.argv:
            raise

if __name__ == '__main__':
    main()
