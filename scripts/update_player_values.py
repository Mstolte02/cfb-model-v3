"""Publish the supported player-prior research, without modifying team forecasts.

Run python -m scripts.update_player_values [--season 2026] [--through-week 3].
The default cutoff is the last fully completed scheduled week, capped at week 15.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import PROJECTION_YEAR, ROOT
from src.data import cfbd_client as c
from src.player_updates import COLUMNS, estimate_players, game_rows


def complete_week(schedule):
    weeks = sorted({int(g['week']) for g in schedule if g.get('week') is not None})
    cutoff = 0
    for week in weeks:
        if week > 15:
            break
        if not all(g.get('completed') for g in schedule if int(g.get('week', -1)) == week):
            break
        cutoff = week
    return cutoff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--season', type=int, default=PROJECTION_YEAR)
    parser.add_argument('--through-week', type=int)
    args = parser.parse_args()
    schedule = c.games(args.season, refresh=True)
    cutoff = complete_week(schedule) if args.through_week is None else args.through_week
    if not 0 <= cutoff <= 15:
        parser.error('--through-week must be between 0 and 15')
    rows = []
    for year in (args.season - 1, args.season):
        slate = schedule if year == args.season else c.games(year)
        fbs = {t['school'] for t in c.fbs_teams(year)}
        limit = cutoff if year == args.season else 15
        completed = {str(g['id']) for g in slate if g.get('completed') and g.get('week', 99) <= limit}
        weeks = sorted({int(g['week']) for g in slate if str(g['id']) in completed})
        for week in weeks:
            rows.extend(game_rows(c.player_ppa_games(year, week, refresh=year == args.season),
                                  year, week, completed, fbs))
    players = estimate_players(pd.DataFrame(rows, columns=COLUMNS), args.season, cutoff)
    if not players:
        raise RuntimeError('Insufficient completed player data; preserving the previous export')
    payload = dict(schema_version=1, season=args.season, through_week=cutoff,
                   generated_at=datetime.now(timezone.utc).isoformat(),
                   metric='Opponent-adjusted EPA/play above positional average',
                   scope='FBS vs FBS, regular season, QB/RB/WR/TE only',
                   affects_team_predictions=False, alpha=10.0, players=players)
    path = ROOT / 'viz' / 'data' / 'player_values.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
    print(f'Published {len(players)} player estimates through week {cutoff}: {path}')


if __name__ == '__main__':
    main()
