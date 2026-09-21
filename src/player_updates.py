"""Measured in-season player estimates; never a team-rating or WAR adjustment."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.qbwar import fit_season_values

POSITIONS = ('QB', 'RB', 'WR', 'TE')
COLUMNS = ['season', 'week', 'id', 'name', 'team', 'position', 'opponent', 'ppa']


def estimate_players(games: pd.DataFrame, season: int, through_week: int) -> list[dict]:
    """Use only previous-season priors and current games through the cutoff.

    Matches the audit's alpha=10 and minimum population (20 rows / 5 players).
    WR uses a personal prior throughout; QB only through week 4. Unknown IDs
    shrink to zero. No appearance is fabricated for injured/inactive players.
    """
    if not 0 <= through_week <= 15:
        raise ValueError('through_week must be between 0 and 15')
    games = games.copy()
    games['id'] = games['id'].astype(str)
    games = games[np.isfinite(games.ppa)]
    out = []
    for pos in POSITIONS:
        current = games[(games.season == season) & (games.week <= through_week)
                        & (games.position == pos)]
        if len(current) < 20 or current.id.nunique() < 5:
            continue
        previous = games[(games.season == season - 1) & (games.position == pos)]
        use_prior = pos == 'WR' or (pos == 'QB' and through_week < 5)
        prior = {}
        if use_prior and len(previous) >= 20 and previous.id.nunique() >= 5:
            values, _ = fit_season_values(previous, alpha=10.0)
            prior = values.set_index('id').qb_value.to_dict()
        values, _ = fit_season_values(current, prior=prior or None, alpha=10.0)
        info = current.sort_values('week', kind='stable').groupby('id').tail(1).set_index('id')
        for row in values.itertuples():
            p = info.loc[row.id]
            out.append(dict(id=row.id, name=p['name'], team=p.team, position=pos,
                            value=float(row.qb_value), games=int(row.n_games),
                            prior_value=prior.get(row.id),
                            shrinkage='personal' if row.id in prior else 'league',
                            last_week=int(p.week)))
    return sorted(out, key=lambda p: (p['team'], p['position'], p['id']))


def game_rows(payload, season, week, completed_ids, fbs):
    """Require a completed game ID: partial/live scores never enter a fit."""
    rows = []
    for p in payload:
        if str(p.get('gameId')) not in completed_ids or p.get('position') not in POSITIONS:
            continue
        value = (p.get('averagePPA') or {}).get('all')
        if p.get('id') is None or value is None or not np.isfinite(float(value)):
            continue
        if p.get('team') not in fbs or p.get('opponent') not in fbs:
            continue
        rows.append(dict(season=season, week=week, id=str(p['id']), name=p.get('name', ''),
                         team=p['team'], position=p['position'], opponent=p['opponent'],
                         ppa=float(value)))
    return rows
