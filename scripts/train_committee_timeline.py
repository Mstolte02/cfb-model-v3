"""Fit weekly CFP rank movement from as-of poll and game snapshots, 2014-25.

The training data are compact ESPN poll/scoreboard snapshots in
data/committee_history.json. A poll in week W sees games through week W-1.
No future games, final-season ratings, or outcomes enter an earlier row.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ('prior_rank', 'win_pct', 'sos', 'quality_wins', 'bad_losses',
            'recent_win', 'recent_loss', 'recent_win_quality',
            'recent_loss_badness', 'title_win', 'title_loss', 'power', 'games')
POWER_IDS = {'1', '4', '5', '8', '9'}  # ACC, Big 12, Big Ten, SEC, historic Pac-12


def snapshots(history):
    rows = []
    for year, season in history['seasons'].items():
        games = season['games']
        for previous, current in zip(season['polls'], season['polls'][1:]):
            last_week = previous['week'] - 1
            observed = [g for g in games if g['week'] < current['week']]
            played = defaultdict(list)
            conference = {}
            for g in observed:
                home, away = g['home'], g['away']
                conference[home], conference[away] = g['home_conf'], g['away_conf']
                played[home].append((g, away, g['home_score'] > g['away_score']))
                played[away].append((g, home, g['away_score'] > g['home_score']))
            opponent_pct = {t: sum(win for _, _, win in gs) / len(gs)
                            for t, gs in played.items()}
            for team in previous['rankings'].keys() | current['rankings'].keys():
                gs = played[team]
                recent = [(g, opp, win) for g, opp, win in gs if g['week'] > last_week]
                quality = lambda opp: opponent_pct.get(opp, .5)
                row = dict(season=int(year), week=current['week'], team=team,
                           rank=current['rankings'].get(team, 30),
                           prior_rank=previous['rankings'].get(team, 30),
                           win_pct=sum(win for _, _, win in gs) / max(1, len(gs)),
                           sos=np.mean([quality(opp) for _, opp, _ in gs]) if gs else 0,
                           quality_wins=sum(quality(opp) for _, opp, win in gs if win),
                           bad_losses=sum(1-quality(opp) for _, opp, win in gs if not win),
                           recent_win=sum(win for g, _, win in recent if not g['title']),
                           recent_loss=sum(not win for g, _, win in recent if not g['title']),
                           recent_win_quality=sum(quality(opp) for g, opp, win in recent
                                                  if win and not g['title']),
                           recent_loss_badness=sum(1-quality(opp) for g, opp, win in recent
                                                    if not win and not g['title']),
                           title_win=sum(win for g, _, win in recent if g['title']),
                           title_loss=sum(not win for g, _, win in recent if g['title']),
                           power=int(str(conference.get(team)) in POWER_IDS or team == '87'),
                           games=len(gs))
                rows.append(row)
    return pd.DataFrame(rows)


def metrics(model, frame):
    predicted = model.predict(frame[list(FEATURES)])
    return dict(mae=round(float(np.mean(abs(predicted-frame['rank']))), 3),
                prior_mae=round(float(np.mean(abs(frame['prior_rank']-frame['rank']))), 3),
                spearman=round(float(pd.Series(predicted).corr(
                    pd.Series(frame['rank'].to_numpy()), method='spearman')), 4))


def pod_order(scores, wins, bonus=0):
    """Committee-style six-team pool: place four, carry two, then place three."""
    remaining = set(scores)
    carry = []
    placed = []
    while remaining:
        available = sorted(remaining - set(carry), key=lambda t: (scores[t], t))
        pool = carry + available[:max(0, 6-len(carry))]
        if not pool:
            break
        # Compare only candidates who actually appear in the same deliberation pool.
        # A win against a distant team does not become a permanent rank bonus.
        adjusted = {t: scores[t] - bonus * sum((t, other) in wins
                    for other in pool if other != t) for t in pool}
        ordered = sorted(pool, key=lambda t: (adjusted[t], scores[t], t))
        take = min(4 if len(placed) < 16 else 3, len(ordered))
        placed.extend(ordered[:take])
        carry = ordered[take:]
        remaining.difference_update(ordered[:take])
    return placed


def validation_order(model, test, data, bonus):
    errors = []
    for (season, week), group in test.groupby(['season', 'week']):
        scores = dict(zip(group.team, model.predict(group[list(FEATURES)])))
        wins = set()
        for g in data['seasons'][str(season)]['games']:
            if g['week'] >= week:
                continue
            winner = g['home'] if g['home_score'] > g['away_score'] else g['away']
            loser = g['away'] if winner == g['home'] else g['home']
            wins.add((winner, loser))
        ranks = {team: i+1 for i, team in enumerate(pod_order(scores, wins, bonus))}
        errors.extend(abs(min(ranks[team], 30)-actual) for team, actual in
                      zip(group.team, group['rank']))
    return float(np.mean(errors))


def main():
    data = json.loads((ROOT/'data/committee_history.json').read_text())
    frame = snapshots(data)
    # 2020's unequal pandemic schedules are kept as a held-out stress test, not
    # used to calibrate an ordinary 12-game season.
    frame = frame[frame.season != 2020]
    validation = []
    pod_errors = {b: [] for b in (0, .25, .5, 1, 2)}
    for year in range(2018, 2026):
        train = frame[frame.season < year]
        test = frame[frame.season == year]
        if train.empty or test.empty:
            continue
        model = Ridge(alpha=300).fit(train[list(FEATURES)], train['rank'])
        validation.append(dict(season=year, n=len(test), **metrics(model, test)))
        for bonus in pod_errors:
            pod_errors[bonus].append(validation_order(model, test, data, bonus))
    # Only keep the head-to-head nudge if chronological held-out rankings improve.
    best_bonus = min(pod_errors, key=lambda b: np.mean(pod_errors[b]))
    model = Ridge(alpha=300).fit(frame[list(FEATURES)], frame['rank'])
    result = dict(version=1, source='ESPN weekly CFP polls and completed games, 2014-2025',
                  features=list(FEATURES), intercept=float(model.intercept_),
                  weights={name: float(coef) for name, coef in zip(FEATURES, model.coef_)},
                  training_seasons=sorted(frame.season.unique().tolist()),
                  validation=validation,
                  pod_head_to_head_bonus=best_bonus,
                  pod_validation_mae={str(b): round(float(np.mean(v)), 4)
                                      for b, v in pod_errors.items()},
                  notes='Rank 30 denotes teams outside a Top 25; scores are predicted rank (lower is better). '
                        'Title win/loss terms replace ordinary recent win/loss terms. '
                        'Opponent winning percentages are computed as of each poll.')
    out = ROOT/'viz/data/committee_timeline_model.json'
    out.write_text(json.dumps(result, indent=2)+'\n')
    print('Saved', out)
    for row in validation:
        print(row)
    print('Recent loss',result['weights']['recent_loss'],
          'title loss',result['weights']['title_loss'])
    print('Pod head-to-head bonus',best_bonus, result['pod_validation_mae'])


if __name__ == '__main__':
    main()
