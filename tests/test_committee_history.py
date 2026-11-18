"""The shipped season snapshots must reproduce the published committee fit."""
import json
import sys
from pathlib import Path

from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from train_committee_timeline import FEATURES, load_history, snapshots  # noqa: E402

history = load_history()
assert sorted(history['seasons']) == [str(year) for year in range(2014, 2027)]
assert all(season['polls'] and season['games'] for season in history['seasons'].values())

rows = snapshots(history)
train = rows[(rows.season != 2020) & (rows.season <= 2025)]
fit = Ridge(alpha=300).fit(train[list(FEATURES)], train['rank'])
published = json.loads((ROOT / 'viz/data/committee_timeline_model.json').read_text())
assert published['training_seasons'] == sorted(train.season.unique().tolist())
assert abs(float(fit.intercept_) - published['intercept']) < 1e-8
assert all(abs(float(weight) - published['weights'][feature]) < 1e-8
           for feature, weight in zip(FEATURES, fit.coef_))
print('Historical CFP snapshots reproduce the published committee model')
