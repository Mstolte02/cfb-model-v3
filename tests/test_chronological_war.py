import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scripts import chronological_war_rebuild as R
from scripts import war_matchup_chronological as W
from src import opponent_strength as O


class ChronologicalWarTests(unittest.TestCase):
    def test_future_measurements_cannot_change_normalization(self):
        raw=pd.DataFrame({'season':np.repeat([2022,2023,2024],250),
                          'facet':'test','metric':np.arange(750.),'snaps':100.})
        expected=R.fitted_moments(raw,2023)
        raw.loc[raw.season>=2023,'metric']=1e10
        self.assertEqual(expected,R.fitted_moments(raw,2023))
        self.assertAlmostEqual(expected['test'][0],124.5)

    def unit_data(self):
        rng=np.random.default_rng(22)
        return pd.DataFrame({f'{p}_{s}_{f}':rng.normal(size=40)
                             for p in W.PAIRS for s in ('O','D') for f in ('h','a')})

    def test_opposing_features_are_reciprocal_and_train_scaled(self):
        train=self.unit_data();held=train.iloc[:5].copy()
        a,b,scale=W.unit_transform(train,held)
        _,_,changed=W.unit_transform(train,held*1e6)
        self.assertEqual(scale,changed)
        swapped=held.rename(columns={f'{p}_{s}_{f}':f'{p}_{s}_{other}'
                    for p in W.PAIRS for s in ('O','D') for f,other in [('h','a'),('a','h')]})
        _,reverse,_=W.unit_transform(train,swapped)
        np.testing.assert_allclose(b[W.MAIN+W.PRODUCT+W.GAP],-reverse[W.MAIN+W.PRODUCT+W.GAP])
        p=next(iter(W.PAIRS))
        np.testing.assert_allclose(b[p+'_product'],b[p+'_O_h']*b[p+'_D_a']-b[p+'_O_a']*b[p+'_D_h'])

    def test_interactions_have_identical_main_effect_controls(self):
        for p in W.PAIRS:
            control=W.FAMILIES[p+'_linear']
            for nonlinear in ('product','gap'):
                self.assertEqual(W.FAMILIES[p+'_'+nonlinear][:-1],control)

    def test_missing_opponent_does_not_create_fake_interaction(self):
        train=self.unit_data();held=train.iloc[:1].copy()
        for p in W.PAIRS:held[p+'_D_a']=np.nan
        _,result,_=W.unit_transform(train,held)
        np.testing.assert_array_equal(result[W.PRODUCT+W.GAP],0.)

    def test_postseason_week_one_cannot_enter_early_opponent_strength(self):
        row={'week':1,'team':'A','opponent':'B','offense':{'ppa':.1},'defense':{'ppa':.2}}
        with tempfile.TemporaryDirectory() as tmp,patch.object(O,'ROOT',Path(tmp)):
            path=Path(tmp)/'data/raw';path.mkdir(parents=True)
            (path/'game_advanced_2023.json').write_text(json.dumps([
                {**row,'seasonType':'regular'},{**row,'seasonType':'postseason'}]))
            O.games.cache_clear()
            try:self.assertEqual(len(O.games(2023)),1)
            finally:O.games.cache_clear()

    def test_published_folds_and_scores_match_prediction_ledger(self):
        report=json.loads(W.OUT.read_text())
        ledger=pd.read_csv(W.OUT.with_name('chronological_war_matchup_predictions.csv'))
        self.assertFalse(ledger.duplicated(W.V.KEYS).any())
        for fold in report['folds']:
            self.assertTrue(all(y<fold['season'] for y in fold['train_seasons']))
            for fit in fold['fits']:
                self.assertTrue(all(y<fold['season'] for y in fit['training_seasons']))
        for fold in report['war_provenance']['folds']:
            self.assertTrue(all(y<fold['season'] for y in fold['training_seasons']))
        scored=ledger[ledger.season>=2024]
        self.assertEqual(len(scored),report['evaluation_games'])
        for name in W.FAMILIES:
            actual=W.metrics(scored.y,scored[name])
            for metric in ('brier','log_loss','accuracy','auc','ece'):
                self.assertAlmostEqual(actual[metric],report['metrics'][name]['overall'][metric],places=12)


if __name__=='__main__':unittest.main()
