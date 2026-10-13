import copy
import json
import unittest
from pathlib import Path
import numpy as np
from scipy.special import expit
from scripts import ensemble_blend_research as B
from scripts import ensemble_replay as ER

class ResearchContracts(unittest.TestCase):
    def test_fitted_models_preserve_reciprocity(self):
        rng=np.random.default_rng(17);X=rng.normal(size=(80,4));y=(rng.random(80)<expit(X[:,0])).astype(int)
        b=B.logistic_fit(X,y,anchor=np.full(4,.25))
        np.testing.assert_allclose(expit(X@b)+expit((-X)@b),1,atol=1e-12)
        w=B.convex_fit(expit(X),y)
        self.assertAlmostEqual(w.sum(),1,places=10)
        self.assertTrue(np.all(w>=0))
        np.testing.assert_allclose(expit(X)@w+(1-expit(X))@w,1,atol=1e-12)

    def test_matchup_features_negate_when_teams_swap(self):
        e=json.loads((B.ROOT/'viz/data/model_v4.json').read_text())['ensemble']
        p,x,c=B.features(e,e['state'],'Alabama','Notre Dame',0)
        q,z,d=B.features(e,e['state'],'Notre Dame','Alabama',0)
        np.testing.assert_allclose(p+q,1,atol=1e-12)
        np.testing.assert_allclose(c,-d,atol=1e-12)
        for k in x:self.assertAlmostEqual(x[k],-z[k],places=12)

    def test_future_results_do_not_enter_earlier_inputs(self):
        # Runtime stop_before is also the research extractor's cutoff.
        e=json.loads((B.ROOT/'viz/data/model_v4.json').read_text())['ensemble']
        finals=[{'week':1,'home':'Alabama','away':'Notre Dame','neutral':True,'home_score':21,'away_score':24},
                {'week':3,'home':'Alabama','away':'Notre Dame','neutral':True,'home_score':50,'away_score':0}]
        a=ER.replay(e,finals,[],stop_before=3)['state']
        finals[-1]['home_score']=0;finals[-1]['away_score']=100
        b=ER.replay(e,finals,[],stop_before=3)['state']
        self.assertEqual(a,b)

if __name__=='__main__':unittest.main()
