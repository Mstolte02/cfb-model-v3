import unittest
import numpy as np
from scipy.special import expit
from scripts import historical_ensemble_research as H


class HistoricalContracts(unittest.TestCase):
    def test_future_labels_cannot_change_earlier_fold(self):
        rng=np.random.default_rng(8)
        years=np.repeat([2023,2024,2025],60)
        X=rng.normal(size=(180,4)); y=(rng.random(180)<expit(X[:,0])).astype(int)
        train,test=H.chronological_masks(years,2024)
        self.assertTrue(np.all(years[train]<2024))
        first=H.B.logistic_fit(X[train],y[train])
        changed=y.copy();changed[~train]=1-changed[~train]
        np.testing.assert_array_equal(first,H.B.logistic_fit(X[train],changed[train]))
        self.assertEqual(int(test.sum()),60)

    def test_union_contains_original_inputs_once(self):
        self.assertEqual(len(H.UNION),len(set(H.UNION)))
        self.assertIn('player_prod_war',H.UNION)
        self.assertIn('talent_x_returning',H.UNION)
        self.assertNotIn('prior_level',H.UNION)

    def test_calibration_covers_boundaries(self):
        p=np.array([0.,.1,.5,.9,1.]); y=np.array([0,0,1,1,1])
        bins=H.calibration(y,p)['bins']
        self.assertEqual(sum(b['n'] for b in bins),len(y))

    def test_paired_intervals_are_zero_for_identical_predictions(self):
        p=np.array([.1,.8,.4,.6]);y=np.array([0,1,1,0])
        r=H.uncertainty(y,p,p,np.array([2024,2024,2025,2025]),np.array([1,2,1,2]))
        for metric in ('brier','log_loss'):
            np.testing.assert_allclose(r[metric]['season_week_bootstrap_95'],[0,0])


if __name__=='__main__':unittest.main()
