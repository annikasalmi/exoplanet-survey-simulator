"""Scientific invariants of the shared conditional-on-N comparison."""
import unittest
import json
import tempfile
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.stats import binom
from science.population_comparison import (ComparisonConfig, BINS, POSTERIOR_BINS,
    count_log_likelihood, posterior_from_logs, rng_for, zone_mask, observed_counts,
    model_draws, load_sample, run_comparison)


def sample():
    frame = pd.DataFrame({'planet_name':['edge_mass','edge_radius','inside'],
        'mass':[2.5,3.,3.], 'radius':[1.5,1.4,1.5], 'insolation':[1.,1.,1.],
        'transit_flag':[1,1,1], 'mass_limit_flag':[0,0,0], 'radius_limit_flag':[0,0,0]})
    for q in ('mass','radius'):
        for side in ('plus','minus'):
            frame[f'{q}_error_{side}'] = .1
    return frame


class ComparisonTests(unittest.TestCase):
    def test_strict_boundaries_and_half_open_bins(self):
        cfg=ComparisonConfig(draws=50)
        self.assertEqual(zone_mask(sample(), .01, 50, cfg).tolist(), [False,False,True])
        s=sample().iloc[[2]].copy();s['insolation']=50
        self.assertFalse(zone_mask(s,.01,50,cfg).any())
        self.assertTrue(zone_mask(s,50,10000,cfg).all())

    def test_fixed_observed_membership(self):
        s=sample();s.loc[2,'mass_error_minus']=5
        zone,counts,_,q=observed_counts(s,ComparisonConfig(draws=80),'test',.01,50)
        self.assertEqual(zone.planet_name.tolist(),['inside'])
        self.assertEqual(len(counts),80)
        self.assertAlmostEqual(q.sum(),1.)
        self.assertTrue(np.isin(counts,[0,1]).all())

    def test_independent_count_convolution_retains_rare_combinations(self):
        s=pd.concat([sample().iloc[[2]]]*2,ignore_index=True)
        def classes(mass,radius,*curve):
            return np.array([[True,False],[False,True]]) if np.asarray(mass).ndim==2 else np.array([True,True])
        with patch('science.population_comparison.is_volatile',side_effect=classes):
            _,counts,_,q=observed_counts(s,ComparisonConfig(draws=2),'test',.01,50)
        np.testing.assert_array_equal(counts,[1,1])
        np.testing.assert_allclose(q,[.25,.5,.25])

    def test_likelihood_reduces_to_binomial(self):
        actual=count_log_likelihood(np.full(10,3),np.full(20,.4),5)
        self.assertAlmostEqual(actual,binom.logpmf(3,5,.4))

    def test_mixture_likelihood_not_plugin_mean(self):
        ps=np.array([.1,.9]);expected=np.mean(binom.pmf(5,5,ps))
        self.assertAlmostEqual(np.exp(count_log_likelihood(np.array([5]),ps,5)),expected)
        self.assertNotAlmostEqual(expected,binom.pmf(5,5,ps.mean()))

    def test_observed_count_ambiguity_is_marginalized(self):
        expected=.5*binom.pmf(1,3,.4)+.5*binom.pmf(2,3,.4)
        self.assertAlmostEqual(np.exp(count_log_likelihood(np.array([1,2]),np.array([.4]),3)),expected)

    def test_named_streams_and_posterior(self):
        cfg=ComparisonConfig()
        first=rng_for(cfg,'bin','model').normal(size=20)
        rng_for(cfg,'unrelated').normal(size=1000)
        np.testing.assert_array_equal(first,rng_for(cfg,'bin','model').normal(size=20))
        self.assertNotEqual(first[0],rng_for(replace(cfg,seed=1),'bin','model').normal())
        self.assertAlmostEqual(sum(posterior_from_logs({'a':-1000.,'b':-1001.}).values()),1.)

    def test_no_overlapping_posterior_bins(self):
        intervals=[(lo,hi) for name,lo,hi in BINS if name in POSTERIOR_BINS]
        self.assertEqual(len(intervals),2)
        self.assertEqual(intervals[0][1],intervals[1][0])
        self.assertNotIn('I < 10',POSTERIOR_BINS)

    def test_strict_catalogue_eligibility(self):
        s=pd.concat([sample().iloc[[2]]]*5,ignore_index=True)
        s.loc[1,'transit_flag']=0;s.loc[2,'mass_limit_flag']=-1
        s.loc[3,'radius_limit_flag']=1;s.loc[4,'mass_error_minus']=np.nan
        with patch('science.population_comparison.load_measured_planets',return_value=s),patch('science.population_comparison.read_nasa_csv',return_value=s):
            accepted,audit=load_sample()
        self.assertEqual(len(accepted),1);self.assertEqual(audit['strict_sample_rows'],1)

    def test_cache_names_with_periods_keep_configuration_and_reuse(self):
        s=sample()
        pool=pd.DataFrame({'mass':np.full(60,4.),'radius':np.linspace(1.41,2.19,60),
                           'insolation':np.ones(60),'joint_detected':np.ones(60,dtype=bool)})
        config=ComparisonConfig(draws=20)
        with tempfile.TemporaryDirectory() as directory, \
             patch('science.population_comparison.CACHE_DIR',Path(directory)), \
             patch('science.population_comparison.load_sample',return_value=(s,{})), \
             patch('science.population_comparison.source_digest',return_value='test'), \
             patch('science.population_comparison.model_draws',wraps=model_draws) as draw:
            kwargs=dict(populations={'rocky_formation':pool},scenario='Author_et_al._2020',bins=(('I < 50',.01,50.),))
            first,arrays=run_comparison(config,**kwargs)
            other,_=run_comparison(replace(config,draws=21),**kwargs)
            cached,saved=run_comparison(config,**kwargs)
            self.assertEqual(draw.call_count,2)
            self.assertEqual(len(list(Path(directory).glob('*.json'))),2)
            self.assertEqual(other['provenance']['config']['draws'],21)
            self.assertEqual(json.loads(json.dumps(first)),cached)
            np.testing.assert_array_equal(arrays['b0_rocky_formation'],saved['b0_rocky_formation'])

    def test_mock_fractions_have_exact_observed_denominator(self):
        s=sample().iloc[[2]].copy()
        pool=pd.DataFrame({'mass':np.full(60,4.),'radius':np.linspace(1.41,2.19,60),
                           'insolation':np.ones(60),'joint_detected':np.ones(60,dtype=bool)})
        cfg=ComparisonConfig(draws=30)
        draws,p,eligible=model_draws(pool,s,cfg,'test',.01,50,'a',7)
        np.testing.assert_allclose(draws*7,np.round(draws*7))
        self.assertEqual(len(draws),30);self.assertGreaterEqual(eligible,7)
        self.assertTrue(((p>0)&(p<1)).all())
        with self.assertRaisesRegex(ValueError,'only 60'):
            model_draws(pool,s,cfg,'test',.01,50,'b',61)


if __name__=='__main__':
    unittest.main()
