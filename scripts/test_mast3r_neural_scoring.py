"""Ensure clean-pose oracles cannot enter the noisy-channel comparator set."""
import unittest
from score_mast3r_neural_transfer import summarize,REFS,PUBLIC,read


class NeuralScoringTests(unittest.TestCase):
    def test_unavailable_clean_oracle_is_excluded_only_from_noisy_comparison(self):
        rows=[];observations={};historical={'grids':{}}
        metrics=lambda value:dict(accuracy_mean_m=value,completeness_mean_m=value)
        for job in read(PUBLIC/'plan.json')['jobs']:
            key=job['scene']+'_base'
            for post in job['post']:
                tag=job['case']+'_'+post
                value=.0001 if post=='KDP_clean' and job['arm'] in ('zero','D','K') else .8 if job['arm']=='J' else .9 if job['arm']=='N' else 1.
                observations.setdefault(key,{})[tag]={ref:metrics(value) for ref in REFS}
                rows.append(dict(**{k:job[k] for k in ('scene','grid','arm')},input_key=key,post=post,prediction=tag))
        for grid in ('centered','shifted'):
            references={}
            for ref in REFS:
                scenes={key:{arm:dict(raw=metrics(.001 if arm in ('pose','joint') else .95 if arm=='depth' else 1.))
                             for arm in ('standard','depth','pose','joint')} for key in observations}
                references[ref]=dict(scenes=scenes)
            historical['grids'][grid]=dict(references=references)
        summary=summarize(dict(rows=rows),observations,historical)
        for cells in summary.values():
            clean,noisy=cells['clean'],cells['noisy']
            self.assertTrue(clean['best_control'].endswith('KDP_clean'))
            self.assertEqual(noisy['best_control'],'historical_depth')
            self.assertFalse(any(a.endswith('KDP_clean') or a in ('historical_pose','historical_joint') for a in noisy['eligible_controls']))
            self.assertTrue(noisy['advance'])
            self.assertFalse(clean['advance'])
