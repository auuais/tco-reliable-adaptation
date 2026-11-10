"""Check real upstream loss/gradient isolation and reject mismatched provenance."""
import copy
import pathlib
import sys
import unittest
import numpy as np
import torch
from omegaconf import OmegaConf
from prior_ablation_config import CONDITIONS, effective_config, validate_loss_components
from aggregate_prior_ablation import validate_record
from run_prior_ablation_batch import plan

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party/TCO'))
import tco_loss
from unittest.mock import patch


class LossIsolationTests(unittest.TestCase):
    def setUp(self):
        self.base=OmegaConf.to_container(OmegaConf.load(ROOT/'configs/7scenes_baseline.yaml'))

    def test_actual_upstream_losses_and_gradients_are_additive_by_group(self):
        # Real pose/intrinsics penalties; deterministic differentiable visual fixture.
        poses=torch.eye(4,dtype=torch.float64).repeat(1,3,1,1)
        poses[0,:,0,3]=torch.tensor([0.,1.1,1.8]);poses[0,1,1,3]=.2
        poses.requires_grad_()
        gt=poses.detach().clone();gt[0,1,0,3]=1.;gt[0,1,1,3]=0
        gt=torch.linalg.inv(gt)
        k=torch.tensor([[460.,0,260],[0,445,196],[0,0,1]],dtype=torch.float64).repeat(1,3,1,1).requires_grad_()
        kg=k.detach().clone();kg[:,:,0,0]=450.;kg[:,:,0,2]=259.
        depth=torch.ones(1,3,4,4,dtype=torch.float64,requires_grad=True)
        predictions=dict(camera_poses=poses,intrinsics=k,depth=depth,pose_type='abs')
        results={}
        def visual(**kw):return kw['depth'].square().mean()+kw['intrinsics'][...,0,0].square().mean()*1e-5
        for condition,(ps,ks) in CONDITIONS.items():
            cfg=effective_config(self.base,ps,ks,40,42,.0003)
            with patch.object(tco_loss,'depth_multiview_consistency_energy',side_effect=visual):
                loss,parts=tco_loss.compute_tco_losses(predictions,torch.zeros(1,3,3,4,4,dtype=torch.float64),
                    gt_extrinsics=gt,gt_intrinsics=kg,lambda_pose=cfg['lambda_pose'],
                    pose_translation_weight=2,lambda_intrinsics=cfg['lambda_intrinsics'],
                    lambda_mv_consistency=.2,canonicalize_to_first_view=True,pose_rot_loss_type='angle',pose_trans_loss_type='normed_l1')
            values={name:float(value) for name,value in parts.items()};values['total']=float(loss)
            validate_loss_components(cfg,values)
            grads=torch.autograd.grad(loss,(poses,k,depth),allow_unused=True,retain_graph=True)
            flat=torch.cat([(torch.zeros_like(t) if g is None else g).flatten() for g,t in zip(grads,(poses,k,depth))])
            results[condition]=(loss.detach(),flat)
            self.assertEqual(cfg['lambda_mv_consistency'],.2)
        for i in [0,1]:
            torch.testing.assert_close(results['both'][i]-results['visual'][i],
                results['pose_only'][i]+results['intrinsics_only'][i]-2*results['visual'][i],rtol=1e-9,atol=1e-10)
        self.assertGreater(float(torch.linalg.vector_norm(results['pose_only'][1]-results['visual'][1])),0)
        self.assertGreater(float(torch.linalg.vector_norm(results['intrinsics_only'][1]-results['visual'][1])),0)

    def test_disabled_residual_and_objective_mismatch_rejected(self):
        cfg=effective_config(self.base,0,1,40,42,.0003)
        parts=dict(pose_rot_loss=0.,pose_trans_loss=0.,intrinsics_loss=2.,mv_consistency_loss=3.,depth_loss=0.,redundancy_loss=0.,total=.62)
        validate_loss_components(cfg,parts)
        for key,value in [('pose_rot_loss',.1),('total',.8),('depth_loss',1.)]:
            changed=dict(parts);changed[key]=value
            with self.assertRaises(AssertionError):validate_loss_components(cfg,changed)


class ProvenanceTests(unittest.TestCase):
    def test_full_factorial_and_wrong_component_rejected(self):
        p=plan();self.assertEqual(len(p['jobs']),39)
        self.assertEqual(sum(j['steps'] for j in p['jobs']),1440)
        for scene in ['heads/seq-01','chess/seq-03','fire/seq-03']:
            for seed in [42,123,2026]:
                jobs=[j for j in p['jobs'] if j['scene']==scene and j['seed']==seed and j['steps']]
                self.assertEqual({j['condition'] for j in jobs},set(CONDITIONS))
        j=next(j for j in p['jobs'] if j['condition']=='pose_only')
        control={k:'same' for k in ['research_commit','input_sha256','reference_sha256','upstream_commit',
            'upstream_method_sha256','versions','gpu','gpu_identity','gsplat_backend','checkpoint_download','image_shape','evaluation_tensor_sha256','prior_metadata']}
        control['configuration']=OmegaConf.to_container(OmegaConf.load(ROOT/'configs/7scenes_baseline.yaml'))
        control['metrics']=dict(accuracy_mean_m=.01,completeness_mean_m=.02,normal_consistency_mean=.8,protocol='same')
        r=copy.deepcopy(control)
        r.update(status='complete',research_dirty=False,study_id=p['study_id'],manifest_sha256='frozen',
            base_parameters_modified=[],runner_sha256=p['sources']['scripts/run_prior_ablation.py'],
            prior_generator_sha256=p['sources']['scripts/prior_corruptions.py'],
            ablation_config_sha256=p['sources']['scripts/prior_ablation_config.py'],optimizer_learning_rates=[.0003],
            inference_and_adaptation_seconds=1,arguments=dict(scene=j['scene'],steps=40,frames=j['frames'],size=518,
            seed=j['seed'],prior_profile='clean',prior_weight=1.,prior_seed=1729,learning_rate=.0003,
            pose_weight_scale=1.,intrinsics_weight_scale=0.))
        r['configuration']=effective_config(control['configuration'],1,0,40,j['seed'],.0003)
        validate_record(r,j,p,'frozen',control)
        for key,value in [('configuration',control['configuration']),('research_dirty',True),
            ('ablation_config_sha256','wrong'),('optimizer_learning_rates',[.001]),('reference_sha256','wrong')]:
            changed=copy.deepcopy(r);changed[key]=value
            with self.assertRaises(AssertionError):validate_record(changed,j,p,'frozen',control)


if __name__=='__main__':unittest.main()
