"""CPU-only derivative audit of the actual released camera decoder and K loss."""
import argparse
import hashlib
import json
import pathlib
import sys
import torch
from calibration_audit import NOMINAL_FOCAL, resized_k

ROOT = pathlib.Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=pathlib.Path, default=ROOT/'reports/calibration_v5/parameterization.json')
args = parser.parse_args()
UPSTREAM = ROOT/'third_party/TCO'
sys.path.insert(0, str(UPSTREAM))
from utils.geometry import pose_encoding_to_extri_intri
from constraints import direct_intrinsics_energy

height, width = 392, 518
target = torch.from_numpy(resized_k(NOMINAL_FOCAL, uniform=True)).float()[None,None]
encoding = torch.zeros(1,1,9)
encoding[...,6] = 1  # identity quaternion in xyzw convention
encoding[...,7] = 2*torch.atan(torch.tensor(height/(2*430.)))
encoding[...,8] = 2*torch.atan(torch.tensor(width/(2*430.)))
encoding.requires_grad_()
_, predicted = pose_encoding_to_extri_intri(encoding, (height,width))
torch.testing.assert_close(predicted[...,0,2], torch.tensor([[width/2]]))
torch.testing.assert_close(predicted[...,1,2], torch.tensor([[height/2]]))
loss = direct_intrinsics_energy(predicted, target)
matched_principal = target.clone()
matched_principal[...,0,2] = predicted[...,0,2].detach()
matched_principal[...,1,2] = predicted[...,1,2].detach()
focal_loss = direct_intrinsics_energy(predicted, matched_principal)
gradient = torch.autograd.grad(loss, encoding, retain_graph=True)[0]
focal_gradient = torch.autograd.grad(focal_loss, encoding, retain_graph=True)[0]
torch.testing.assert_close(gradient, focal_gradient, rtol=0, atol=0)
assert torch.linalg.vector_norm(gradient) > 0
principal_gradient = torch.autograd.grad(predicted[...,0,2].sum()+predicted[...,1,2].sum(), encoding)[0]
assert not torch.count_nonzero(principal_gradient)
principal_penalty = torch.nn.functional.smooth_l1_loss(predicted[...,0,2], target[...,0,2])
principal_penalty += torch.nn.functional.smooth_l1_loss(predicted[...,1,2], target[...,1,2])
principal_penalty = float(principal_penalty/6)
assert principal_penalty > 0
output = args.output
assert not output.exists()
output.parent.mkdir(parents=True, exist_ok=True)
report = dict(status='passed', scope='CPU synthetic derivative check; no adaptation experiment',
    decoder_principal_point=[width/2,height/2], released_prior_principal_point=target[0,0,:2,2].tolist(),
    principal_point_gradient_max=float(principal_gradient.abs().max()),
    gradient_identical_after_removing_principal_point_constant=True,
    irreducible_principal_component_unweighted=principal_penalty,
    irreducible_principal_component_weighted_001=principal_penalty*.01,
    focal_loss_parameterization='Smooth L1 on focal lengths measured in pixels; not relative or resolution normalized',
    source_sha256={name:hashlib.sha256((UPSTREAM/name).read_bytes()).hexdigest() for name in ['utils/geometry.py','constraints.py']},
    limitations=['A constant loss offset does not itself establish harmful parameter updates.',
                 'This checks the released decoder and loss in FP32, not a new CUDA experiment.',
                 'Adding learnable principal points or normalized residuals requires separate controlled tests.'])
output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report))
