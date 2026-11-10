"""Export logged loss components and total gradient norms, without new inference."""
import argparse
import csv
import json
import pathlib
from prior_ablation_config import validate_loss_components

ROOT = pathlib.Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=pathlib.Path, required=True)
parser.add_argument('--output', type=pathlib.Path, required=True)
args = parser.parse_args()
plan = json.loads((ROOT/'manifests/prior_ablation_v5.json').read_text())
rows = []
for job in plan['jobs']:
    if not job['steps']:
        continue
    folder = args.runs/job['run_id']
    record = json.loads((folder/'run.json').read_text())
    cfg = record['configuration']
    losses = json.loads((folder/'losses.json').read_text())
    gradients = json.loads((folder/'gradients.json').read_text())
    assert len(losses) == len(gradients) == job['steps']
    for loss, grad in zip(losses, gradients):
        validate_loss_components(cfg, loss)
        rows.append(dict(run_id=job['run_id'], scene=job['scene'], condition=job['condition'],
            seed=job['seed'], step=loss['step'], total=loss['total'],
            weighted_pose=cfg['lambda_pose']*(loss['pose_rot_loss']+cfg['pose_translation_weight']*loss['pose_trans_loss']),
            weighted_intrinsics=cfg['lambda_intrinsics']*loss['intrinsics_loss'],
            weighted_visual=cfg['lambda_mv_consistency']*loss['mv_consistency_loss'],
            gradient_norm_after_upstream_clipping=grad['gradient_norm_after_upstream_clipping']))
assert len(rows) == 1440
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open('w', newline='', encoding='utf-8') as out:
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
