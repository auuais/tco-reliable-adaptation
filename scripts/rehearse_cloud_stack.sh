#!/usr/bin/env bash
# Local-only clean-container rehearsal. Container networking must be disabled.
# /transfer is this project, read-only. /workspace is a new writable scratch mount.
# The verified local checkpoint is mounted at /workspace/tco-assets/checkpoints.
set -euo pipefail
export PATH="/opt/conda/bin:/usr/local/cuda/bin:$PATH"
python /transfer/scripts/prepare_cloud_workspace.py --transfers /transfer/cloud --workspace /workspace
cd /workspace/tco-research
bash scripts/cloud_bootstrap.sh --local-rehearsal
python - <<'PY'
import json,pathlib,subprocess
from gsplat import csrc
root=pathlib.Path('/workspace/tco-research')
objects=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--list-elf',csrc.__file__],text=True)
assert 'sm_86' in objects and 'sm_89' in objects
(root/'environment/precompiled_cuda_architectures.txt').write_text(objects)
run=json.loads((root/'runs/rehearsal_precompiled224_s1_seed42/run.json').read_text())
assert run['status']=='complete' and run['gsplat_backend']['module']=='gsplat.csrc'
gradients=json.loads((root/'runs/rehearsal_precompiled224_s1_seed42/gradients.json').read_text())
assert len(gradients)==1 and gradients[0]['finite'] and not run['base_parameters_modified']
print('Clean offline runtime, precompiled sm_86/sm_89 presence, real renderer and one TCO update passed.')
PY
