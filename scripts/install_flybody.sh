#!/usr/bin/env bash
# Install TuragaLab/flybody without downgrading Confluence's numpy 2.x.
# Upstream pyproject pins numpy==1.26.4; we install the package --no-deps.
set -euo pipefail
COMMIT="${FLYBODY_COMMIT:-d015e9bfe441bd90ae431bac24c55cb74bdbce26}"
python3 -m pip install 'mujoco>=3.1' dm_control h5py mediapy pillow
python3 -m pip install --no-deps "flybody @ git+https://github.com/TuragaLab/flybody.git@${COMMIT}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
python3 - <<'PY'
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
from flybody.fly_envs import template_task
env = template_task()
env.reset()
pix = env.physics.render(height=120, width=160, camera_id="walker/hero")
assert pix.shape[-1] == 3
print("flybody fruitfly.xml render OK", pix.shape)
PY
