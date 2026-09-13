# Scientific visualization — **complete** (no Higgsfield)

This folder is the **finished** scientific-viz path for PR #2. Embodiment
frames come from **TuragaLab/flybody** MuJoCo; optional **Blender 4.x**
HUD stills/animation burn in `SIMULATION / RESEARCH`. There is **no**
Higgsfield (or other generative-video) dependency.

## Status

| Artifact | Role |
|---|---|
| `frames/*.png` | MuJoCo/OSMesa stills (hero mesh) |
| `mujoco_preview.mp4` | ffmpeg stitch of those frames |
| `telemetry.json` / `telemetry.csv` | Per-frame 15-D + `U` + `I_surv` + `A_ready` |
| `pose.json` | Root qpos / heading / xpos |
| `manifest.json` | Provenance (`hero_mesh`, `viz_complete`) |
| `INSTALL_FLYBODY.txt` | Written **only** if the mesh is missing |
| `renders/blender_still.png` | Blender 4.x HUD still (when Blender is installed) |
| `renders/blender_anim.mp4` | Optional short HUD animation |

`manifest.json` sets `"viz_complete": true` after a live-mesh export.

## 1. Export sidecar (required)

```bash
pip install -e ".[dev]"
python3 -m confluence.demo_blender --out docs/demo/blender
```

Requires a **live** TuragaLab/flybody install (`pip install -e '.[flybody]'`
plus `MUJOCO_GL=osmesa`). Stub/CPG frames are **refused**. `--allow-stub`
is rejected. If the mesh is missing, the command writes
`INSTALL_FLYBODY.txt` and exits non-zero.

## 2. Optional Blender HUD (workstation or this VM)

Blender is **not** a Python dependency. On Ubuntu:

```bash
sudo apt-get update && sudo apt-get install -y blender
```

Headless still (always stamps `SIMULATION / RESEARCH`):

```bash
blender --background --python docs/demo/blender/confluence_blender_hud.py -- \
  --root docs/demo/blender
```

Short animation (few frames; also stamped):

```bash
blender --background --python docs/demo/blender/confluence_blender_hud.py -- \
  --root docs/demo/blender --animation --max-frames 8
```

If `blender` is not on `PATH`, skip this step. The MuJoCo preview MP4
and sidecar remain the complete scientific record.

## Honesty

This is a **simulation overlay**, not a clinical scan, not a treatment
video, and not evidence of a cure. See
[DISCLAIMER.md](../../DISCLAIMER.md) and
[AWAITING_CLINICAL_VALIDATION.md](../../AWAITING_CLINICAL_VALIDATION.md).
