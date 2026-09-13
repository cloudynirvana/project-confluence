# Blender scientific visualization (data-driven)

**SIMULATION / RESEARCH.** Logged Confluence closed-loop series + TuragaLab/flybody `fruitfly.xml` frames. Not a clinical trial, not a cure, not FDA/EMA readiness, not generative tumor-shrink footage (no Higgsfield in this pass).

HUD numbers (burden, H, fusion AF, antibody readiness, U) are **exactly** the ODE / controller values written to the sidecar. The bpy script does not invent biology.

## One-command export

```bash
export MUJOCO_GL=osmesa
python3 -m confluence.demo_blender --out docs/demo/blender
```

Requires `.[flybody]` for the PNG / pose dump. The JSON/CSV sidecar is written from `ClosedLoopSimulator` even if MuJoCo is missing (then `n_png=0` and the README tells you to install flybody). Mesh frames **refuse** the CPG stub.

## Outputs

| File | What |
|------|------|
| `telemetry.json` / `telemetry.csv` | HUD series keyed by `frame` and `t_days` |
| `pose.json` | MuJoCo root pose (`qpos_root`, xpos, heading) per frame |
| `frames/frame_XXXX.png` | Clean `walker/hero` RGB from `env.physics.render` |
| `still_mesh.png` / `still_mid.png` | First / mid MuJoCo stills |
| `mujoco_preview.mp4` | ffmpeg of the PNG sequence (MuJoCo dump, not a Blender render) |
| `manifest.json` | Paths + honesty flags |
| `confluence_blender_hud.py` | Local Blender 4.x importer / HUD driver |

## Open in local Blender 4.x

This agent VM has no Blender GUI. On a workstation with Blender 4.x:

```bash
blender --background --python docs/demo/blender/confluence_blender_hud.py -- --root docs/demo/blender
# optional full sequence:
blender --background --python docs/demo/blender/confluence_blender_hud.py -- --root docs/demo/blender --animation
```

The script loads the PNG sequence as an emission plane and drives text + a burden curve from `telemetry.json`. Overlay copy always includes `SIMULATION / RESEARCH`.

`fruitfly.xml` itself can also be opened in Blender via community MJCF importers if you want a true mesh; this path ships the **already rendered** scientific frames plus the synced sidecar so nothing is hallucinated.
