# Cinematic demo (real flybody mesh)

Short closed-loop clip for README / sharing. The subject is the **TuragaLab/flybody** anatomical Drosophila (`fruitfly.xml`, `walker/hero` camera). Slim HUD ticks show simulated burden / resistance / DA / protein channels.

This is a **research visualization**. Simulated therapy metrics are **research scores**, not a clinical outcome.

The CPG / bead-fly stub is **not** used here. `python -m confluence.demo_cinematic` exits nonzero if MuJoCo cannot render `fruitfly.xml`.

## Files

| File | What |
|------|------|
| `cinematic.mp4` | ≈12 s, 1280×720, H.264, real mesh |
| `still_hero.png` | opening still (`walker/hero`, slim HUD) |
| `still_mid.png` | mid-clip still |
| `still_mesh.png` | clean `walker/hero` frame, no HUD (Nature close-up) |
| `still_track.png` | clean `walker/track1` frame (Menagerie-style 3/4) |
| `immune_chimeric.mp4` | fly-brain immune + chimeric-protein demo (real mesh) |
| `still_immune_hero.png` | opening immune-demo still |
| `still_immune_mid.png` | mid immune-demo still |
| `immune_chimeric.json` | I_act / fusion AF / protein metrics (research scores) |

## Re-render

```bash
bash scripts/install_flybody.sh
export MUJOCO_GL=osmesa
python -m confluence.demo_cinematic --seconds 12 --out docs/demo/cinematic.mp4
python -m confluence.demo_immune --seconds 8 --out docs/demo/immune_chimeric.mp4
```

Live UI immune demo: open `http://127.0.0.1:8765/?demo=immune` (auto-resets to melanoma + controller F, then plays). Research visualization only — not a clinical immune-therapy demo.

## Blender scientific path (data-driven)

Logged closed-loop HUD + MuJoCo `fruitfly.xml` frames for local Blender 4.x. **SIMULATION / RESEARCH** — not generative biology.

```bash
export MUJOCO_GL=osmesa
python3 -m confluence.demo_blender --out docs/demo/blender
```

See [`docs/demo/blender/README.md`](blender/README.md).

Body on screen: TuragaLab/flybody `fruitfly.xml` (Apache 2.0). Not NeuroMechFly / FlyGym unless you change the bridge and document it.
