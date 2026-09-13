# Cinematic demo (real flybody mesh)

Short closed-loop clip for README / sharing. The subject is the **TuragaLab/flybody** anatomical Drosophila (`fruitfly.xml`, `walker/hero` camera). Slim HUD ticks show simulated burden / resistance / DA / protein channels.

This is a **research visualization**. Simulated therapy metrics are **not** a computational cancer cure.

The CPG / bead-fly stub is **not** used here. `python -m confluence.demo_cinematic` exits nonzero if MuJoCo cannot render `fruitfly.xml`.

## Files

| File | What |
|------|------|
| `cinematic.mp4` | ≈12 s, 1280×720, H.264, real mesh |
| `still_hero.png` | opening still (`walker/hero`, slim HUD) |
| `still_mid.png` | mid-clip still |
| `still_mesh.png` | clean `walker/hero` frame, no HUD (Nature close-up) |
| `still_track.png` | clean `walker/track1` frame (Menagerie-style 3/4) |

## Re-render

```bash
bash scripts/install_flybody.sh
export MUJOCO_GL=osmesa
python -m confluence.demo_cinematic --seconds 12 --out docs/demo/cinematic.mp4
```

Body on screen: TuragaLab/flybody `fruitfly.xml` (Apache 2.0). Not NeuroMechFly / FlyGym unless you change the bridge and document it.
