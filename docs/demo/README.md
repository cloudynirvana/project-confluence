# Cinematic demo (dark lab)

Short closed-loop clip for README / sharing. One subject: the fly. Slim HUD ticks for burden, resistance, DA, and active protein channels.

This is a **research visualization**, not a clinical or product trailer.

## Files

| File | What |
|------|------|
| `cinematic.mp4` | ≈12 s, 1280×720, H.264 |
| `still_hero.png` | opening still |
| `still_mid.png` | mid-clip still |

## Re-render

```bash
# Stub look (this VM / default CI — no MuJoCo)
python -m confluence.demo_cinematic --seconds 12 --out docs/demo/cinematic.mp4

# Physics-real flybody (local extra)
pip install -e ".[flybody]"
export MUJOCO_GL=osmesa   # or egl
python -m confluence.demo_cinematic --seconds 12 --prefer-real --out docs/demo/cinematic.mp4
```

The live UI at `http://127.0.0.1:8765` uses the same hero composition. Append `?cinema=1` to hide the control drawer while recording. When `.[flybody]` is installed the viewport streams `physics.render` JPEGs; otherwise it uses the cinematic CPG stub (beige Drosophila, not a stick figure).
