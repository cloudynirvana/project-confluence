# Blender HUD stills (research)

Headless Blender 4.x output. Every frame is stamped **SIMULATION / RESEARCH**.

```bash
# still
blender --background --python docs/demo/blender/confluence_blender_hud.py -- \
  --root docs/demo/blender

# short animation (few frames)
blender --background --python docs/demo/blender/confluence_blender_hud.py -- \
  --root docs/demo/blender --animation --max-frames 8
```

Committed samples: `blender_still.png`, `anim_0001.png`, `anim_0008.png`.
Regenerate the rest with `--animation`. If `blender` is not on `PATH`,
use [`../mujoco_preview.mp4`](../mujoco_preview.mp4) and the sidecar.
Not a clinical film.
