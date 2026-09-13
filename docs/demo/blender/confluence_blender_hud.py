"""Blender 4.x HUD compositor for logged Confluence scientific frames.

Open locally (this VM has no Blender GUI):

    blender --background --python docs/demo/blender/confluence_blender_hud.py -- \\
        --root docs/demo/blender

Loads the MuJoCo PNG sequence as an image plane and drives text / a curve
from telemetry.json. Every overlay is labeled SIMULATION / RESEARCH.

Not a clinical film. Not generative tumor-shrink footage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


SIMULATION_LABEL = "SIMULATION / RESEARCH"


def _argv_after_dd() -> list:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def _load(root: Path) -> dict:
    payload = json.loads((root / "telemetry.json").read_text(encoding="utf-8"))
    if not payload.get("rows"):
        raise SystemExit("telemetry.json has no rows — run python3 -m confluence.demo_blender first")
    return payload


def _text(bpy, name: str, body: str, loc, scale=0.12):
    curve = bpy.data.curves.new(name=name, type="FONT")
    curve.body = body
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.location = loc
    obj.scale = (scale, scale, scale)
    return obj


def build_scene(root: Path) -> None:
    import bpy

    payload = _load(root)
    rows = payload["rows"]
    n = len(rows)
    frames_dir = root / "frames"
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = max(n, 1)
    scene.render.fps = int(payload.get("fps") or 12)
    scene.render.resolution_x = int(payload.get("width") or 640)
    scene.render.resolution_y = int(payload.get("height") or 360)
    scene.render.filepath = str(root / "blender_render" / "frame_")
    scene.render.image_settings.file_format = "PNG"

    # Wipe default cube.
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    if (frames_dir / "frame_0000.png").exists():
        img = bpy.data.images.load(str(frames_dir / "frame_0000.png"))
        img.source = "SEQUENCE"
        try:
            img.frame_duration = n
        except Exception:
            pass
        mat = bpy.data.materials.new("mujoco_seq")
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        tex.image_user.use_auto_refresh = True
        tex.image_user.frame_duration = n
        emit = nt.nodes.new("ShaderNodeEmission")
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
        nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
        bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
        plane = bpy.context.active_object
        plane.name = "mujoco_fruitfly_sequence"
        plane.data.materials.append(mat)

    title = _text(bpy, "hud_title", SIMULATION_LABEL, (-1.1, 1.05, 0.2), 0.08)
    hud = _text(bpy, "hud_series", "", (-1.1, -1.05, 0.2), 0.06)
    curve_data = bpy.data.curves.new("burden_curve", type="CURVE")
    curve_data.dimensions = "3D"
    spline = curve_data.splines.new("POLY")
    burdens = [float(r["tumor_burden"]) for r in rows]
    b0 = max(max(burdens), 1e-6)
    spline.points.add(max(len(rows) - 1, 0))
    for i, b in enumerate(burdens):
        spline.points[i].co = (i / max(n - 1, 1) * 1.6 - 0.8, -0.55 + 0.35 * (b / b0), 0.05, 1.0)
    curve_obj = bpy.data.objects.new("burden_from_ode", curve_data)
    bpy.context.collection.objects.link(curve_obj)

    def _on_frame(scene_in):
        idx = max(0, min(n - 1, int(scene_in.frame_current) - 1))
        r = rows[idx]
        hud.data.body = (
            f"{SIMULATION_LABEL}\n"
            f"t={r['t_days']:.2f}d  burden={r['tumor_burden']:.3f}  "
            f"H={r['H']:.3f}\n"
            f"fusionAF={r['fusion_allele_fraction']:.3f}  "
            f"Ab_ready={r['A_ready']:.3f}\n"
            f"logged CancerODE — not a clinical outcome"
        )
        title.data.body = SIMULATION_LABEL

    bpy.app.handlers.frame_change_post.clear()
    bpy.app.handlers.frame_change_post.append(_on_frame)
    scene.frame_set(1)
    _on_frame(scene)


def main() -> int:
    argv = _argv_after_dd()
    root = Path("docs/demo/blender")
    still_only = True
    i = 0
    while i < len(argv):
        if argv[i] == "--root" and i + 1 < len(argv):
            root = Path(argv[i + 1])
            i += 2
            continue
        if argv[i] == "--animation":
            still_only = False
            i += 1
            continue
        i += 1
    root = root.resolve()
    try:
        import bpy  # noqa: F401
    except ImportError:
        print("bpy not available — open this script in local Blender 4.x")
        print("root", root)
        return 2
    build_scene(root)
    import bpy

    out_still = root / "blender_still.png"
    bpy.context.scene.render.filepath = str(out_still)
    bpy.ops.render.render(write_still=True)
    print("wrote", out_still, SIMULATION_LABEL)
    if not still_only:
        bpy.context.scene.render.filepath = str(root / "blender_render" / "frame_")
        bpy.ops.render.render(animation=True)
        print("wrote animation under", root / "blender_render")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
