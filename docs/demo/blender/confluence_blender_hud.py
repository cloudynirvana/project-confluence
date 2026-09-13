"""Blender 4.x headless HUD for logged Confluence scientific frames.

    blender --background --python docs/demo/blender/confluence_blender_hud.py -- \\
        --root docs/demo/blender

    blender --background --python docs/demo/blender/confluence_blender_hud.py -- \\
        --root docs/demo/blender --animation

Always burns in SIMULATION / RESEARCH. Loads MuJoCo PNG sequence + telemetry.json.
Not a clinical film. Not generative tumor-shrink footage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SIMULATION_LABEL = "SIMULATION / RESEARCH"
NON_CLAIM = "logged CancerODE — not a clinical outcome"


def _argv_after_dd() -> list:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def _parse(argv: list) -> dict:
    root = Path("docs/demo/blender")
    animation = False
    max_frames = 12
    i = 0
    while i < len(argv):
        if argv[i] == "--root" and i + 1 < len(argv):
            root = Path(argv[i + 1])
            i += 2
            continue
        if argv[i] == "--animation":
            animation = True
            i += 1
            continue
        if argv[i] == "--max-frames" and i + 1 < len(argv):
            max_frames = int(argv[i + 1])
            i += 2
            continue
        i += 1
    return {"root": root.resolve(), "animation": animation, "max_frames": max_frames}


def _load(root: Path) -> dict:
    path = root / "telemetry.json"
    if not path.exists():
        raise SystemExit(
            "telemetry.json missing — run: python3 -m confluence.demo_blender --out docs/demo/blender"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("rows"):
        raise SystemExit("telemetry.json has no rows")
    return payload


def _hud_stamp(row: dict) -> str:
    return (
        f"{SIMULATION_LABEL}  |  t={row['t_days']:.2f}d  "
        f"burden={row['tumor_burden']:.3f}  H={row['H']:.3f}  "
        f"fusionAF={row['fusion_allele_fraction']:.3f}  "
        f"Ab_ready={row['A_ready']:.3f}  |  {NON_CLAIM}"
    )


def _clear_scene(bpy) -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _add_camera_and_world(bpy, scene) -> None:
    cam_data = bpy.data.cameras.new("viz_cam")
    cam_data.lens = 35
    cam = bpy.data.objects.new("viz_cam", cam_data)
    bpy.context.collection.objects.link(cam)
    # New cameras look along local −Z. Sit above the XY billboard.
    cam.location = (0.0, 0.0, 3.8)
    cam.rotation_euler = (0.0, 0.0, 0.0)
    scene.camera = cam
    world = bpy.data.worlds.new("dark_lab")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.02, 0.03, 0.05, 1.0)
        bg.inputs[1].default_value = 0.3
    scene.world = world
    scene.render.engine = "BLENDER_WORKBENCH"
    shading = scene.display.shading
    shading.light = "FLAT"
    shading.color_type = "TEXTURE"
    scene.render.film_transparent = False
    scene.render.use_stamp = True
    scene.render.use_stamp_note = True
    scene.render.stamp_note_text = SIMULATION_LABEL
    scene.render.stamp_font_size = 18
    scene.render.use_stamp_camera = False
    scene.render.use_stamp_lens = False
    scene.render.use_stamp_scene = False
    scene.render.use_stamp_filename = False
    scene.render.use_stamp_date = False
    scene.render.use_stamp_time = False
    scene.render.use_stamp_render_time = False
    scene.render.use_stamp_frame = False
    scene.render.use_stamp_marker = False
    scene.render.use_stamp_sequencer_strip = False


def build_scene(root: Path, max_frames: int) -> int:
    import bpy

    payload = _load(root)
    rows = payload["rows"]
    n = min(len(rows), max(1, int(max_frames)))
    frames_dir = root / "frames"
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = n
    scene.render.fps = int(payload.get("fps") or 12)
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.resolution_percentage = 100
    renders = root / "renders"
    renders.mkdir(parents=True, exist_ok=True)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(renders / "frame_")

    _clear_scene(bpy)
    _add_camera_and_world(bpy, scene)

    has_png = (frames_dir / "frame_0000.png").exists()
    if has_png:
        img = bpy.data.images.load(str(frames_dir / "frame_0000.png"))
        img.source = "SEQUENCE"
        try:
            img.frame_duration = len(rows)
        except Exception:
            pass
        mat = bpy.data.materials.new("mujoco_seq")
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        tex.image_user.use_auto_refresh = True
        tex.image_user.frame_duration = len(rows)
        emit = nt.nodes.new("ShaderNodeEmission")
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
        nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
        bpy.ops.mesh.primitive_plane_add(size=2.2, location=(0.0, 0.15, 0.0))
        plane = bpy.context.active_object
        plane.name = "mujoco_fruitfly_sequence"
        if plane.data.materials:
            plane.data.materials[0] = mat
        else:
            plane.data.materials.append(mat)
    else:
        scene.render.stamp_note_text = (
            f"{SIMULATION_LABEL}  |  INSTALL FLYBODY  |  "
            "python3 -m confluence.demo_blender --require-mesh"
        )

    curve_data = bpy.data.curves.new("burden_curve", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = 0.004
    spline = curve_data.splines.new("POLY")
    burdens = [float(r["tumor_burden"]) for r in rows[:n]]
    b0 = max(max(burdens), 1e-6)
    spline.points.add(max(len(burdens) - 1, 0))
    for i, b in enumerate(burdens):
        spline.points[i].co = (
            i / max(len(burdens) - 1, 1) * 1.8 - 0.9,
            -0.62 + 0.28 * (b / b0),
            0.04,
            1.0,
        )
    curve_obj = bpy.data.objects.new("burden_from_ode", curve_data)
    bpy.context.collection.objects.link(curve_obj)

    def _on_frame(scene_in):
        idx = max(0, min(len(rows) - 1, int(scene_in.frame_current) - 1))
        note = _hud_stamp(rows[idx])
        if not has_png:
            note = f"{note}  |  INSTALL FLYBODY"
        scene_in.render.stamp_note_text = note

    bpy.app.handlers.frame_change_post.clear()
    bpy.app.handlers.frame_change_post.append(_on_frame)
    scene.frame_set(1)
    _on_frame(scene)
    return n


def main() -> int:
    opts = _parse(_argv_after_dd())
    root = opts["root"]
    try:
        import bpy  # noqa: F401
    except ImportError:
        print("bpy not available — open this script in local Blender 4.x")
        print("one-liner:")
        print(
            "  blender --background --python docs/demo/blender/confluence_blender_hud.py "
            "-- --root docs/demo/blender"
        )
        print("root", root)
        return 2
    n = build_scene(root, opts["max_frames"])
    import bpy

    renders = root / "renders"
    renders.mkdir(parents=True, exist_ok=True)
    still = renders / "blender_still.png"
    bpy.context.scene.render.filepath = str(still)
    bpy.ops.render.render(write_still=True)
    print("wrote", still, SIMULATION_LABEL)
    if opts["animation"]:
        bpy.context.scene.frame_end = n
        bpy.context.scene.render.filepath = str(renders / "anim_")
        bpy.ops.render.render(animation=True)
        print("wrote short animation", n, "frames under", renders)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
