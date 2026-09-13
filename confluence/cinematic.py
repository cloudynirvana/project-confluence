"""Dark-lab cinematic renderer for the fly hero (numpy, no MuJoCo required).

Used by the offline demo clip and as the visual contract for the live canvas.
When TuragaLab/flybody is installed the live UI can overlay a real
``physics.render`` JPEG; this module is the intentional stub look.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

N_LEGS = 6
JOINTS_PER_LEG = 3

# Beige Drosophila / viral flybody still.
BODY = np.array([214, 196, 164], dtype=np.float32)
BODY_DARK = np.array([168, 148, 118], dtype=np.float32)
EYE = np.array([186, 36, 40], dtype=np.float32)
EYE_GLOSS = np.array([240, 90, 80], dtype=np.float32)
LEG = np.array([150, 132, 104], dtype=np.float32)
WING = np.array([210, 205, 195], dtype=np.float32)
AMBER = np.array([212, 160, 84], dtype=np.float32)
TEAL = np.array([62, 207, 192], dtype=np.float32)
ROSE = np.array([224, 107, 92], dtype=np.float32)


@dataclass
class Camera:
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov: float = 36.0
    near: float = 0.08

    def matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        f = self.target - self.eye
        f = f / (np.linalg.norm(f) + 1e-9)
        r = np.cross(f, self.up)
        r = r / (np.linalg.norm(r) + 1e-9)
        u = np.cross(r, f)
        view = np.array([r, u, -f], dtype=np.float32)
        return view, self.eye.astype(np.float32)


def _look(cam: Camera, pts: np.ndarray) -> np.ndarray:
    view, eye = cam.matrices()
    return (pts.astype(np.float32) - eye) @ view.T


def project(cam: Camera, pts: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    loc = _look(cam, np.atleast_2d(pts))
    z = np.maximum(-loc[:, 2], cam.near)
    f = 0.5 * h / np.tan(np.radians(cam.fov) * 0.5)
    x = w * 0.5 + f * loc[:, 0] / z
    y = h * 0.55 - f * loc[:, 1] / z
    return np.stack([x, y], axis=1), z


def _disk(rgb: np.ndarray, zbuf: np.ndarray, cx: float, cy: float, z: float, radius: float, color: np.ndarray, shade: float = 1.0) -> None:
    if radius < 0.6 or z <= 0:
        return
    h, w, _ = rgb.shape
    r = int(radius) + 1
    x0 = max(int(cx) - r, 0)
    x1 = min(int(cx) + r + 1, w)
    y0 = max(int(cy) - r, 0)
    y1 = min(int(cy) + r + 1, h)
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.ogrid[y0:y1, x0:x1]
    dx = xs.astype(np.float32) - cx
    dy = ys.astype(np.float32) - cy
    rr = dx * dx + dy * dy
    rad2 = radius * radius
    mask = rr <= rad2
    if not np.any(mask):
        return
    # Soft limb-darkening + rim.
    fall = np.sqrt(np.clip(1.0 - rr / (rad2 + 1e-6), 0.0, 1.0))
    light = (0.35 + 0.65 * fall) * shade
    col = np.clip(color[None, None, :] * light[..., None], 0, 255)
    zb = zbuf[y0:y1, x0:x1]
    vis = mask & (z < zb)
    if not np.any(vis):
        return
    dest = rgb[y0:y1, x0:x1]
    dest[vis] = col[vis]
    zb[vis] = z


def _capsule(rgb, zbuf, p0, p1, z0, z1, radius, color) -> None:
    span = max(np.hypot(p1[0] - p0[0], p1[1] - p0[1]), 1.0)
    steps = max(6, int(span / max(radius * 0.45, 0.8)))
    for i in range(steps + 1):
        t = i / steps
        _disk(
            rgb,
            zbuf,
            p0[0] + t * (p1[0] - p0[0]),
            p0[1] + t * (p1[1] - p0[1]),
            z0 + t * (z1 - z0),
            radius * (1.0 - 0.12 * t),
            color,
            shade=0.9 + 0.15 * (1 - t),
        )


def _pixel_radius(cam: Camera, z: float, world_r: float, height: int) -> float:
    f = 0.5 * height / np.tan(np.radians(cam.fov) * 0.5)
    return float(f * world_r / max(z, cam.near))


def _grid(rgb: np.ndarray, cam: Camera) -> None:
    h, w, _ = rgb.shape
    # Fade already painted; draw faint perspective lines.
    pts = []
    for i in range(-8, 9):
        pts.append([[i * 0.18, -1.6, 0.0], [i * 0.18, 1.8, 0.0]])
        pts.append([[-1.6, i * 0.18, 0.0], [1.8, i * 0.18, 0.0]])
    for a, b in pts:
        pa, za = project(cam, np.array(a), w, h)
        pb, zb = project(cam, np.array(b), w, h)
        if za[0] < 0.05 or zb[0] < 0.05:
            continue
        x0, y0 = pa[0]
        x1, y1 = pb[0]
        n = int(max(abs(x1 - x0), abs(y1 - y0), 1))
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
        for x, y in zip(xs, ys):
            xi, yi = int(x), int(y)
            if 0 <= xi < w and 0 <= yi < h:
                rgb[yi, xi] = np.minimum(rgb[yi, xi] + np.array([18, 18, 22], dtype=np.uint8), 40)


def fly_kinematics(joints: Sequence[float], heading: float, xpos, t_fly: float) -> Dict[str, np.ndarray]:
    """Build a beige Drosophila-like articulated pose in world space."""
    x, y, z = (float(xpos[0]), float(xpos[1]), float(xpos[2])) if xpos is not None else (0.0, 0.0, 0.12)
    c, s = np.cos(heading), np.sin(heading)

    def wld(local):
        lx, ly, lz = local
        return np.array([x + c * lx - s * ly, y + s * lx + c * ly, lz], dtype=np.float32)

    thorax = wld((0.02, 0.0, 0.095))
    head = wld((0.095, 0.0, 0.102))
    abdomen = wld((-0.10, 0.0, 0.085))
    eyes = [wld((0.118, 0.020, 0.110)), wld((0.118, -0.020, 0.110))]
    antennae = [
        (wld((0.125, 0.014, 0.122)), wld((0.155, 0.026, 0.148))),
        (wld((0.125, -0.014, 0.122)), wld((0.155, -0.026, 0.148))),
    ]
    wing_l0 = wld((0.00, 0.02, 0.128))
    wing_l1 = wld((-0.14, 0.10, 0.148 + 0.008 * np.sin(2.2 * t_fly)))
    wing_r0 = wld((0.00, -0.02, 0.128))
    wing_r1 = wld((-0.14, -0.10, 0.148 + 0.008 * np.sin(2.2 * t_fly + 0.4)))

    attach = [
        (0.08, 0.03, 0.07),
        (0.01, 0.036, 0.065),
        (-0.06, 0.03, 0.06),
        (0.08, -0.03, 0.07),
        (0.01, -0.036, 0.065),
        (-0.06, -0.03, 0.06),
    ]
    legs = []
    j = list(joints) + [0.0] * 18
    for i in range(N_LEGS):
        side = 1.0 if i < 3 else -1.0
        coxa = float(j[i * 3])
        femur = float(j[i * 3 + 1])
        tibia = float(j[i * 3 + 2])
        a0 = wld(attach[i])
        # Forward kinematics in a simplified sagittal/lateral plane.
        yaw = side * (0.85 + 0.45 * coxa)
        pitch = 0.35 + 0.55 * femur
        L1, L2, L3 = 0.055, 0.085, 0.075
        d1 = np.array([np.cos(yaw) * L1 * np.cos(pitch), np.sin(yaw) * L1, -L1 * np.sin(pitch)])
        a1 = a0 + np.array([c * d1[0] - s * d1[1], s * d1[0] + c * d1[1], d1[2]])
        pitch2 = pitch + 0.55 + 0.35 * tibia
        d2 = np.array([np.cos(yaw) * L2 * np.cos(pitch2), np.sin(yaw) * L2, -L2 * np.sin(pitch2)])
        a2 = a1 + np.array([c * d2[0] - s * d2[1], s * d2[0] + c * d2[1], d2[2]])
        pitch3 = pitch2 + 0.7
        d3 = np.array([np.cos(yaw) * L3 * np.cos(pitch3), np.sin(yaw) * L3, -L3 * np.sin(pitch3)])
        a3 = a2 + np.array([c * d3[0] - s * d3[1], s * d3[0] + c * d3[1], d3[2]])
        a3[2] = max(a3[2], 0.002)
        legs.append((a0, a1, a2, a3))
    return {
        "thorax": thorax,
        "head": head,
        "abdomen": abdomen,
        "eyes": eyes,
        "antennae": antennae,
        "wings": [(wing_l0, wing_l1), (wing_r0, wing_r1)],
        "legs": legs,
    }


def _vignette(rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    ys = (np.linspace(-1, 1, h) ** 2)[:, None]
    xs = (np.linspace(-1, 1, w) ** 2)[None, :]
    v = np.clip(1.0 - 0.55 * (xs + ys), 0.35, 1.0)
    rgb[:] = (rgb.astype(np.float32) * v[..., None]).astype(np.uint8)


def _sparks(rgb, zbuf, cam, thorax, mbon, da, secretory, w, h, rng) -> None:
    rates = list(mbon or []) + list(secretory or [])
    if not rates:
        rates = [0.2]
    n = min(48, 12 + 4 * len(rates))
    base = np.asarray(thorax, dtype=np.float32)
    for i in range(n):
        amp = float(rates[i % len(rates)])
        jitter = rng.normal(0.0, 0.045, size=3).astype(np.float32)
        jitter[2] += 0.04 + 0.08 * abs(amp)
        p = base + jitter
        uv, z = project(cam, p, w, h)
        col = AMBER if (da or 0) >= 0 else TEAL
        glow = 0.45 + 0.55 * min(1.0, abs(amp) * 1.6 + abs(da or 0))
        _disk(rgb, zbuf, float(uv[0, 0]), float(uv[0, 1]), float(z[0]) - 0.01, 1.6 + 2.4 * glow, col * glow)


def _hud(rgb: np.ndarray, burden: float, resist: float, da: float, proteins: Iterable[str], t_days: float) -> None:
    """A few tracked-out labels — film UI, not a dashboard."""
    h, w, _ = rgb.shape

    def stamp(x, y, color, scale=1):
        # Tiny 3×5 bitmap font for a handful of glyphs is overkill; draw bars instead.
        x = int(x)
        y = int(y)
        if 0 <= y < h and 0 <= x < w:
            rgb[y, x] = color

    # Top-left wordmark bar.
    rgb[18:20, 28:88] = AMBER.astype(np.uint8)
    # Burden / resist ticks at bottom-left.
    def bar(x, y, val, color):
        length = int(np.clip(val, 0, 1.6) / 1.6 * 90)
        rgb[y : y + 3, x : x + 90] = 28
        rgb[y : y + 3, x : x + max(length, 1)] = color.astype(np.uint8)

    bar(28, h - 48, burden, ROSE)
    bar(28, h - 38, resist, AMBER)
    bar(28, h - 28, 0.5 + 0.5 * np.clip(da, -1, 1), TEAL)
    # Protein pips.
    pips = list(proteins)[:4]
    for i, _ in enumerate(pips):
        x = 28 + i * 14
        rgb[h - 16 : h - 12, x : x + 8] = AMBER.astype(np.uint8)
    # Time tick top-right.
    rgb[18:20, w - 90 : w - 28] = np.array([80, 80, 86], dtype=np.uint8)


def render_frame(
    width: int = 1280,
    height: int = 720,
    embodiment: Optional[Dict] = None,
    connectome: Optional[Dict] = None,
    latent: Optional[Dict] = None,
    proteins: Optional[Sequence[str]] = None,
    t_days: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Return an RGB uint8 hero frame (dark lab, single fly)."""
    emb = embodiment or {}
    conn = connectome or {}
    lat = latent or {}
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    # Cool-black gradient stage.
    yy = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    rgb[:] = (
        np.array([6, 7, 10], dtype=np.float32) * (1.0 - 0.35 * yy)
        + np.array([4, 4, 6], dtype=np.float32) * (0.35 * yy)
    ).astype(np.uint8)
    zbuf = np.full((height, width), 1e6, dtype=np.float32)

    heading = float(emb.get("heading") or 0.0)
    xpos = emb.get("xpos") or (0.0, 0.0, 0.12)
    joints = emb.get("joints") or []
    t_fly = float(emb.get("t_fly") or 0.0)
    kin = fly_kinematics(joints, heading, xpos, t_fly)

    eye = np.array([0.22 + 0.55 * float(xpos[0]), -0.52, 0.24], dtype=np.float32)
    target = np.array([0.02 + 0.85 * float(xpos[0]), 0.015 * float(xpos[1]), 0.08], dtype=np.float32)
    cam = Camera(eye=eye, target=target, up=np.array([0.0, 0.0, 1.0]), fov=32.0)

    _grid(rgb, cam)

    sh, zs = project(cam, np.array([[kin["thorax"][0], kin["thorax"][1], 0.0]]), width, height)
    _disk(rgb, zbuf, float(sh[0, 0]), float(sh[0, 1]) + 6, 8.0, 26, np.array([12, 12, 14], dtype=np.float32), shade=0.55)

    # Tapered abdomen as overlapping beads so it reads as one body.
    c, s = np.cos(heading), np.sin(heading)
    abd_local = [(-0.02, 0.048), (-0.07, 0.042), (-0.12, 0.034), (-0.16, 0.024)]
    parts = []
    for lx, wr in abd_local:
        p = kin["thorax"] + np.array([c * (lx - 0.02), s * (lx - 0.02), -0.004], dtype=np.float32)
        uv, z = project(cam, p, width, height)
        parts.append((float(z[0]), "abd", uv[0], wr))
    uv, z = project(cam, kin["thorax"], width, height)
    parts.append((float(z[0]), "thor", uv[0], 0.058))
    uv, z = project(cam, kin["head"], width, height)
    parts.append((float(z[0]), "head", uv[0], 0.040))
    for e in kin["eyes"]:
        uv, z = project(cam, e, width, height)
        parts.append((float(z[0]), "eye", uv[0], 0.016))

    for a, b in kin["wings"]:
        pa, za = project(cam, a, width, height)
        pb, zb = project(cam, b, width, height)
        _capsule(rgb, zbuf, pa[0], pb[0], float(za[0]), float(zb[0]), _pixel_radius(cam, float(za[0]), 0.006, height), WING * 0.45)

    for a, b in kin["antennae"]:
        pa, za = project(cam, a, width, height)
        pb, zb = project(cam, b, width, height)
        _capsule(rgb, zbuf, pa[0], pb[0], float(za[0]), float(zb[0]), _pixel_radius(cam, float(za[0]), 0.004, height), BODY_DARK)

    for a0, a1, a2, a3 in kin["legs"]:
        chain = [a0, a1, a2, a3]
        uvs, zs = [], []
        for p in chain:
            uv, z = project(cam, p, width, height)
            uvs.append(uv[0])
            zs.append(float(z[0]))
        wr = [0.0075, 0.006, 0.0045]
        for i in range(3):
            _capsule(rgb, zbuf, uvs[i], uvs[i + 1], zs[i], zs[i + 1], _pixel_radius(cam, zs[i], wr[i], height), LEG)

    for z, kind, uv, wr in sorted(parts, key=lambda t: -t[0]):
        rad = _pixel_radius(cam, z, wr, height)
        if kind == "eye":
            _disk(rgb, zbuf, float(uv[0]), float(uv[1]), z, rad, EYE, shade=1.2)
            _disk(rgb, zbuf, float(uv[0]) - rad * 0.25, float(uv[1]) - rad * 0.25, z - 0.001, rad * 0.28, EYE_GLOSS, shade=1.35)
        elif kind == "abd":
            _disk(rgb, zbuf, float(uv[0]), float(uv[1]), z, rad, BODY * 0.92, shade=0.98)
        else:
            _disk(rgb, zbuf, float(uv[0]), float(uv[1]), z, rad, BODY, shade=1.08)

    rng = np.random.default_rng(seed + int(t_fly * 40))
    _sparks(
        rgb,
        zbuf,
        cam,
        kin["thorax"],
        conn.get("mbon_rates") or [],
        float(conn.get("da") or 0.0),
        conn.get("secretory_rates") or [],
        width,
        height,
        rng,
    )
    _vignette(rgb)
    _hud(
        rgb,
        float(lat.get("tumor_burden") or 0.0),
        float(lat.get("resistance_frequency") or 0.0),
        float(conn.get("da") or 0.0),
        proteins or [],
        t_days,
    )
    return rgb
