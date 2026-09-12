"""Optional bridge from Confluence motor decode to DeepMind/Janelia flybody.

Real API (TuragaLab/flybody, Apache 2.0), verified from upstream source:

    from flybody.fly_envs import walk_imitation, template_task, flight_imitation
    env = walk_imitation()          # walking imitation, action dim 59
    env = template_task()           # lightest no-op walking task (smoke tests)
    timestep = env.step(action)     # dm_env TimeStep
    pixels = env.physics.render(camera_id=1)

This module never imports flybody at package import time. If MuJoCo / flybody
are missing, ``FlybodyBridge`` uses a documented kinematic stub so the
interactive UI still streams pose telemetry.

Motor map (documented affine clip):
    features = concat(U ∈ R^5, MBON rates)
    features = features / (1 + |features|)
    a = W @ features + b
    action = clip(a, low, high)

    W is (action_dim, n_features), seeded, not a trained policy.
    This is a thin linear readout, not a biomechanical inverse model.

Sensory map:
    joints / touch / vestibular observables (or stub CPG angles) are pooled
    into a 5-D vector that can be mixed into Y before W_in:

    Y_mix = (1 − α) Y_cancer + α Y_proprio    (default α = 0.25)

Clocks are independent: cancer steps in days; flybody walking control is
~20 ms (``_WALK_CONTROL_TIMESTEP``). Play/pause/step are shared; physics time
is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

FLYBODY_REPO = "https://github.com/TuragaLab/flybody"
FLYBODY_COMMIT = "d015e9bfe441bd90ae431bac24c55cb74bdbce26"
WALK_ACTION_DIM = 59  # official README walk_imitation example
FLIGHT_ACTION_DIM = 22
SENSORY_DIM = 5
N_LEGS = 6
JOINTS_PER_LEG = 3

# Default walking action bounds used by the stub and as a fallback when
# dm_env specs are unavailable. Real env bounds come from action_spec.
DEFAULT_ACTION_LOW = -1.0
DEFAULT_ACTION_HIGH = 1.0


def flybody_available() -> bool:
    try:
        import flybody.fly_envs  # noqa: F401
        return True
    except Exception:
        return False


def flybody_status() -> Dict[str, Any]:
    ok = flybody_available()
    return {
        "available": ok,
        "repo": FLYBODY_REPO,
        "commit": FLYBODY_COMMIT,
        "license": "Apache-2.0",
        "citation": "Vaxenburg et al., Nature 643:1312–1320 (2025) doi:10.1038/s41586-025-09029-4",
        "backend": "flybody" if ok else "kinematic_stub",
    }


def _flatten_obs(obj: Any, prefix: str = "") -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if isinstance(obj, dict):
        for key, val in obj.items():
            out.update(_flatten_obs(val, f"{prefix}{key}/"))
        return out
    arr = np.asarray(obj, dtype=float).ravel()
    out[prefix.rstrip("/")] = arr
    return out


def _pick_vector(flat: Dict[str, np.ndarray], suffixes: Sequence[str]) -> Optional[np.ndarray]:
    for name, vec in flat.items():
        low = name.lower()
        if any(low.endswith(s) or s in low for s in suffixes) and vec.size:
            return vec
    return None


@dataclass
class EmbodimentTelemetry:
    t_fly: float
    backend: str
    task: str
    action: np.ndarray
    action_rms: float
    reward: float
    discount: float
    last: bool
    sensory: np.ndarray
    joints: np.ndarray
    contacts: np.ndarray
    xpos: Tuple[float, float, float]
    heading: float
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        joints = [float(v) for v in self.joints[:24]]
        action_ds = self.action
        if action_ds.size > 32:
            stride = action_ds.size // 32
            action_ds = action_ds[: stride * 32].reshape(32, stride).mean(axis=1)
        return {
            "t_fly": float(self.t_fly),
            "backend": self.backend,
            "task": self.task,
            "action_dim": int(self.action.size),
            "action_rms": float(self.action_rms),
            "action": [float(v) for v in action_ds],
            "reward": float(self.reward),
            "discount": float(self.discount),
            "last": bool(self.last),
            "sensory": [float(v) for v in self.sensory],
            "joints": joints,
            "contacts": [float(v) for v in self.contacts],
            "xpos": [float(v) for v in self.xpos],
            "heading": float(self.heading),
            "notes": self.notes,
        }


class MotorMap:
    """Affine map from (U, MBON) → flybody action, clipped to env bounds."""

    def __init__(
        self,
        action_dim: int,
        n_u: int = 5,
        n_mbon: int = 8,
        seed: int = 11,
        low: float = DEFAULT_ACTION_LOW,
        high: float = DEFAULT_ACTION_HIGH,
    ):
        self.action_dim = int(action_dim)
        self.n_u = int(n_u)
        self.n_mbon = int(n_mbon)
        self.low = float(low)
        self.high = float(high)
        rng = np.random.default_rng(seed)
        n_in = self.n_u + self.n_mbon
        # Structured tiles so each input channel fans out across legs/wings.
        self.W = rng.normal(0.0, 0.15, size=(self.action_dim, n_in))
        for i in range(n_in):
            self.W[i :: max(n_in, 1), i] += 0.55
        self.b = rng.normal(0.0, 0.02, size=self.action_dim)

    def features(self, u: Sequence[float], mbon: Sequence[float]) -> np.ndarray:
        u = np.asarray(u, dtype=float).ravel()
        mbon = np.asarray(mbon, dtype=float).ravel()
        u_p = np.zeros(self.n_u, dtype=float)
        m_p = np.zeros(self.n_mbon, dtype=float)
        u_p[: min(self.n_u, u.size)] = u[: self.n_u]
        m_p[: min(self.n_mbon, mbon.size)] = mbon[: self.n_mbon]
        feat = np.concatenate([u_p, m_p])
        return feat / (1.0 + np.abs(feat))

    def __call__(self, u: Sequence[float], mbon: Sequence[float]) -> np.ndarray:
        a = self.W @ self.features(u, mbon) + self.b
        return np.clip(a, self.low, self.high)


class _KinematicStub:
    """Six-leg CPG walker used when flybody/MuJoCo is not installed."""

    def __init__(self, seed: int = 0):
        self.action_dim = WALK_ACTION_DIM
        self.t = 0.0
        self.dt = 0.02
        self.xy = np.zeros(2, dtype=float)
        self.heading = 0.0
        self.rng = np.random.default_rng(seed)
        self.phases = np.linspace(0, 2 * np.pi, N_LEGS, endpoint=False)
        # Tripod gait offset: even vs odd legs.
        self.phases[1::2] += np.pi

    def reset(self) -> None:
        self.t = 0.0
        self.xy[:] = 0.0
        self.heading = 0.0

    def step(self, action: np.ndarray) -> EmbodimentTelemetry:
        drive = float(np.clip(np.mean(np.abs(action[:12])), 0.0, 1.0))
        turn = float(np.clip(action[0] - action[1], -1.0, 1.0))
        self.heading += 0.08 * turn
        speed = 0.015 * (0.25 + drive)
        self.xy += speed * np.array([np.cos(self.heading), np.sin(self.heading)])
        self.t += self.dt
        joints = []
        contacts = []
        for i in range(N_LEGS):
            ph = self.phases[i] + 8.0 * self.t
            coxa = 0.35 * np.sin(ph)
            femur = 0.55 * np.sin(ph + 0.6)
            tibia = 0.40 * np.sin(ph + 1.1)
            joints.extend([coxa, femur, tibia])
            contacts.append(1.0 if np.sin(ph) < 0.0 else 0.0)
        joints_a = np.asarray(joints, dtype=float)
        contacts_a = np.asarray(contacts, dtype=float)
        sensory = np.array(
            [
                float(np.mean(np.abs(joints_a))),
                float(np.mean(contacts_a)),
                float(self.xy[0]),
                float(self.heading),
                drive,
            ],
            dtype=float,
        )
        return EmbodimentTelemetry(
            t_fly=self.t,
            backend="kinematic_stub",
            task="cpg_walk",
            action=np.asarray(action, dtype=float),
            action_rms=float(np.sqrt(np.mean(np.square(action)))),
            reward=float(drive),
            discount=1.0,
            last=False,
            sensory=sensory,
            joints=joints_a,
            contacts=contacts_a,
            xpos=(float(self.xy[0]), float(self.xy[1]), 0.12),
            heading=float(self.heading),
            notes="Kinematic CPG stub — install flybody extra for MuJoCo physics.",
        )


class FlybodyBridge:
    """Construct a flybody env if present, else a kinematic stub.

    Parameters
    ----------
    task:
        ``walk_imitation`` (default demo), ``template`` (lightest real env),
        or ``flight_imitation``.
    prefer_real:
        If True and flybody imports, use the real composer Environment.
    """

    def __init__(
        self,
        task: str = "walk_imitation",
        prefer_real: bool = True,
        seed: int = 7,
        n_mbon: int = 8,
    ):
        self.task_name = task
        self.seed = seed
        self._env = None
        self._timestep = None
        self.backend = "kinematic_stub"
        self.notes = ""
        action_dim = WALK_ACTION_DIM if "flight" not in task else FLIGHT_ACTION_DIM
        low, high = DEFAULT_ACTION_LOW, DEFAULT_ACTION_HIGH

        if prefer_real and flybody_available():
            try:
                self._env = self._make_real_env(task, seed)
                spec = self._env.action_spec()
                action_dim = int(np.prod(spec.shape))
                low = float(np.min(spec.minimum))
                high = float(np.max(spec.maximum))
                self._timestep = self._env.reset()
                self.backend = "flybody"
                self.notes = (
                    f"TuragaLab/flybody@{FLYBODY_COMMIT[:7]} task={task} "
                    f"action_dim={action_dim}"
                )
            except Exception as exc:
                self._env = None
                self.notes = f"flybody import succeeded but env failed ({exc}); using stub."

        self.action_dim = action_dim
        self.mapper = MotorMap(
            action_dim=action_dim, n_mbon=n_mbon, seed=seed, low=low, high=high
        )
        self._stub = _KinematicStub(seed=seed)
        self.last: Optional[EmbodimentTelemetry] = None
        if self._env is None and not self.notes:
            self.notes = (
                "flybody not installed. "
                f"pip install -e \".[flybody]\"  # pins {FLYBODY_REPO}@{FLYBODY_COMMIT[:7]}"
            )

    def _make_real_env(self, task: str, seed: int):
        from flybody.fly_envs import flight_imitation, template_task, walk_imitation

        rng = np.random.RandomState(seed)
        if task in {"template", "template_task"}:
            return template_task(random_state=rng)
        if task in {"flight", "flight_imitation"}:
            return flight_imitation(random_state=rng)
        # Default: walking imitation in inference mode (no HDF5 dataset).
        return walk_imitation(random_state=rng, terminal_com_dist=float("inf"))

    def reset(self) -> EmbodimentTelemetry:
        self._stub.reset()
        if self._env is not None:
            self._timestep = self._env.reset()
        idle = np.zeros(self.action_dim, dtype=float)
        self.last = self._observe(idle, reward=0.0, discount=1.0, last=False)
        return self.last

    def step(
        self,
        u: Sequence[float],
        mbon_rates: Optional[Sequence[float]] = None,
    ) -> EmbodimentTelemetry:
        action = self.mapper(u, [] if mbon_rates is None else mbon_rates)
        if self._env is None:
            self.last = self._stub.step(action)
            return self.last
        try:
            self._timestep = self._env.step(action)
            ts = self._timestep
            if bool(getattr(ts, "last", False)):
                self._timestep = self._env.reset()
                ts = self._timestep
            reward = float(ts.reward or 0.0)
            discount = float(ts.discount if ts.discount is not None else 1.0)
            self.last = self._observe(action, reward, discount, bool(getattr(ts, "last", False)))
        except Exception as exc:
            self.last = self._stub.step(action)
            self.last.notes = f"real step failed ({exc}); stub fallback"
        return self.last

    def _observe(
        self,
        action: np.ndarray,
        reward: float,
        discount: float,
        last: bool,
    ) -> EmbodimentTelemetry:
        joints = np.zeros(N_LEGS * JOINTS_PER_LEG)
        contacts = np.zeros(N_LEGS)
        sensory = np.zeros(SENSORY_DIM)
        xpos = (0.0, 0.0, 0.12)
        heading = 0.0
        t_fly = float(self._stub.t)
        if self._env is not None and self._timestep is not None:
            flat = _flatten_obs(self._timestep.observation)
            jp = _pick_vector(flat, ("joints_pos", "joints/pos", "qpos"))
            touch = _pick_vector(flat, ("touch", "force", "contact"))
            gyro = _pick_vector(flat, ("gyro", "velocimeter"))
            if jp is not None:
                joints = jp[: N_LEGS * JOINTS_PER_LEG]
                if joints.size < N_LEGS * JOINTS_PER_LEG:
                    joints = np.pad(joints, (0, N_LEGS * JOINTS_PER_LEG - joints.size))
            if touch is not None:
                contacts = np.clip(np.abs(touch[:N_LEGS]), 0.0, 1.0)
            pool = []
            for vec in (jp, touch, gyro):
                if vec is not None and vec.size:
                    pool.append(float(np.mean(vec)))
                    pool.append(float(np.std(vec)))
            if pool:
                sensory = np.zeros(SENSORY_DIM)
                sensory[: min(SENSORY_DIM, len(pool))] = pool[:SENSORY_DIM]
            try:
                physics = self._env.physics
                t_fly = float(physics.data.time)
                root = np.asarray(physics.data.qpos[:7], dtype=float)
                xpos = (float(root[0]), float(root[1]), float(root[2]))
                # qpos[3:7] is a unit quaternion (w, x, y, z); yaw from that.
                if root.size >= 7:
                    w, x, y, z = root[3:7]
                    heading = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
            except Exception:
                pass
        return EmbodimentTelemetry(
            t_fly=t_fly,
            backend=self.backend,
            task=self.task_name,
            action=np.asarray(action, dtype=float),
            action_rms=float(np.sqrt(np.mean(np.square(action)))),
            reward=reward,
            discount=discount,
            last=last,
            sensory=sensory,
            joints=np.asarray(joints, dtype=float),
            contacts=np.asarray(contacts, dtype=float),
            xpos=xpos,
            heading=heading,
            notes=self.notes,
        )

    def mix_observation(self, y: Sequence[float], alpha: float = 0.25) -> np.ndarray:
        """Blend cancer Y with last proprio embedding for W_in."""
        y = np.asarray(y, dtype=float).ravel()
        if self.last is None:
            return y
        proprio = self.last.sensory
        n = min(y.size, proprio.size)
        mixed = y.copy()
        mixed[:n] = (1.0 - alpha) * y[:n] + alpha * proprio[:n]
        return mixed
