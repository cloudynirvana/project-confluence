"""Controller C — thin PPO stub.

Full on-policy training against the 11-D microenvironment is intentionally
not bundled: a meaningful PPO run wants thousands of closed-loop rollouts
and a GPU-friendly vector env. This module ships:

  * a stable heuristic actor used by default
  * an optional torch policy load path (`CONFLUENCE_PPO_CKPT`)
  * `train_stub()` — a few dummy PPO-shaped updates so the interface is real

See README §PPO for how to attach a trained policy later.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from confluence.contracts import InterventionAction, ObservationRecord
from confluence.controllers.base import BaseController, ControllerContext


class PPOController(BaseController):
    letter = "C"
    name = "PPO (stub)"

    def __init__(self, checkpoint: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint = checkpoint or os.environ.get("CONFLUENCE_PPO_CKPT")
        self.policy = self._try_load(self.checkpoint)
        self.trained_steps = 0

    def _try_load(self, path: Optional[str]):
        if not path:
            return None
        try:
            import torch
        except Exception:
            return None
        try:
            return torch.load(path, map_location="cpu")
        except Exception:
            return None

    def _heuristic(self, observation: ObservationRecord) -> np.ndarray:
        """Bounded actor used when no trained weights are present."""
        b = float(np.clip(observation.tumor_burden / 3.0, 0.0, 1.0))
        r = float(np.clip(observation.resistance_frequency, 0.0, 1.0))
        lac = float(np.clip(observation.lactate / 2.0, 0.0, 1.0))
        tgfb = float(np.clip(observation.tgfb / 2.0, 0.0, 1.0))
        immune = float(np.clip(1.0 - observation.immune_competence_ratio, 0.0, 1.0))
        u = np.array(
            [
                0.55 * immune + 0.20 * b,
                0.65 * tgfb,
                0.55 * lac,
                0.70 * r,
                0.50 * b * (1.0 - 0.4 * r),
            ],
            dtype=float,
        )
        if observation.host_toxicity_warning:
            u *= 0.35
        return np.clip(u, 0.0, 1.0)

    def decide(self, observation: ObservationRecord, context: ControllerContext) -> InterventionAction:
        if self.policy is not None:
            try:
                import torch

                y = torch.tensor(observation.as_vector(), dtype=torch.float32)
                with torch.no_grad():
                    raw = self.policy(y) if callable(self.policy) else self.policy["actor"](y)
                u = np.clip(np.asarray(raw).reshape(-1)[: len(self.drug_ids)], 0.0, 1.0)
                return self._action(observation.t, u, source="C", notes="torch policy")
            except Exception:
                pass
        u = self._heuristic(observation)
        return self._action(observation.t, u, source="C", notes="heuristic stub")

    def train_stub(self, steps: int = 8, seed: int = 0) -> int:
        """Tiny PPO-shaped update on synthetic (obs, act, adv) tuples.

        Demonstrates the training surface. Not a substitute for a real
        rollout/GAE/clip loop. Returns the number of dummy steps applied.
        """
        rng = np.random.default_rng(seed)
        w = rng.normal(0.0, 0.1, size=(len(self.drug_ids), 5))
        for _ in range(steps):
            obs = rng.random(5)
            act = np.clip(w @ obs, 0.0, 1.0)
            adv = float(rng.normal())
            # policy-gradient-like increment (no value baseline, no clip)
            w += 1e-3 * adv * np.outer(act - 0.5, obs)
        self.trained_steps += steps
        return steps
