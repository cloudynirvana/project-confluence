"""Closed-loop training: observe → sparse net → protein+drug U → ODE → Y.

Inner update is the existing dopaminergic three-factor rule on the
secretory readout. An optional (1+1)-ES outer loop proposes weight
perturbations and keeps the better episode.

This trains simulated infusion / expression rates. It does not synthesize
proteins and is not a clinical optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from confluence.controllers.full_brain import FullBrainController
from confluence.loop import ClosedLoopSimulator
from confluence.neural_engine.full_brain import INTERACTIVE_BRAIN_NEURONS
from confluence.training.checkpoint import save_checkpoint


@dataclass
class TrainConfig:
    n_neurons: int = INTERACTIVE_BRAIN_NEURONS
    episodes: int = 3
    days: float = 80.0
    dt: float = 0.5
    archetype: str = "glioblastoma"
    seed: int = 0
    outer: str = "both"  # da | es | both
    checkpoint: str = "results/full_brain/ckpt.npz"
    embodiment: bool = False
    sigma: float = 0.05
    es_clip: float = 1.5


@dataclass
class EpisodeLog:
    episode: int
    reward: float
    mean_da: float
    mean_burden: float
    mean_toxicity: float
    final_burden: float
    final_health: float
    protein_active: List[str]
    steps: int
    kept: bool


@dataclass
class TrainResult:
    config: TrainConfig
    episodes: List[EpisodeLog] = field(default_factory=list)
    checkpoint: Optional[str] = None
    best_reward: float = -1e9

    def as_dict(self) -> Dict:
        return {
            "n_neurons": self.config.n_neurons,
            "episodes": len(self.episodes),
            "best_reward": self.best_reward,
            "checkpoint": self.checkpoint,
            "history": [e.__dict__ for e in self.episodes],
        }


def step_reward(latent, tox_load: float) -> float:
    """Dense shaping used by the outer loop (in-silico score only)."""
    return float(
        -1.2 * latent.tumor_burden
        - 0.9 * latent.resistance_frequency
        - 0.6 * max(0.0, 0.7 - latent.H)
        - 0.08 * tox_load
        + 0.10 * latent.I_act
    )


def run_live_episode(
    sim: ClosedLoopSimulator,
    days: float,
    collect: bool = True,
) -> Dict:
    """Train / roll one episode on an existing simulator (UI + CLI)."""
    sim.reset()
    n_steps = max(1, int(round(days / max(sim.dt, 1e-6))))
    rewards = []
    das = []
    burdens = []
    toxes = []
    last = None
    for _ in range(n_steps):
        last = sim.step(run_cancer=True, run_embodiment=bool(sim.embodiment_enabled))
        tox = sim.pk.toxicity_load(sim.c)
        rewards.append(step_reward(last.latent, tox))
        tel = last.connectome or {}
        das.append(float(tel.get("da") or 0.0))
        burdens.append(last.latent.tumor_burden)
        toxes.append(tox)
        if last.terminal:
            rewards[-1] -= 5.0
            break
    tel = (last.connectome if last else {}) or {}
    return {
        "reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_da": float(np.mean(das)) if das else 0.0,
        "mean_burden": float(np.mean(burdens)) if burdens else 0.0,
        "mean_toxicity": float(np.mean(toxes)) if toxes else 0.0,
        "final_burden": float(last.latent.tumor_burden) if last else 0.0,
        "final_health": float(last.latent.H) if last else 0.0,
        "protein_active": list(tel.get("protein_active") or []),
        "steps": len(rewards),
        "t": float(last.t) if last else 0.0,
    }


def train(config: Optional[TrainConfig] = None) -> TrainResult:
    config = config or TrainConfig()
    if config.outer not in {"da", "es", "both"}:
        raise ValueError("outer must be da, es, or both")
    rng = np.random.default_rng(config.seed)
    controller = FullBrainController(n_neurons=config.n_neurons, seed=config.seed)
    if config.outer == "es":
        controller.network.config.plastic = False
    sim = ClosedLoopSimulator(
        archetype=config.archetype,
        controller=controller,
        dt=config.dt,
        seed=config.seed,
        embodiment_enabled=config.embodiment,
    )
    result = TrainResult(config=config)
    net = controller.network
    best_snap = net.snapshot_weights()
    best_reward = -1e18

    for ep in range(config.episodes):
        start_snap = net.snapshot_weights()
        proposed = False
        if config.outer in {"es", "both"}:
            noise = rng.normal(0.0, config.sigma, size=net.w_sec.shape).astype(np.float32)
            net.w_sec = np.clip(
                net.w_sec + noise, -config.es_clip, config.es_clip
            ).astype(np.float32)
            proposed = True
        metrics = run_live_episode(sim, days=config.days)
        reward = metrics["reward"]
        kept = True
        if proposed and reward < best_reward:
            net.restore_weights(start_snap)
            kept = False
        elif reward >= best_reward:
            best_reward = reward
            best_snap = net.snapshot_weights()
            kept = True
        log = EpisodeLog(
            episode=ep,
            reward=reward,
            mean_da=metrics["mean_da"],
            mean_burden=metrics["mean_burden"],
            mean_toxicity=metrics["mean_toxicity"],
            final_burden=metrics["final_burden"],
            final_health=metrics["final_health"],
            protein_active=list(metrics["protein_active"]),
            steps=metrics["steps"],
            kept=kept,
        )
        result.episodes.append(log)
        ckpt = save_checkpoint(
            net,
            config.checkpoint,
            extra={
                "episode": np.int64(ep),
                "reward": np.float64(best_reward),
            },
        )
        result.checkpoint = str(ckpt)

    net.restore_weights(best_snap)
    result.best_reward = float(best_reward)
    if result.checkpoint:
        save_checkpoint(
            net,
            result.checkpoint,
            extra={
                "episode": np.int64(max(0, config.episodes - 1)),
                "reward": np.float64(result.best_reward),
            },
        )
    return result
