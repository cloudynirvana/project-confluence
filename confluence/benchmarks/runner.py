"""Run A–E controllers across archetypes × stochastic seeds."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np

from confluence.benchmarks.metrics import TrialMetrics, compute_metrics, summarize
from confluence.controllers import make_controller
from confluence.loop import ClosedLoopSimulator


DEFAULT_ARCHETYPES = ("glioblastoma", "pancreatic_pdac", "melanoma_persister")
DEFAULT_CONTROLLERS = ("A", "B", "C", "D", "E")


def run_trial(
    archetype: str,
    controller_name: str,
    seed: int,
    horizon_days: float = 80.0,
    dt: float = 0.5,
    n_kc: int = 128,
) -> TrialMetrics:
    kwargs = {}
    if controller_name in {"D", "E", "reservoir", "plastic_mb"}:
        kwargs["n_kc"] = n_kc
        kwargs["seed"] = seed
    controller = make_controller(controller_name, **kwargs)
    sim = ClosedLoopSimulator(
        archetype=archetype,
        controller=controller,
        dt=dt,
        seed=seed,
    )
    n_steps = int(horizon_days / dt)
    times, burdens, resists, health, concs, infs = [], [], [], [], [], []
    baseline = None
    for _ in range(n_steps):
        frame = sim.step()
        if baseline is None:
            baseline = frame.latent.tumor_burden
        times.append(frame.t)
        burdens.append(frame.latent.tumor_burden)
        resists.append(frame.latent.resistance_frequency)
        health.append(frame.latent.H)
        concs.append(sum(frame.concentrations.values()))
        infs.append(sum(frame.action.infusion.values()))
        if frame.terminal:
            break
    return compute_metrics(
        np.asarray(times),
        np.asarray(burdens),
        np.asarray(resists),
        np.asarray(health),
        np.asarray(concs),
        np.asarray(infs),
        archetype=archetype,
        controller=controller_name,
        seed=seed,
        baseline_burden=float(baseline or 1.0),
    )


def run_benchmark(
    archetypes: Sequence[str] = DEFAULT_ARCHETYPES,
    controllers: Sequence[str] = DEFAULT_CONTROLLERS,
    n_trials: int = 3,
    horizon_days: float = 60.0,
    dt: float = 0.5,
    n_kc: int = 96,
) -> Dict:
    rows: List[TrialMetrics] = []
    for arch in archetypes:
        for ctrl in controllers:
            for trial in range(n_trials):
                rows.append(
                    run_trial(
                        arch,
                        ctrl,
                        seed=1000 + 17 * trial + (sum(ord(c) for c in arch + ctrl) % 97),
                        horizon_days=horizon_days,
                        dt=dt,
                        n_kc=n_kc,
                    )
                )
    return {
        "trials": [r.as_dict() for r in rows],
        "summary": summarize(rows),
        "n_trials": n_trials,
        "horizon_days": horizon_days,
    }


def main(argv: Iterable[str] | None = None) -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Confluence v2 controller benchmark")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--horizon", type=float, default=40.0)
    parser.add_argument("--dt", type=float, default=0.5)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_benchmark(n_trials=args.trials, horizon_days=args.horizon, dt=args.dt)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
