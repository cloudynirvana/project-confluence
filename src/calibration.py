"""
Parameter Calibration Module
=============================

Provides grid-search and local-refinement utilities for fitting
the Geometric Achievement Protocol parameters against target
clinical trajectories.

Usage:
    from calibration import coarse_grid_search, refine_around_best

    # simulate_fn(base_force, exhaustion_rate) -> dict[str, float]
    best = coarse_grid_search(simulate_fn, target)
    refined = refine_around_best(simulate_fn, target, best)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class CalibrationResult:
    """Result of a single parameter-set evaluation."""
    base_force: float
    exhaustion_rate: float
    treg_load: float = 0.3
    noise_scale: float = 0.1
    score: float = float("inf")  # Lower is better (squared error)


def squared_error(predicted: Dict[str, float], target: Dict[str, float]) -> float:
    """Compute sum-of-squares error between predicted and target scenario outcomes."""
    return sum((predicted.get(k, 0.0) - v) ** 2 for k, v in target.items())


def coarse_grid_search(
    simulate_fn: Callable[..., Dict[str, float]],
    target: Dict[str, float],
    force_range: Optional[List[float]] = None,
    exhaustion_range: Optional[List[float]] = None,
    treg_range: Optional[List[float]] = None,
    noise_range: Optional[List[float]] = None,
) -> CalibrationResult:
    """
    Run a coarse grid search over immune/geometric parameters.

    Args:
        simulate_fn: Must accept (base_force, exhaustion_rate, treg_load, noise_scale)
                     and return a dict[str, float] of scenario-name -> final-distance.
        target:      Target outcomes, e.g. {"Standard": 5.76, "Geometric": 0.14}
        force_range:      Grid values for base_force (default: 5 points in [0.5, 1.5])
        exhaustion_range: Grid values for exhaustion_rate (default: 5 points in [0.04, 0.16])
        treg_range:       Grid values for treg_load (default: [0.2, 0.3, 0.4])
        noise_range:      Grid values for noise_scale (default: [0.08, 0.1, 0.15])

    Returns:
        CalibrationResult with the best parameter set found.
    """
    if force_range is None:
        force_range = [0.5, 0.7, 0.85, 1.0, 1.5]
    if exhaustion_range is None:
        exhaustion_range = [0.04, 0.08, 0.1, 0.12, 0.16]
    if treg_range is None:
        treg_range = [0.2, 0.3, 0.4]
    if noise_range is None:
        noise_range = [0.08, 0.1, 0.15]

    best = CalibrationResult(base_force=0.0, exhaustion_rate=0.0)

    total = len(force_range) * len(exhaustion_range) * len(treg_range) * len(noise_range)
    evaluated = 0

    for bf in force_range:
        for er in exhaustion_range:
            for tl in treg_range:
                for ns in noise_range:
                    evaluated += 1
                    try:
                        pred = simulate_fn(bf, er, tl, ns)
                        score = squared_error(pred, target)
                    except Exception:
                        score = float("inf")

                    if score < best.score:
                        best = CalibrationResult(
                            base_force=bf,
                            exhaustion_rate=er,
                            treg_load=tl,
                            noise_scale=ns,
                            score=score,
                        )

    print(f"[Calibration] Evaluated {evaluated}/{total} parameter sets")
    print(f"[Calibration] Best score: {best.score:.6f}")
    print(f"[Calibration]   base_force={best.base_force}, exhaustion_rate={best.exhaustion_rate}")
    print(f"[Calibration]   treg_load={best.treg_load}, noise_scale={best.noise_scale}")

    return best


def refine_around_best(
    simulate_fn: Callable[..., Dict[str, float]],
    target: Dict[str, float],
    coarse_best: CalibrationResult,
    resolution: int = 5,
    shrink_factor: float = 0.3,
) -> CalibrationResult:
    """
    Local refinement around the coarse-grid best result.

    Creates a fine grid centered on `coarse_best` with ±shrink_factor relative span.

    Args:
        simulate_fn:   Same signature as coarse_grid_search
        target:        Target outcomes
        coarse_best:   Result from coarse_grid_search
        resolution:    Number of grid points per axis (default 5)
        shrink_factor: How much ±% of the center value to explore (default 0.3 = 30%)

    Returns:
        Refined CalibrationResult.
    """
    def _linspace(center: float, frac: float, n: int, lo: float = 0.01) -> List[float]:
        half = center * frac
        start = max(lo, center - half)
        stop = center + half
        if n <= 1:
            return [center]
        step = (stop - start) / (n - 1)
        return [start + i * step for i in range(n)]

    force_range = _linspace(coarse_best.base_force, shrink_factor, resolution)
    exhaust_range = _linspace(coarse_best.exhaustion_rate, shrink_factor, resolution, lo=0.001)
    treg_range = _linspace(coarse_best.treg_load, shrink_factor, resolution, lo=0.0)
    noise_range = _linspace(coarse_best.noise_scale, shrink_factor, resolution, lo=0.01)

    print(f"\n[Refinement] Searching {resolution**4} points around best...")

    refined = coarse_grid_search(
        simulate_fn, target,
        force_range=force_range,
        exhaustion_range=exhaust_range,
        treg_range=treg_range,
        noise_range=noise_range,
    )

    # Keep the better of coarse vs refined
    if coarse_best.score <= refined.score:
        return coarse_best
    return refined


def calibrate_full(
    simulate_fn: Callable[..., Dict[str, float]],
    target: Dict[str, float],
    n_refinements: int = 2,
) -> CalibrationResult:
    """
    Full calibration pipeline: coarse search + N rounds of refinement.

    Args:
        simulate_fn: Simulation function
        target:      Target outcomes
        n_refinements: Number of refinement passes (default 2)

    Returns:
        Best CalibrationResult found.
    """
    print("=" * 50)
    print("CALIBRATION: Coarse Grid Search")
    print("=" * 50)
    best = coarse_grid_search(simulate_fn, target)

    for i in range(n_refinements):
        print(f"\n{'=' * 50}")
        print(f"CALIBRATION: Refinement Pass {i + 1}/{n_refinements}")
        print(f"{'=' * 50}")
        best = refine_around_best(simulate_fn, target, best, shrink_factor=0.2 / (i + 1))

    print(f"\n{'=' * 50}")
    print(f"CALIBRATION COMPLETE")
    print(f"  Best Score:          {best.score:.8f}")
    print(f"  base_force:          {best.base_force:.4f}")
    print(f"  exhaustion_rate:     {best.exhaustion_rate:.4f}")
    print(f"  treg_load:           {best.treg_load:.4f}")
    print(f"  noise_scale:         {best.noise_scale:.4f}")
    print(f"{'=' * 50}")

    return best
