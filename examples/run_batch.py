"""
CLI Batch Runner
=================

Command-line interface for running simulation batches and parameter sweeps.

Usage:
    # Run single simulation with default params
    python examples/run_batch.py

    # Run with custom parameters from JSON file
    python examples/run_batch.py --params params.json

    # Run calibration sweep against target distances
    python examples/run_batch.py --sweep --target Standard=5.76,Geometric=0.14

    # Override simulation length
    python examples/run_batch.py --days 90 --output results.json
"""

import argparse
import json
import sys
from pathlib import Path
from math import sqrt, exp
from dataclasses import dataclass, asdict


# ─── Inline Lightweight Simulation (zero external dependencies) ───────────

@dataclass(frozen=True)
class Intervention:
    name: str
    category: str
    curvature_multiplier: float = 1.0
    noise_delta: float = 0.0
    checkpoint_gain: float = 0.0


DRUG_LIBRARY = {
    "anti_pd1": Intervention("Anti-PD-1", "vector_rectifier", checkpoint_gain=0.45),
    "epogen":   Intervention("Epogen", "geometric_deepener", curvature_multiplier=1.12),
    "hyper":    Intervention("Hyperthermia", "entropic_driver", noise_delta=0.35),
    "dca":      Intervention("DCA", "curvature_reducer", curvature_multiplier=0.72),
    "metformin":Intervention("Metformin", "curvature_reducer", curvature_multiplier=0.74),
}


def _eigenvalues_2x2(A):
    a, b = A[0]
    c, d = A[1]
    tr = a + d
    det = a * d - b * c
    disc = complex(tr * tr - 4.0 * det) ** 0.5
    return (tr + disc) / 2.0, (tr - disc) / 2.0


def _curvature(A):
    eigvals = _eigenvalues_2x2(A)
    stable = [ev.real for ev in eigvals if ev.real < 0]
    return sum(abs(v) for v in stable) / len(stable) if stable else 0.0


def _resonance(A, eps=1e-6):
    return 1.0 / max(_curvature(A), eps)


class _ImmuneSys:
    def __init__(self, base_force, exhaustion_rate, treg_friction=0.2):
        self.bf = float(base_force)
        self.er = float(exhaustion_rate)
        self.tf = float(treg_friction)
        self.fr = 1.0

    def exhaust(self, depth, t):
        return exp(-self.er * t * max(depth, 0.0))

    def blockade(self, gain=0.3, fd=0.4):
        self.fr = min(2.0, self.fr + gain)
        self.tf = max(0.0, self.tf * (1.0 - fd))

    def tilt(self, depth, t):
        mag = self.bf * self.fr * self.exhaust(depth, t) * (1.0 - self.tf)
        return mag * sqrt(2.0) / 2.0  # Project onto (-1,-1) direction


def simulate(base_force, exhaustion_rate, treg_load=0.24, noise_scale=0.08, days=60):
    """Run all three protocols. Returns dict {scenario_name -> final_distance}."""
    A = [[-1.6, 0.1], [0.0, -1.2]]
    curv0 = _curvature(A)
    res0 = _resonance(A)

    def _run(protocol):
        curv = curv0
        noise = noise_scale
        immune = _ImmuneSys(base_force, exhaustion_rate, treg_load)
        dist = 6.2
        sched = {d: iv for d, iv in protocol}
        for t in range(days):
            if t in sched:
                iv = sched[t]
                curv *= iv.curvature_multiplier
                noise += iv.noise_delta
                if iv.checkpoint_gain > 0:
                    immune.blockade(gain=iv.checkpoint_gain, fd=0.4)
            bd = 0.01 * (1 + res0 * noise)
            ip = immune.tilt(curv, t)
            bh = 0.18 if (curv < 1.0 and noise > 0.35 and immune.fr > 1.2) else 0.0
            dist += curv * 0.004 - bd - ip * 0.11 - bh
            dist = max(0.0, dist)
        return round(dist, 4)

    D = DRUG_LIBRARY
    return {
        "Standard":   _run([(0, D["anti_pd1"])]),
        "Iatrogenic": _run([(0, D["epogen"]), (1, D["anti_pd1"])]),
        "Geometric":  _run([(0, D["dca"]), (5, D["metformin"]),
                            (20, D["hyper"]), (35, D["anti_pd1"])]),
    }


# ─── Calibration Grid Search (inline, no imports from src) ────────────────

def _squared_error(pred, target):
    return sum((pred.get(k, 0.0) - v) ** 2 for k, v in target.items())


def grid_search(target, days=60):
    """Coarse grid search over immune/noise parameters."""
    best_score = float("inf")
    best_params = {}

    forces = [0.5, 0.7, 0.85, 1.0, 1.3]
    exhaustions = [0.04, 0.08, 0.12, 0.16]
    tregs = [0.18, 0.24, 0.30]
    noises = [0.06, 0.08, 0.10, 0.15]

    total = len(forces) * len(exhaustions) * len(tregs) * len(noises)
    count = 0

    for bf in forces:
        for er in exhaustions:
            for tl in tregs:
                for ns in noises:
                    count += 1
                    try:
                        pred = simulate(bf, er, tl, ns, days)
                        score = _squared_error(pred, target)
                    except Exception:
                        score = float("inf")
                    if score < best_score:
                        best_score = score
                        best_params = {
                            "base_force": bf,
                            "exhaustion_rate": er,
                            "treg_load": tl,
                            "noise_scale": ns,
                            "score": round(score, 8),
                        }

    print(f"[sweep] Evaluated {count}/{total} sets, best score: {best_score:.8f}")
    return best_params


# ─── CLI ───────────────────────────────────────────────────────────────────

def parse_target(s: str) -> dict:
    """Parse 'Standard=5.76,Geometric=0.14' into dict."""
    out = {}
    for pair in s.split(","):
        k, v = pair.strip().split("=")
        out[k.strip()] = float(v.strip())
    return out


def main():
    parser = argparse.ArgumentParser(
        description="SAEM Geometric Cure: Batch Simulation & Calibration CLI",
    )
    parser.add_argument("--params", type=str, default=None,
                        help="JSON file with base_force, exhaustion_rate, treg_load, noise_scale")
    parser.add_argument("--sweep", action="store_true",
                        help="Run calibration grid search")
    parser.add_argument("--target", type=str, default=None,
                        help="Calibration targets: 'Standard=5.76,Geometric=0.14'")
    parser.add_argument("--days", type=int, default=60,
                        help="Simulation duration in days (default: 60)")
    parser.add_argument("--output", type=str, default=None,
                        help="Write results to JSON file")
    args = parser.parse_args()

    print("=" * 60)
    print("  SAEM Geometric Cure — Batch Runner")
    print("=" * 60)

    if args.sweep:
        # Calibration mode
        if args.target:
            target = parse_target(args.target)
        else:
            target = {"Standard": 5.76, "Iatrogenic": 6.10, "Geometric": 0.14}

        print(f"\n[mode] CALIBRATION SWEEP (target: {target})")
        best = grid_search(target, days=args.days)
        print(f"\nBest parameters: {json.dumps(best, indent=2)}")

        # Run with best params
        pred = simulate(best["base_force"], best["exhaustion_rate"],
                        best["treg_load"], best["noise_scale"], args.days)
        print(f"\nPredicted distances: {json.dumps(pred, indent=2)}")
        results = {"mode": "sweep", "best_params": best, "predictions": pred}

    else:
        # Single simulation mode
        bf, er, tl, ns = 0.85, 0.12, 0.24, 0.08  # Defaults

        if args.params:
            with open(args.params) as f:
                p = json.load(f)
            bf = p.get("base_force", bf)
            er = p.get("exhaustion_rate", er)
            tl = p.get("treg_load", tl)
            ns = p.get("noise_scale", ns)

        print(f"\n[mode] SINGLE RUN (bf={bf}, er={er}, tl={tl}, ns={ns}, days={args.days})")
        pred = simulate(bf, er, tl, ns, args.days)

        for name, dist in pred.items():
            status = "SUCCESS" if dist < 1.0 else "FAIL"
            print(f"  {name:15s} -> {status:7s} (dist: {dist:.4f})")

        results = {"mode": "single", "params": {"base_force": bf, "exhaustion_rate": er,
                   "treg_load": tl, "noise_scale": ns}, "predictions": pred}

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[output] Results written to {args.output}")

    print()
    return results


if __name__ == "__main__":
    main()
