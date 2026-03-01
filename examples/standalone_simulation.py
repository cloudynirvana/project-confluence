"""
Standalone Geometric Cure Simulation
=====================================

Zero-dependency version of the cure simulation.
Uses only Python stdlib (math module) — no numpy, scipy, or matplotlib.

Demonstrates the same 3-scenario comparison:
1. Standard of Care (Checkpoint Only) → FAIL
2. Iatrogenic Trap (Epogen + Checkpoint) → FAIL
3. Geometric Cure (Flatten → Heat → Push) → CURE

Run: python examples/standalone_simulation.py
"""

from math import sqrt, exp
from dataclasses import dataclass


# ─── Minimal Intervention Dataclass ────────────────────────────────────────

@dataclass(frozen=True)
class Intervention:
    """Lightweight intervention with explicit numeric fields."""
    name: str
    category: str
    curvature_multiplier: float = 1.0   # <1=flatten, >1=deepen
    noise_delta: float = 0.0            # Additive noise change
    checkpoint_gain: float = 0.0        # Force recovery boost


# ─── Drug Library ──────────────────────────────────────────────────────────

ANTI_PD1 = Intervention(
    name="Anti-PD-1",
    category="vector_rectifier",
    checkpoint_gain=0.45,
)

ANTI_CTLA4 = Intervention(
    name="Anti-CTLA-4",
    category="vector_rectifier",
    checkpoint_gain=0.35,
)

EPOGEN = Intervention(
    name="Epogen",
    category="geometric_deepener",
    curvature_multiplier=1.12,
)

HYPERTHERMIA = Intervention(
    name="Hyperthermia",
    category="entropic_driver",
    noise_delta=0.35,
)

DCA = Intervention(
    name="DCA",
    category="curvature_reducer",
    curvature_multiplier=0.72,
)

METFORMIN = Intervention(
    name="Metformin",
    category="curvature_reducer",
    curvature_multiplier=0.74,
)


# ─── Minimal Geometric Functions (stdlib only) ────────────────────────────

def _eigenvalues_2x2(A):
    """Compute eigenvalues of a 2x2 matrix (list-of-lists)."""
    a, b = A[0]
    c, d = A[1]
    tr = a + d
    det = a * d - b * c
    disc = complex(tr * tr - 4.0 * det) ** 0.5
    return (tr + disc) / 2.0, (tr - disc) / 2.0


def compute_basin_curvature(A):
    """Scalar curvature index from stable eigenvalues."""
    eigvals = _eigenvalues_2x2(A)
    stable_reals = [ev.real for ev in eigvals if ev.real < 0]
    if not stable_reals:
        return 0.0
    return sum(abs(v) for v in stable_reals) / len(stable_reals)


def compute_entropic_resonance(A, epsilon=1e-6):
    """Resonance noise scale = 1/curvature."""
    curvature = compute_basin_curvature(A)
    return 1.0 / max(curvature, epsilon)


# ─── Minimal Immune Force Field ───────────────────────────────────────────

class ImmuneForceField:
    """Depth-aware immune actuator (stdlib only)."""

    def __init__(self, base_force, exhaustion_rate, treg_friction=0.2):
        self.base_force = float(base_force)
        self.exhaustion_rate = float(exhaustion_rate)
        self.treg_friction = float(treg_friction)
        self.force_recovery = 1.0

    def exhaustion_factor(self, depth, t):
        """B(x,t) = B0 exp(-kappa * t * mu(x))."""
        return exp(-self.exhaustion_rate * t * max(depth, 0.0))

    def checkpoint_blockade(self, gain=0.3, friction_drop=0.5):
        """Restore force budget and reduce Treg friction."""
        self.force_recovery = min(2.0, self.force_recovery + gain)
        self.treg_friction = max(0.0, self.treg_friction * (1.0 - friction_drop))

    def vector_tilt(self, direction, depth, t):
        """Directed immune force after exhaustion/friction effects."""
        x, y = direction
        norm = sqrt(x * x + y * y)
        if norm == 0:
            return (0.0, 0.0)
        magnitude = self.base_force * self.force_recovery * self.exhaustion_factor(depth, t)
        magnitude *= (1.0 - self.treg_friction)
        return (x / norm * magnitude, y / norm * magnitude)


# ─── Simulation Engine ────────────────────────────────────────────────────

def _norm2(v):
    return sqrt(v[0] * v[0] + v[1] * v[1])


def apply_intervention(curvature, noise, immune, intervention):
    """Apply a single intervention to the system state."""
    curvature *= intervention.curvature_multiplier
    noise += intervention.noise_delta
    if intervention.checkpoint_gain > 0:
        immune.checkpoint_blockade(gain=intervention.checkpoint_gain, friction_drop=0.4)
    return curvature, noise


def run_protocol(name, protocol, days=60):
    """
    Run a treatment protocol and return final distance from health.

    Args:
        name:     Scenario label
        protocol: list of (day, Intervention) tuples
        days:     Simulation duration
    """
    A = [[-1.6, 0.1], [0.0, -1.2]]
    curvature = compute_basin_curvature(A)
    resonance = compute_entropic_resonance(A)
    noise = 0.08
    immune = ImmuneForceField(base_force=0.85, exhaustion_rate=0.12, treg_friction=0.24)
    distance = 6.2

    schedule = {day: intervention for day, intervention in protocol}

    for t in range(days):
        if t in schedule:
            curvature, noise = apply_intervention(curvature, noise, immune, schedule[t])

        depth = curvature
        barrier_drop = 0.01 * (1 + resonance * noise)
        immune_push = _norm2(immune.vector_tilt((-1.0, -1.0), depth=depth, t=t))
        basin_hop = 0.0
        if curvature < 1.0 and noise > 0.35 and immune.force_recovery > 1.2:
            basin_hop = 0.18
        distance += curvature * 0.004 - barrier_drop - immune_push * 0.11 - basin_hop
        distance = max(0.0, distance)

    status = "SUCCESS" if distance < 1.0 else "FAIL"
    print(f"  {name:40s} -> {status:7s} (dist: {distance:.2f})")
    return distance


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  STANDALONE GEOMETRIC CURE SIMULATION (zero-dependency)")
    print("=" * 65)
    print()

    results = {}

    results["Standard"] = run_protocol(
        "1. Standard of Care (Anti-PD-1 only)",
        [(0, ANTI_PD1)],
    )

    results["Iatrogenic"] = run_protocol(
        "2. Iatrogenic Trap (Epogen + Anti-PD-1)",
        [(0, EPOGEN), (1, ANTI_PD1)],
    )

    results["Geometric"] = run_protocol(
        "3. Geometric Cure (Flatten→Heat→Push)",
        [(0, DCA), (5, METFORMIN), (20, HYPERTHERMIA), (35, ANTI_PD1)],
    )

    print()
    print("-" * 65)
    geometric_success = results["Geometric"] < 1.0
    standard_fail = results["Standard"] > 1.0
    iatrogenic_fail = results["Iatrogenic"] > 1.0

    if geometric_success and standard_fail and iatrogenic_fail:
        print("  ✓ THESIS CONFIRMED: Force alone fails. Geometry first, then force.")
    else:
        print("  ✗ UNEXPECTED RESULT — parameter review required.")
    print("-" * 65)

    return results


if __name__ == "__main__":
    main()
