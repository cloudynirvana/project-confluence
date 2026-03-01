"""
Optimal Correction Calibration
================================

Calibrates the Geometric Achievement Protocol parameters to find
the optimal correction step — the exact immune/geometric/noise
settings that maximize escape probability across all cancer types.

Outputs:
  - Calibrated parameters (base_force, exhaustion_rate, treg_load, noise_scale)
  - Per-cancer-type correction vectors
  - Codex-ready handoff payload with optimal starting conditions
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tnbc_ode import TNBCODESystem, METABOLITES, simulate_trajectory
from geometric_optimization import GeometricOptimizer, TherapeuticProtocolOptimizer
from intervention import InterventionMapper, DrugEfficiencyEngine
from immune_dynamics import LymphocyteForceField, ImmuneParams
from coherence import CoherenceAnalyzer
from restoration import RestorationComputer
from calibration_targets import build_scenario_targets, STABILITY_TARGETS, SPARSE_CORRECTIONS


def run_scenario_suite(base_force, exhaustion_rate, treg_load, noise_scale):
    """Run all 3 scenarios and return final distances."""
    A_tnbc = TNBCODESystem.tnbc_generator()
    n = 10
    mapper = InterventionMapper()
    lib = {i.name: i for i in mapper.intervention_library}
    optimizer = GeometricOptimizer(n)

    # Initial state deep in attractor
    val, vec = np.linalg.eig(A_tnbc)
    idx = np.argsort(val.real)
    x0 = np.real(vec[:, idx[0]]) * 5.0

    results = {}
    for scenario_name, protocol_steps in [
        ("Standard", [(0, 60, [lib["Anti-PD-1 (Pembrolizumab)"]])]),
        ("Iatrogenic", [(0, 60, [lib["Epogen (Epoetin alfa)"], lib["Anti-PD-1 (Pembrolizumab)"]])]),
        ("Geometric", [
            (0, 25, [lib["Dichloroacetate (DCA)"], lib["Metformin"]]),
            (20, 25, [lib["Entropic Heating (Hyperthermia)"]]),
            (25, 60, [lib["Anti-PD-1 (Pembrolizumab)"], lib["Dichloroacetate (DCA)"], lib["Metformin"]]),
        ]),
    ]:
        x = x0.copy()
        n_days = 60
        dt = 0.1
        steps = int(n_days / dt)
        immune = LymphocyteForceField(n, ImmuneParams(
            base_force=base_force,
            exhaustion_rate=exhaustion_rate,
            treg_load=treg_load,
        ))
        rng = np.random.default_rng(42)

        for i in range(steps):
            t = i * dt
            A_eff = A_tnbc.copy()
            noise = noise_scale

            for start, end, drugs in protocol_steps:
                if start <= t <= end:
                    for drug in drugs:
                        A_eff += drug.expected_effect
                        if drug.entropic_driver > 0:
                            noise *= drug.entropic_driver
                        if drug.immune_modifiers:
                            if 'pd1_blockade' in drug.immune_modifiers:
                                immune.params.pd1_blockade = max(
                                    immune.params.pd1_blockade,
                                    drug.immune_modifiers['pd1_blockade']
                                )
                            if 'ctla4_blockade' in drug.immune_modifiers:
                                immune.params.ctla4_blockade = max(
                                    immune.params.ctla4_blockade,
                                    drug.immune_modifiers['ctla4_blockade']
                                )

            mu = optimizer.compute_basin_curvature(A_eff)
            f = immune.compute_net_force(x, mu, dt)
            x += (A_eff @ x + f) * dt + rng.standard_normal(n) * noise * np.sqrt(dt)

        results[scenario_name] = float(np.linalg.norm(x))
        immune.reset()

    return results


def calibrate_optimal():
    """Run grid search + refinement to find optimal correction parameters."""
    print("=" * 60)
    print("OPTIMAL CORRECTION CALIBRATION")
    print("=" * 60)

    target = build_scenario_targets()
    print(f"\nTargets: {target}")

    # Coarse grid
    best_score = float('inf')
    best_params = None

    force_vals = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5]
    exhaust_vals = [0.04, 0.08, 0.10, 0.12, 0.16]
    treg_vals = [0.15, 0.24, 0.30, 0.40]
    noise_vals = [0.06, 0.08, 0.10, 0.15]

    total = len(force_vals) * len(exhaust_vals) * len(treg_vals) * len(noise_vals)
    evaluated = 0

    print(f"\nPhase 1: Coarse grid search ({total} parameter sets)...")

    for bf in force_vals:
        for er in exhaust_vals:
            for tl in treg_vals:
                for ns in noise_vals:
                    evaluated += 1
                    try:
                        pred = run_scenario_suite(bf, er, tl, ns)
                        score = sum((pred.get(k, 0) - v) ** 2 for k, v in target.items())
                    except Exception:
                        score = float('inf')

                    if score < best_score:
                        best_score = score
                        best_params = {
                            'base_force': bf,
                            'exhaustion_rate': er,
                            'treg_load': tl,
                            'noise_scale': ns,
                            'score': score,
                        }
                        if evaluated % 50 == 0:
                            print(f"  [{evaluated}/{total}] New best: score={score:.4f} params={best_params}")

    print(f"\n  Coarse best: score={best_params['score']:.6f}")
    print(f"    base_force={best_params['base_force']}")
    print(f"    exhaustion_rate={best_params['exhaustion_rate']}")
    print(f"    treg_load={best_params['treg_load']}")
    print(f"    noise_scale={best_params['noise_scale']}")

    # Local refinement
    print(f"\nPhase 2: Local refinement...")
    center = best_params
    shrink = 0.25
    ref_n = 4

    def linspace_around(c, frac, n, lo=0.01):
        half = c * frac
        return [max(lo, c - half + (2 * half / max(n - 1, 1)) * i) for i in range(n)]

    for _ in range(2):
        for bf in linspace_around(center['base_force'], shrink, ref_n):
            for er in linspace_around(center['exhaustion_rate'], shrink, ref_n, 0.001):
                for tl in linspace_around(center['treg_load'], shrink, ref_n, 0.0):
                    for ns in linspace_around(center['noise_scale'], shrink, ref_n, 0.01):
                        try:
                            pred = run_scenario_suite(bf, er, tl, ns)
                            score = sum((pred.get(k, 0) - v) ** 2 for k, v in target.items())
                        except Exception:
                            score = float('inf')

                        if score < best_params['score']:
                            best_params = {
                                'base_force': bf,
                                'exhaustion_rate': er,
                                'treg_load': tl,
                                'noise_scale': ns,
                                'score': score,
                            }
        shrink *= 0.5

    print(f"\n  Refined best: score={best_params['score']:.6f}")
    print(f"    base_force={best_params['base_force']:.4f}")
    print(f"    exhaustion_rate={best_params['exhaustion_rate']:.4f}")
    print(f"    treg_load={best_params['treg_load']:.4f}")
    print(f"    noise_scale={best_params['noise_scale']:.4f}")

    return best_params


def compute_pan_cancer_corrections(params):
    """Compute optimal correction vectors for each cancer type."""
    print("\n" + "=" * 60)
    print("PAN-CANCER CORRECTION ANALYSIS")
    print("=" * 60)

    A_healthy = TNBCODESystem.healthy_generator()
    generators = TNBCODESystem.pan_cancer_generators()
    optimizer = GeometricOptimizer(10)
    restorer = RestorationComputer(sparsity_weight=0.1)
    coherence = CoherenceAnalyzer()

    corrections = {}

    for name, A_cancer in generators.items():
        print(f"\n  {name}:")

        # Direct correction
        delta_A = restorer.compute_direct_correction(A_cancer, A_healthy)

        # Sparse correction (top 5)
        delta_sparse, targets = restorer.compute_sparse_correction(A_cancer, A_healthy, max_interventions=5)

        # Analyze correction effect
        effect = restorer.analyze_correction_effect(A_cancer, delta_sparse)

        # Basin geometry before/after
        curv_before = optimizer.compute_basin_curvature(A_cancer)
        curv_after = optimizer.compute_basin_curvature(A_cancer + delta_sparse)

        # Kramers escape rate before/after
        esc_before = optimizer.compute_kramers_escape_rate(A_cancer, params['noise_scale'])
        esc_after = optimizer.compute_kramers_escape_rate(
            A_cancer + delta_sparse, params['noise_scale'], params['base_force']
        )

        # Coherence score
        coh_before = coherence.analyze(A_cancer, A_healthy)['overall_score']
        coh_after = coherence.analyze(A_cancer + delta_sparse, A_healthy)['overall_score']

        print(f"    Curvature:   {curv_before:.4f} → {curv_after:.4f}  (Δ={curv_before - curv_after:.4f})")
        print(f"    Escape Rate: {esc_before:.2e} → {esc_after:.2e}  ({esc_after/max(esc_before, 1e-20):.1f}x)")
        print(f"    Coherence:   {coh_before:.4f} → {coh_after:.4f}")
        print(f"    Top targets: {[(METABOLITES[i], METABOLITES[j], f'{delta_sparse[i,j]:.3f}') for i, j in targets[:3]]}")

        corrections[name] = {
            'curvature_before': float(curv_before),
            'curvature_after': float(curv_after),
            'escape_rate_before': float(esc_before),
            'escape_rate_after': float(esc_after),
            'coherence_before': float(coh_before),
            'coherence_after': float(coh_after),
            'sparse_targets': [(int(i), int(j), float(delta_sparse[i, j])) for i, j in targets],
            'stability_improvement': float(effect['stability_improvement']),
            'is_stabilizing': bool(effect['is_stabilizing']),
        }

    return corrections


def main():
    # Step 1: Calibrate
    params = calibrate_optimal()

    # Step 2: Compute pan-cancer corrections with calibrated params
    corrections = compute_pan_cancer_corrections(params)

    # Step 3: Assemble Codex handoff payload
    payload = {
        'calibrated_params': params,
        'pan_cancer_corrections': corrections,
        'target_outcomes': build_scenario_targets(),
        'stability_targets': STABILITY_TARGETS,
        'sparse_correction_reference': {f"{i},{j}": v for (i, j), v in SPARSE_CORRECTIONS.items()},
        'metabolite_names': METABOLITES,
    }

    out_path = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_path, exist_ok=True)

    with open(os.path.join(out_path, 'calibrated_corrections.json'), 'w') as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 60)
    print("CODEX HANDOFF PAYLOAD")
    print("=" * 60)
    print(f"  Calibrated Parameters:")
    print(f"    base_force      = {params['base_force']:.4f}")
    print(f"    exhaustion_rate = {params['exhaustion_rate']:.4f}")
    print(f"    treg_load       = {params['treg_load']:.4f}")
    print(f"    noise_scale     = {params['noise_scale']:.4f}")
    print(f"    calibration_err = {params['score']:.6f}")
    print(f"\n  Cancer Corrections:")
    for name, corr in corrections.items():
        status = "✓" if corr['is_stabilizing'] else "✗"
        print(f"    {status} {name}: curvΔ={corr['curvature_before'] - corr['curvature_after']:.4f}, "
              f"esc={corr['escape_rate_after']:.2e}, coh={corr['coherence_after']:.3f}")
    print(f"\n  Saved to: results/calibrated_corrections.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
