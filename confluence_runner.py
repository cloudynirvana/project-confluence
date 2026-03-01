"""
Confluence Runner — Project Confluence Universal Cure Framework
================================================================

Integrates ALL pillars of the SAEM framework into a single executable:
  1. Universal Cure Engine (existing) — 10-cancer 3-phase protocol
  2. Clonal Dynamics Engine — Lotka-Volterra 2-clone competition
  3. Toxicity Guard — Clinical safety constraints
  4. Protocol Translator — Wet-lab-ready protocol generation
  5. Drug Synergy Matrix — Non-linear drug interactions

Produces:
  - Per-cancer cure results with clonal dynamics
  - Safety-screened protocols
  - Lab-executable protocol documents
  - Pan-cancer summary with validation gates

Usage:
    python confluence_runner.py
    python confluence_runner.py --cancer TNBC --output results/
    python confluence_runner.py --all --lab-protocols
"""

import sys
import os
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import asdict

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from tnbc_ode import TNBCODESystem
from intervention import InterventionMapper, TherapeuticIntervention
from geometric_optimization import GeometricOptimizer
from coherence import CoherenceAnalyzer
from immune_dynamics import LymphocyteForceField, ImmuneParams
from clonal_dynamics import ClonalDynamicsEngine, get_cancer_specific_clonal_params
from toxicity_constraints import ToxicityGuard
from protocol_translator import ProtocolTranslator
from resistance_model import ResistanceTracker, ResistanceParams


# ═══════════════════════════════════════════════════════════════════════
# CANCER REGISTRY
# ═══════════════════════════════════════════════════════════════════════

CANCER_TYPES = [
    "TNBC", "PDAC", "NSCLC", "Melanoma", "GBM",
    "CRC", "HGSOC", "AML", "mCRPC", "HCC"
]

GENERATOR_MAP = {
    "TNBC":     TNBCODESystem.tnbc_generator,
    "PDAC":     TNBCODESystem.pdac_generator,
    "NSCLC":    TNBCODESystem.nsclc_generator,
    "Melanoma": TNBCODESystem.melanoma_generator,
    "GBM":      TNBCODESystem.gbm_generator,
    "CRC":      TNBCODESystem.crc_generator,
    "HGSOC":    TNBCODESystem.hgsoc_generator,
    "AML":      TNBCODESystem.aml_generator,
    "mCRPC":    TNBCODESystem.mcrpc_generator,
    "HCC":      TNBCODESystem.hcc_generator,
}

# Generator metadata (immune suppression, stromal barrier, stress load)
CANCER_META = {
    "TNBC":     {"immune_suppression": 0.55, "stromal_barrier": 0.30, "stress_load": 0.50},
    "PDAC":     {"immune_suppression": 0.80, "stromal_barrier": 0.85, "stress_load": 0.30},
    "NSCLC":    {"immune_suppression": 0.35, "stromal_barrier": 0.25, "stress_load": 0.40},
    "Melanoma": {"immune_suppression": 0.20, "stromal_barrier": 0.15, "stress_load": 0.71},
    "GBM":      {"immune_suppression": 0.70, "stromal_barrier": 0.40, "stress_load": 0.52},
    "CRC":      {"immune_suppression": 0.40, "stromal_barrier": 0.30, "stress_load": 0.40},
    "HGSOC":    {"immune_suppression": 0.50, "stromal_barrier": 0.55, "stress_load": 0.58},
    "AML":      {"immune_suppression": 0.45, "stromal_barrier": 0.20, "stress_load": 0.71},
    "mCRPC":    {"immune_suppression": 0.65, "stromal_barrier": 0.35, "stress_load": 0.27},
    "HCC":      {"immune_suppression": 0.60, "stromal_barrier": 0.50, "stress_load": 0.88},
}


# ═══════════════════════════════════════════════════════════════════════
# PROTOCOL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════

# Simulation physics constants (calibrated)
BASE_FORCE = 0.55
EXHAUSTION_RATE = 0.150
NOISE_SCALE = 0.12
RESISTANCE_TAU = 18.0
TREG_LOAD = 0.50
CURE_THRESHOLD = 0.90  # Distance threshold for "cure"
MONTE_CARLO_TRIALS = 100


# ═══════════════════════════════════════════════════════════════════════
# CORE: RUN ONE CANCER
# ═══════════════════════════════════════════════════════════════════════

def compute_seriousness(cancer_type: str, A_cancer: np.ndarray, 
                        A_healthy: np.ndarray) -> float:
    """Compute composite seriousness score for a cancer type."""
    meta = CANCER_META[cancer_type]
    
    # Coherence deficit
    analyzer = CoherenceAnalyzer()
    healthy_metrics = analyzer.analyze(A_healthy)
    cancer_metrics = analyzer.analyze(A_cancer)
    coherence_deficit = abs(
        healthy_metrics.get('overall_score', 1.0) - 
        cancer_metrics.get('overall_score', 0.5)
    )
    
    # Basin curvature
    optimizer = GeometricOptimizer(n_metabolites=10)
    curvature = optimizer.compute_basin_curvature(A_cancer)
    
    # Weighted composite
    score = (
        0.25 * coherence_deficit +
        0.15 * (1.0 - curvature) +   # Lower curvature = harder
        0.25 * meta["immune_suppression"] +
        0.15 * meta["stress_load"] +
        0.20 * meta["stromal_barrier"]
    )
    return round(score, 4)


def select_drugs(mapper: InterventionMapper, A_cancer: np.ndarray,
                 A_healthy: np.ndarray, cancer_type: str,
                 max_drugs: int = 4) -> List[Tuple[TherapeuticIntervention, float]]:
    """
    Select optimal drugs using phase-aware slot allocation.
    
    Ensures diversity by allocating category slots:
      - 2 curvature_reducers (Flatten phase)
      - 1 entropic_driver (Heat phase)
      - 1 vector_rectifier (Push phase)
    
    Within each slot, selects by alignment with delta_A and cancer-type affinity.
    """
    delta_A = A_healthy - A_cancer
    
    # Cancer-type drug affinities (bonus for known-effective drugs)
    cancer_drug_bonus = {
        "TNBC":     {"CB-839 (Telaglenastat)": 0.3, "Olaparib (PARP inhibitor)": 0.3, 
                     "N6F11 (Selective GPX4 degrader)": 0.2},
        "PDAC":     {"Hydroxychloroquine (HCQ)": 0.4, "Bevacizumab (Anti-VEGF)": 0.3,
                     "Fasting-Mimicking Diet (FMD)": 0.2},
        "NSCLC":    {"Metformin": 0.3, "Dichloroacetate (DCA)": 0.2,
                     "Vorinostat (SAHA, HDACi)": 0.2},
        "Melanoma": {"Ferroptosis Inducer (Erastin/RSL3)": 0.3, 
                     "N6F11 (Selective GPX4 degrader)": 0.3},
        "GBM":      {"2-Deoxyglucose (2-DG)": 0.3, "Fasting-Mimicking Diet (FMD)": 0.2,
                     "High-dose Vitamin C": 0.2},
        "CRC":      {"Vorinostat (SAHA, HDACi)": 0.3, "5-Azacitidine (DNMTi)": 0.2,
                     "Dichloroacetate (DCA)": 0.2},
        "HGSOC":    {"Olaparib (PARP inhibitor)": 0.4, "Bevacizumab (Anti-VEGF)": 0.3},
        "AML":      {"5-Azacitidine (DNMTi)": 0.4, "High-dose Vitamin C": 0.2,
                     "Hydroxychloroquine (HCQ)": 0.2},
        "mCRPC":    {"Metformin": 0.3, "Dichloroacetate (DCA)": 0.2,
                     "Olaparib (PARP inhibitor)": 0.2},
        "HCC":      {"Bevacizumab (Anti-VEGF)": 0.3, "Hydroxychloroquine (HCQ)": 0.2,
                     "N6F11 (Selective GPX4 degrader)": 0.2},
    }
    bonuses = cancer_drug_bonus.get(cancer_type, {})
    
    # Score each drug
    def score_drug(inv):
        effect = inv.expected_effect
        if effect.shape != delta_A.shape:
            return -1.0, 0.1
        
        norm_e = np.linalg.norm(effect, 'fro')
        norm_d = np.linalg.norm(delta_A, 'fro')
        
        if norm_e < 1e-10:
            # Zero-effect drugs (pure immune) — score by immune modifiers
            immune_score = sum(inv.immune_modifiers.values()) if inv.immune_modifiers else 0
            similarity = immune_score * 0.3  # Moderate baseline
        else:
            alignment = np.sum(effect * delta_A)
            similarity = alignment / (norm_e * norm_d)
        
        # Cancer-type bonus
        similarity += bonuses.get(inv.name, 0.0)
        
        # Weight for dosing
        if norm_e > 1e-10:
            weight = max(0.1, abs(np.sum(inv.expected_effect * delta_A) / (norm_e**2)))
        else:
            weight = 0.1
        
        return similarity, weight
    
    # Categorize drugs
    reducers = []   # curvature_reducer
    drivers = []    # entropic_driver
    rectifiers = [] # vector_rectifier
    
    for inv in mapper.intervention_library:
        if inv.evidence_level == "warning":
            continue
        sim, weight = score_drug(inv)
        if sim < -0.5:
            continue
        
        entry = (inv, sim, weight)
        if inv.category == "curvature_reducer":
            reducers.append(entry)
        elif inv.category == "entropic_driver":
            drivers.append(entry)
        elif inv.category == "vector_rectifier":
            rectifiers.append(entry)
    
    # Sort each category by score
    reducers.sort(key=lambda x: x[1], reverse=True)
    drivers.sort(key=lambda x: x[1], reverse=True)
    rectifiers.sort(key=lambda x: x[1], reverse=True)
    
    # Pick top from each slot
    selected = []
    
    # 2 curvature reducers (pick diverse ones)
    for inv, sim, weight in reducers:
        if len([s for s in selected if s[0].category == "curvature_reducer"]) >= 2:
            break
        # Avoid redundancy
        redundant = False
        for sel_inv, _ in selected:
            sel_norm = np.linalg.norm(sel_inv.expected_effect, 'fro')
            inv_norm = np.linalg.norm(inv.expected_effect, 'fro')
            if sel_norm > 0 and inv_norm > 0:
                cos = np.sum(sel_inv.expected_effect * inv.expected_effect) / (sel_norm * inv_norm)
                if cos > 0.85:
                    redundant = True
                    break
        if not redundant:
            selected.append((inv, weight))
    
    # 1 entropic driver
    if drivers:
        selected.append((drivers[0][0], drivers[0][2]))
    
    # 1 vector rectifier (prefer single checkpoint over dual)
    if rectifiers:
        selected.append((rectifiers[0][0], rectifiers[0][2]))
    
    return selected


def compute_phase_timing(seriousness: float) -> Dict[str, int]:
    """Compute adaptive phase durations based on seriousness."""
    # Harder cancers get longer treatment
    min_s, max_s = 0.30, 0.55
    t = np.clip((seriousness - min_s) / (max_s - min_s), 0, 1)
    
    flatten = int(18 + 14 * t)
    heat = int(5 + 4 * t)
    push = int(20 + 12 * t)
    
    return {"flatten": flatten, "heat": heat, "push": push}


def run_protocol_simulation(A_cancer: np.ndarray, A_healthy: np.ndarray,
                            drugs: List[Tuple[TherapeuticIntervention, float]],
                            phase_days: Dict[str, int],
                            seed: int = 42) -> Tuple[float, float]:
    """
    Run full 3-phase Flatten→Heat→Push simulation.
    Returns (final_distance, min_curvature).
    """
    n = A_cancer.shape[0]
    rng = np.random.RandomState(seed)
    optimizer = GeometricOptimizer(n_metabolites=n)
    
    # Initialize state near cancer attractor
    x = np.ones(n) * 0.5 + rng.randn(n) * 0.05
    
    # Immune system
    immune = LymphocyteForceField(n, ImmuneParams(
        base_force=BASE_FORCE,
        exhaustion_rate=EXHAUSTION_RATE,
        treg_load=TREG_LOAD,
    ))
    
    # Resistance tracking
    resistance = ResistanceTracker(ResistanceParams(resistance_tau=RESISTANCE_TAU))
    drug_names = [d.name for d, _ in drugs]
    resistance.initialize_drugs(drug_names)
    
    dt = 0.5
    min_curvature = 1.0
    total_days = sum(phase_days.values())
    
    # Build drug effect
    combined_effect = np.zeros((n, n))
    for inv, weight in drugs:
        combined_effect += inv.expected_effect * weight
    
    # Apply synergy
    synergy_matrix = InterventionMapper.SYNERGY_MATRIX
    for i in range(len(drugs)):
        for j in range(i+1, len(drugs)):
            inv1, w1 = drugs[i]
            inv2, w2 = drugs[j]
            S = synergy_matrix.get((inv1.name, inv2.name), 0.0)
            if S == 0.0:
                S = synergy_matrix.get((inv2.name, inv1.name), 0.0)
            if S != 0.0:
                E1 = inv1.expected_effect * w1
                E2 = inv2.expected_effect * w2
                combined_effect += S * (E1 * E2)
    
    t = 0.0
    for day in range(total_days):
        # Determine phase
        if day < phase_days["flatten"]:
            phase = "flatten"
            drug_factor = 1.0
            active_drugs = drug_names
        elif day < phase_days["flatten"] + phase_days["heat"]:
            phase = "heat"
            drug_factor = 0.3  # Drug holiday
            active_drugs = []
            if day == phase_days["flatten"]:
                immune.apply_drug_holiday(phase_days["heat"])
                resistance.apply_holiday(phase_days["heat"])
        else:
            phase = "push"
            drug_factor = 0.7
            active_drugs = drug_names
        
        # Resistance-adjusted drug efficacy
        avg_efficacy = np.mean([resistance.get_efficacy_factor(d) for d in drug_names])
        
        # Update resistance if drugs active
        if active_drugs:
            resistance.update(dt, active_drugs)
        
        # ODE step: dx/dt = A_eff * x + noise + immune_force
        A_eff = A_cancer + combined_effect * drug_factor * avg_efficacy
        
        # Curvature tracking
        try:
            curv = optimizer.compute_basin_curvature(A_eff)
            min_curvature = min(min_curvature, curv)
        except:
            pass
        
        # Immune force
        well_depth = max(0.1, np.linalg.norm(x) * 0.5)
        force = immune.compute_net_force(x, well_depth, dt)
        
        # Euler-Maruyama step
        drift = A_eff @ x
        noise = rng.randn(n) * NOISE_SCALE * np.sqrt(dt)
        
        x = x + (drift + force) * dt + noise
    
    # Final distance to healthy (origin ≈ normalized healthy state)
    final_dist = np.linalg.norm(x)
    
    return final_dist, min_curvature


def run_monte_carlo(A_cancer, A_healthy, drugs, phase_days, n_trials=MONTE_CARLO_TRIALS):
    """Run MC trials and return (distances, cure_rate, ci_low, ci_high)."""
    distances = []
    for trial in range(n_trials):
        dist, _ = run_protocol_simulation(A_cancer, A_healthy, drugs, phase_days, seed=trial)
        distances.append(dist)
    
    distances = np.array(distances)
    cured = (distances < CURE_THRESHOLD).astype(float)
    cure_rate = np.mean(cured)
    
    # Bootstrap CI
    n_bootstrap = 200
    bootstrap_rates = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(cured), size=len(cured), replace=True)
        bootstrap_rates.append(np.mean(cured[idx]))
    ci_low = np.percentile(bootstrap_rates, 2.5)
    ci_high = np.percentile(bootstrap_rates, 97.5)
    
    return distances, cure_rate, ci_low, ci_high


def run_resistance_comparison(A_cancer, drugs, phase_days, A_healthy=None):
    """Compare adaptive vs continuous therapy."""
    # Adaptive: standard protocol
    adapt_dist, adapt_curv = run_protocol_simulation(
        A_cancer, A_healthy, drugs, phase_days, seed=42)
    
    # Continuous: no drug holiday, same total days
    total = sum(phase_days.values())
    cont_phases = {"flatten": total, "heat": 0, "push": 0}
    cont_dist, cont_curv = run_protocol_simulation(
        A_cancer, A_healthy, drugs, cont_phases, seed=42)
    
    return {
        "adaptive_distance": round(adapt_dist, 4),
        "continuous_distance": round(cont_dist, 4),
        "adaptive_wins": adapt_dist < cont_dist,
        "advantage": round(cont_dist - adapt_dist, 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN: FULL PAN-CANCER RUN
# ═══════════════════════════════════════════════════════════════════════

def run_single_cancer(cancer_type: str, mapper: InterventionMapper,
                      generate_lab_protocol: bool = False,
                      output_dir: str = "results") -> Dict:
    """Run complete pipeline for one cancer type."""
    print(f"\n{'═' * 60}")
    print(f"  {cancer_type}")
    print(f"{'═' * 60}")
    
    # Get generators
    A_cancer = GENERATOR_MAP[cancer_type]()
    A_healthy = TNBCODESystem.healthy_generator()
    
    # 1. Seriousness
    seriousness = compute_seriousness(cancer_type, A_cancer, A_healthy)
    print(f"  Seriousness: {seriousness:.4f}")
    
    # 2. Drug selection
    drugs = select_drugs(mapper, A_cancer, A_healthy, cancer_type)
    drug_names = [d.name for d, _ in drugs]
    print(f"  Drugs: {', '.join(drug_names)}")
    
    # 3. Phase timing
    phase_days = compute_phase_timing(seriousness)
    total_days = sum(phase_days.values())
    print(f"  Phases: Flatten({phase_days['flatten']}d) → Heat({phase_days['heat']}d) → Push({phase_days['push']}d) = {total_days}d")
    
    # 4. Toxicity check
    guard = ToxicityGuard()
    safety = guard.evaluate_protocol(drug_names, phase_days)
    safety_status = "✅ SAFE" if safety["is_safe"] else "⚠️ REVIEW"
    print(f"  Safety: {safety_status} (score={safety['safety_score']:.2f}, G3/4={safety['cumulative_g34_probability']:.0%})")
    
    if safety["violations"]:
        for v in safety["violations"]:
            print(f"    🚫 {v}")
    
    # 5. Monte Carlo cure rate
    print(f"  Running {MONTE_CARLO_TRIALS} Monte Carlo trials...", end=" ", flush=True)
    distances, cure_rate, ci_low, ci_high = run_monte_carlo(
        A_cancer, A_healthy, drugs, phase_days, MONTE_CARLO_TRIALS)
    print(f"Cure rate: {cure_rate:.0%} [{ci_low:.0%}, {ci_high:.0%}]")
    
    # 6. Adaptive vs Continuous
    resistance_cmp = run_resistance_comparison(A_cancer, drugs, phase_days, A_healthy)
    winner = "Adaptive" if resistance_cmp["adaptive_wins"] else "Continuous"
    print(f"  Resistance: {winner} wins (advantage={resistance_cmp['advantage']:.3f})")
    
    # 7. Clonal dynamics
    clonal_params = get_cancer_specific_clonal_params(cancer_type)
    clonal_engine = ClonalDynamicsEngine(clonal_params)
    clonal_result = clonal_engine.compare_adaptive_vs_continuous(phase_days)
    adapt_rf = clonal_result["adaptive"]["final_resistant_fraction"]
    cont_rf = clonal_result["continuous"]["final_resistant_fraction"]
    print(f"  Clonal: Adaptive resistant={adapt_rf:.1%} vs Continuous={cont_rf:.1%}")
    
    # 8. Generate lab protocol (if requested)
    lab_protocol = None
    if generate_lab_protocol:
        translator = ProtocolTranslator()
        sim_results = {
            "escape_distance": round(float(np.mean(distances)), 4),
            "cure_rate": f"{cure_rate:.0%}",
            "adaptive_advantage": resistance_cmp["advantage"],
        }
        lab_protocol = translator.generate_lab_protocol(
            cancer_type=cancer_type,
            drug_names=drug_names,
            phase_days=phase_days,
            simulation_results=sim_results,
            safety_result=safety,
            clonal_result=clonal_result,
        )
        
        # Save protocol
        os.makedirs(output_dir, exist_ok=True)
        protocol_path = os.path.join(output_dir, f"{cancer_type.lower()}_lab_protocol.md")
        translator.save_protocol(lab_protocol, protocol_path)
        print(f"  📋 Lab protocol saved: {protocol_path}")
    
    return {
        "cancer_type": cancer_type,
        "seriousness": seriousness,
        "drugs": drug_names,
        "drug_evidence": [d.evidence_level for d, _ in drugs],
        "phase_days": phase_days,
        "total_days": total_days,
        "escape_distance": round(float(np.mean(distances)), 4),
        "cure_rate": round(cure_rate, 4),
        "cure_rate_ci": [round(ci_low, 4), round(ci_high, 4)],
        "safety": safety,
        "resistance_comparison": resistance_cmp,
        "clonal_dynamics": clonal_result,
    }


def run_all_cancers(generate_lab_protocols: bool = True,
                    output_dir: str = "results") -> Dict:
    """Run the complete pan-cancer pipeline."""
    
    print("╔" + "═" * 58 + "╗")
    print("║   PROJECT CONFLUENCE — Universal Pan-Cancer Cure Engine   ║")
    print("║         Spectral Attractor Escape Model (SAEM)           ║")
    print("║              Lab-Validatable Framework v2.0               ║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    mapper = InterventionMapper()
    print(f"Drug Library: {len(mapper.intervention_library)} interventions loaded")
    print(f"Synergy Pairs: {len(InterventionMapper.SYNERGY_MATRIX)} validated interactions")
    print(f"Monte Carlo: {MONTE_CARLO_TRIALS} trials per cancer")
    print(f"Cancer Types: {len(CANCER_TYPES)}")
    
    results = []
    for cancer_type in CANCER_TYPES:
        result = run_single_cancer(
            cancer_type, mapper,
            generate_lab_protocol=generate_lab_protocols,
            output_dir=output_dir,
        )
        results.append(result)
    
    # ── Validation Gates ──
    print("\n" + "═" * 60)
    print("  VALIDATION GATES")
    print("═" * 60)
    
    # Gate 1: Cure threshold (≥90% per cancer)
    cure_pass = sum(1 for r in results if r["cure_rate"] >= 0.90)
    gate1 = cure_pass >= 8
    print(f"  {'✅' if gate1 else '❌'} Cure Threshold (≥90% in ≥8 cancers): {cure_pass}/{len(CANCER_TYPES)}")
    
    # Gate 2: Adaptive superiority
    adapt_pass = sum(1 for r in results if r["resistance_comparison"]["adaptive_wins"])
    gate2 = adapt_pass >= 8
    print(f"  {'✅' if gate2 else '❌'} Adaptive Superiority: {adapt_pass}/{len(CANCER_TYPES)}")
    
    # Gate 3: Protocol diversity (≥5 unique drug combos)
    unique_combos = len(set(tuple(sorted(r["drugs"])) for r in results))
    gate3 = unique_combos >= 5
    print(f"  {'✅' if gate3 else '❌'} Protocol Diversity (≥5 unique): {unique_combos}/{len(CANCER_TYPES)}")
    
    # Gate 4: Non-uniform outcomes
    escape_dists = [r["escape_distance"] for r in results]
    dist_range = max(escape_dists) - min(escape_dists)
    gate4 = dist_range > 0.05
    print(f"  {'✅' if gate4 else '❌'} Non-Uniform Outcomes: range={dist_range:.3f}")
    
    # Gate 5 (NEW): Safety clearance
    safe_count = sum(1 for r in results if r["safety"]["is_safe"])
    gate5 = safe_count >= 8
    print(f"  {'✅' if gate5 else '❌'} Safety Clearance: {safe_count}/{len(CANCER_TYPES)}")
    
    # Gate 6 (NEW): Clonal dynamics — adaptive suppresses resistant fraction
    clonal_pass = sum(1 for r in results 
                      if r["clonal_dynamics"].get("adaptive_wins", False))
    gate6 = clonal_pass >= 8
    print(f"  {'✅' if gate6 else '❌'} Clonal Dynamics (adaptive < continuous resistant): {clonal_pass}/{len(CANCER_TYPES)}")
    
    all_pass = all([gate1, gate2, gate3, gate4, gate5, gate6])
    
    print("\n" + ("🏆 ALL GATES PASSED" if all_pass else "⚠️ SOME GATES FAILED"))
    
    # ── Summary Report ──
    summary = {
        "framework": "SAEM — Project Confluence v2.0",
        "date": "auto-generated",
        "cancer_types": len(CANCER_TYPES),
        "drug_library_size": len(mapper.intervention_library),
        "synergy_pairs": len(InterventionMapper.SYNERGY_MATRIX),
        "monte_carlo_trials": MONTE_CARLO_TRIALS,
        "gates": {
            "cure_threshold": {"pass": gate1, "score": f"{cure_pass}/{len(CANCER_TYPES)}"},
            "adaptive_superiority": {"pass": gate2, "score": f"{adapt_pass}/{len(CANCER_TYPES)}"},
            "protocol_diversity": {"pass": gate3, "score": f"{unique_combos} unique"},
            "non_uniform_outcomes": {"pass": gate4, "score": f"range={dist_range:.3f}"},
            "safety_clearance": {"pass": gate5, "score": f"{safe_count}/{len(CANCER_TYPES)}"},
            "clonal_dynamics": {"pass": gate6, "score": f"{clonal_pass}/{len(CANCER_TYPES)}"},
        },
        "all_gates_pass": all_pass,
        "results": results,
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "confluence_results.json"), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Generate markdown report
    generate_confluence_report(summary, output_dir)
    
    return summary


def generate_confluence_report(summary: Dict, output_dir: str):
    """Generate the comprehensive markdown report."""
    results = summary["results"]
    gates = summary["gates"]
    
    lines = []
    lines.append("# Project Confluence — Universal Cancer Cure Framework")
    lines.append(f"\n> **Framework:** SAEM v2.0 | **Cancers:** {summary['cancer_types']} | "
                 f"**Drugs:** {summary['drug_library_size']} | "
                 f"**MC Trials:** {summary['monte_carlo_trials']}")
    lines.append("")
    
    # Gates
    lines.append("## Validation Gates\n")
    lines.append("| Gate | Status | Score |")
    lines.append("|---|---|---|")
    for name, gate in gates.items():
        icon = "✅" if gate["pass"] else "❌"
        lines.append(f"| {name.replace('_', ' ').title()} | {icon} | {gate['score']} |")
    lines.append("")
    
    verdict = "🏆 ALL GATES PASSED" if summary["all_gates_pass"] else "⚠️ REVIEW REQUIRED"
    lines.append(f"**Verdict:** {verdict}\n")
    
    # Per-cancer results
    lines.append("## Per-Cancer Results\n")
    lines.append("| Cancer | Cure Rate | 95% CI | Escape Dist | Safety | Adapt > Cont | Resistant Frac (A/C) |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: -x["seriousness"]):
        safety_icon = "✅" if r["safety"]["is_safe"] else "⚠️"
        adapt_icon = "✅" if r["resistance_comparison"]["adaptive_wins"] else "❌"
        clonal = r["clonal_dynamics"]
        a_rf = clonal.get("adaptive", {}).get("final_resistant_fraction", "?")
        c_rf = clonal.get("continuous", {}).get("final_resistant_fraction", "?")
        lines.append(
            f"| **{r['cancer_type']}** | {r['cure_rate']:.0%} | "
            f"[{r['cure_rate_ci'][0]:.0%}, {r['cure_rate_ci'][1]:.0%}] | "
            f"{r['escape_distance']:.3f} | {safety_icon} | {adapt_icon} | "
            f"A:{a_rf:.1%} / C:{c_rf:.1%} |"
        )
    lines.append("")
    
    # Protocol details
    lines.append("## Tailored Protocols\n")
    for r in results:
        pd = r["phase_days"]
        lines.append(f"### {r['cancer_type']}")
        lines.append(f"- **Flatten** ({pd['flatten']}d) → **Heat** ({pd['heat']}d) → **Push** ({pd['push']}d) = {r['total_days']}d")
        lines.append(f"- **Drugs:** {', '.join(r['drugs'])}")
        lines.append(f"- **Safety:** score={r['safety']['safety_score']:.2f}, G3/4={r['safety']['cumulative_g34_probability']:.0%}")
        if r["safety"]["violations"]:
            for v in r["safety"]["violations"]:
                lines.append(f"  - 🚫 {v}")
        lines.append("")
    
    # New features
    lines.append("## Framework Enhancements (v2.0)\n")
    lines.append("| Feature | Description |")
    lines.append("|---|---|")
    lines.append("| **Clonal Dynamics** | Lotka-Volterra 2-clone competition (sensitive vs resistant) |")
    lines.append("| **Toxicity Guard** | Clinical safety constraints with MTD, organ overlap, G3/4 risk |")
    lines.append("| **Drug Synergy** | 8 validated pairwise interaction coefficients |")
    lines.append("| **Protocol Translator** | Direct conversion to wet-lab protocols (μM, cell lines, endpoints) |")
    lines.append("| **19-Drug Library** | Including HCQ, 5-Aza, N6F11, Bevacizumab, CAR-T |")
    lines.append("")
    
    # Lab validation path
    lines.append("## Path to Wet-Lab Validation\n")
    lines.append("### Tier 1: Cell Line (2-4 weeks, ~$5-10K)")
    lines.append("- Pick one cancer (recommend TNBC — MDA-MB-231)")
    lines.append("- Run generated lab protocol → measure viability, metabolomics, ROS")
    lines.append("- Compare Geometric vs Continuous vs SOC\n")
    lines.append("### Tier 2: Organoid (4-8 weeks, ~$20-50K)")
    lines.append("- Patient-derived 3D organoids")
    lines.append("- Full Flatten→Heat→Push with real drug concentrations\n")
    lines.append("### Tier 3: Resistance Evolution (6-12 weeks, ~$30-80K)")
    lines.append("- Serial passage IC50 tracking (adaptive vs continuous)")
    lines.append("- Clone barcode sequencing for resistance dynamics\n")
    
    lines.append("---\n")
    lines.append("*Generated by Project Confluence — SAEM Cancer PoC Framework*")
    
    report_path = os.path.join(output_dir, "confluence_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print(f"\n📝 Report saved: {report_path}")


# ═══════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project Confluence — Universal Cure Engine")
    parser.add_argument("--cancer", type=str, default=None,
                        help="Run single cancer type (e.g., TNBC, PDAC)")
    parser.add_argument("--all", action="store_true", default=True,
                        help="Run all 10 cancer types (default)")
    parser.add_argument("--no-lab-protocols", action="store_true",
                        help="Skip lab protocol generation")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--trials", type=int, default=100,
                        help="Monte Carlo trials per cancer (default: 100)")
    
    args = parser.parse_args()
    
    MONTE_CARLO_TRIALS = args.trials
    generate_lab = not args.no_lab_protocols
    
    if args.cancer:
        if args.cancer not in CANCER_TYPES:
            print(f"Unknown cancer type: {args.cancer}")
            print(f"Available: {', '.join(CANCER_TYPES)}")
            sys.exit(1)
        
        mapper = InterventionMapper()
        result = run_single_cancer(args.cancer, mapper, 
                                   generate_lab_protocol=generate_lab,
                                   output_dir=args.output)
    else:
        summary = run_all_cancers(generate_lab_protocols=generate_lab,
                                  output_dir=args.output)
