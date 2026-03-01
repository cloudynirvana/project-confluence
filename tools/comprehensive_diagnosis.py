"""
Comprehensive Diagnosis Optimizer
=================================

A full diagnosis-conditioned pipeline that models all requested stages explicitly:
1. Six cancer generators + healthy reference loading
2. Seriousness scoring
3. Geometric flattening (ΔA) optimization
4. Intervention mapping
5. Lymphocyte force estimation
6. Kramers-style escape proxy
7. Tailored 3-phase protocol (Flatten -> Heat -> Push) per cancer type.
"""

import sys
import os
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from tnbc_ode import TNBCODESystem
from geometric_optimization import GeometricOptimizer
from intervention import InterventionMapper
from immune_dynamics import LymphocyteForceField, ImmuneParams
from coherence import CoherenceAnalyzer

# CALIBRATED PARAMETERS
CALIBRATED = {
    'base_force': 0.375,
    'exhaustion_rate': 0.200,
    'treg_load': 0.500,
    'noise_scale': 0.1875,
}

@dataclass
class DiagnosisProfile:
    patient_id: str
    cancer_type: str
    basal_generator: np.ndarray
    healthy_reference: np.ndarray
    age: int
    prior_treatments: List[str]

@dataclass
class OptimizationResult:
    cancer_type: str
    seriousness_score: float
    depth_ratio: float
    flattening_effort: float
    immune_push_force: float
    effective_barrier: float
    escape_rate: float
    protocol_phases: List[dict]

def build_comprehensive_diagnosis():
    n = 10
    mapper = InterventionMapper()
    lib = {i.name: i for i in mapper.intervention_library}
    optimizer = GeometricOptimizer(n)
    coherence_analyzer = CoherenceAnalyzer()
    
    generators = TNBCODESystem.pan_cancer_generators()
    healthy_ref = TNBCODESystem.healthy_generator()
    mu_healthy = optimizer.compute_basin_curvature(healthy_ref)
    
    results = []
    
    for name, A_cancer in generators.items():
        profile = DiagnosisProfile(
            patient_id=f"PATIENT_{name.upper()}_001",
            cancer_type=name,
            basal_generator=A_cancer,
            healthy_reference=healthy_ref,
            age=55,
            prior_treatments=[]
        )
        
        # 1. Coherence & Seriousness Scoring
        metrics = coherence_analyzer.analyze(A_cancer, healthy_ref)
        coherence_score = metrics.get('overall_score', 0.5)
        # Lower coherence means higher seriousness
        coherence_deficit = 1.0 - coherence_score
        
        mu_cancer = optimizer.compute_basin_curvature(A_cancer)
        depth_ratio = mu_cancer / max(mu_healthy, 1e-6)
        
        # Combine curvature and coherence loss for a holistic "seriousness" score
        seriousness_score = mu_cancer * (1.0 + coherence_deficit * 2.0)
        
        # 2. Geometric Flattening (ΔA) Optimization
        delta_A_needed = healthy_ref - A_cancer
        flattening_effort = float(np.linalg.norm(delta_A_needed))
        
        # Find exactly the right interventions for this specific cancer
        matched_interventions = mapper.map_correction_to_interventions(delta_A_needed, max_interventions=2)
        
        # 3. Generating Tailored Protocol
        # We start with the core Flatten -> Heat -> Push
        phase1_drugs = [interv.name for interv, weight in matched_interventions]
        # Always add DCA and Metformin if not there as foundational flattening
        if "Dichloroacetate (DCA)" not in phase1_drugs:
            phase1_drugs.append("Dichloroacetate (DCA)")
        if "Metformin" not in phase1_drugs:
            phase1_drugs.append("Metformin")
            
        protocol = [
            {
                "phase": "Flatten",
                "days": "0-25",
                "drugs": phase1_drugs,
                "rationale": "Metabolic suppression to reduce curvature"
            },
            {
                "phase": "Heat",
                "days": "20-25",
                "drugs": ["Entropic Heating (Hyperthermia)"],
                "rationale": "Entropic noise injection"
            },
            {
                "phase": "Push",
                "days": "25-60",
                "drugs": ["Anti-PD-1 (Pembrolizumab)"] + phase1_drugs,
                "rationale": "Immune vectoring and maintenance"
            }
        ]
        
        # 4. Lymphocyte Force & Escape Proxy Simulation (Simplified endpoint metrics)
        val, vec = np.linalg.eig(A_cancer)
        idx = np.argsort(val.real)
        x0 = np.real(vec[:, idx[0]]) * 1.5 # Start displaced
        
        immune = LymphocyteForceField(n, ImmuneParams(
            base_force=CALIBRATED['base_force'],
            exhaustion_rate=CALIBRATED['exhaustion_rate'],
            treg_load=CALIBRATED['treg_load'],
            pd1_blockade=0.8 # Apply PD-1 effect as part of Push
        ))
        
        # We approximate the "flattened" A_eff by applying the primary metabolic drugs
        A_eff = A_cancer.copy()
        for d in phase1_drugs[:2]: # simplified
            if d in lib:
                A_eff += lib[d].expected_effect * 0.5 
        
        effective_barrier = optimizer.compute_basin_curvature(A_eff)
        # Force calculation
        dt = 0.1
        f = immune.compute_net_force(x0, effective_barrier, dt)
        push_force_mag = float(np.linalg.norm(f))
        
        escape_rate = optimizer.compute_kramers_escape_rate(A_eff, CALIBRATED['noise_scale'] * 1.5, push_force_mag)
        
        res = OptimizationResult(
            cancer_type=name,
            seriousness_score=float(seriousness_score),
            depth_ratio=float(depth_ratio),
            flattening_effort=float(flattening_effort),
            immune_push_force=float(push_force_mag),
            effective_barrier=float(effective_barrier),
            escape_rate=float(escape_rate),
            protocol_phases=protocol
        )
        results.append(res)
        
    return results

def generate_report(results: List[OptimizationResult]):
    os.makedirs('results', exist_ok=True)
    report_path = 'results/comprehensive_diagnosis_report.md'
    
    # Sort by seriousness
    sorted_results = sorted(results, key=lambda x: x.seriousness_score, reverse=True)
    
    lines = [
        "# Comprehensive Diagnosis & Optimization Report",
        "",
        "## Overall Seriousness Ranking",
        "",
        "| Rank | Cancer Type | Seriousness Score | Depth Ratio | Flattening Effort | Effective Barrier | Escape Rate |",
        "|------|-------------|-------------------|-------------|-------------------|-------------------|-------------|"
    ]
    
    for i, res in enumerate(sorted_results):
        lines.append(f"| {i+1} | **{res.cancer_type}** | {res.seriousness_score:.3f} | {res.depth_ratio:.2f}x | {res.flattening_effort:.3f} | {res.effective_barrier:.3f} | {res.escape_rate:.4e} |")
        
    lines.append("")
    lines.append("## Tailored Protocols per Diagnosis")
    
    for res in sorted_results:
        lines.append(f"### {res.cancer_type}")
        lines.append(f"- **Immune Push Force:** {res.immune_push_force:.3f}")
        for p in res.protocol_phases:
            drugs_str = ", ".join(p['drugs'])
            lines.append(f"- **Phase {p['phase']} (Days {p['days']}):** {drugs_str}")
            lines.append(f"  - *Rationale:* {p['rationale']}")
        lines.append("")
        
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
        
    print(f"✅ Generated {report_path}")

if __name__ == "__main__":
    results = build_comprehensive_diagnosis()
    generate_report(results)

