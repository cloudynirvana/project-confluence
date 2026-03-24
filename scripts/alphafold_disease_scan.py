#!/usr/bin/env python3
"""
AlphaFold Disease Scanner — Project Confluence
=================================================

End-to-end pipeline:
  1. Scan disease gene panels through AlphaFold structural database
  2. Compute structural vulnerability scores
  3. Score drug-target binding compatibility
  4. Run ODE simulations with structurally-informed parameters
  5. Output drug-target match rankings and Φ restoration predictions

Usage:
    python scripts/alphafold_disease_scan.py --disease TNBC
    python scripts/alphafold_disease_scan.py --disease Alzheimers --mock
    python scripts/alphafold_disease_scan.py --all --mock --top-drugs 5
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from models.alphafold_client import (
    AlphaFoldClient,
    DISEASE_PANELS,
    create_mock_structure,
)
from models.structure_bridge import StructureBridge, DRUG_TARGET_MAP
from models.structural_docking import DrugTargetMatcher, DRUG_PROFILES, compute_docking_score
from models.ode_system import (
    ComplexAttractorODE,
    ExtendedParams,
    TNBCParams,
    AlzheimersParams,
    ParkinsonsParams,
    DiabetesParams,
    GlioblastomaParams,
    ALSParams,
    LupusParams,
)
from models.complexity_profiler import ComplexityProfiler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Disease → ODE Params mapping
DISEASE_PARAMS = {
    "TNBC": TNBCParams,
    "Alzheimers": AlzheimersParams,
    "Parkinsons": ParkinsonsParams,
    "Diabetes": DiabetesParams,
    "GBM": GlioblastomaParams,
    "ALS": ALSParams,
    "Lupus": LupusParams,
}


def run_disease_scan(
    disease: str,
    use_mock: bool = True,
    top_n_drugs: int = 5,
    output_dir: str = "results/alphafold_scan",
) -> dict:
    """
    Run full AlphaFold-enhanced disease analysis.

    Returns dict with structural vulnerability profiles, drug rankings,
    and Φ-distance predictions.
    """
    print(f"\n{'='*70}")
    print(f"  AlphaFold × Confluence — Disease Scan: {disease}")
    print(f"{'='*70}\n")

    # ── Step 1: Fetch/mock structures ──────────────────────────────
    print("▸ Step 1: Fetching protein structures...")
    panel = DISEASE_PANELS.get(disease, {})

    if use_mock:
        structures = {}
        for gene, uniprot_id in panel.items():
            structures[gene] = create_mock_structure(
                uniprot_id=uniprot_id,
                gene_name=gene,
                n_residues=400,
            )
            print(f"    {gene} ({uniprot_id}): {structures[gene].sequence_length} residues "
                  f"[mock], pLDDT={structures[gene].mean_plddt:.1f}, "
                  f"pockets={len(structures[gene].pockets)}")
    else:
        client = AlphaFoldClient()
        structures = client.fetch_panel(disease)

    # ── Step 2: Structural vulnerability analysis ─────────────────
    print("\n▸ Step 2: Computing structural vulnerabilities...")
    bridge = StructureBridge()
    profile = bridge.profile_disease(
        disease=disease,
        structures=structures,
        use_mock=False,  # We already have structures
    )

    print(f"\n  Aggregate Vulnerability: {profile.aggregate_vulnerability:.3f}")
    print(f"  Top Drug Targets: {', '.join(profile.top_drug_targets)}")

    print("\n  Per-Gene Structural Analysis:")
    print(f"  {'Gene':<12} {'Stability':<10} {'Pocket':<10} {'ODE Mult':<10} {'Tier':<12}")
    print(f"  {'-'*54}")
    for gene, mod in profile.gene_modifiers.items():
        print(f"  {gene:<12} {mod.stability_score:<10.3f} {mod.pocket_accessibility:<10.3f} "
              f"{mod.ode_parameter_multiplier:<10.3f} {mod.confidence_tier:<12}")

    # ── Step 3: Drug-target docking ───────────────────────────────
    print("\n▸ Step 3: Scoring drug-target binding compatibility...")
    matcher = DrugTargetMatcher()
    docking_results = []

    for drug_name in DRUG_PROFILES:
        for gene, structure in structures.items():
            if structure.pockets:
                results = matcher.dock(drug_name, structure)
                if results:
                    best = results[0]
                    docking_results.append({
                        "drug": drug_name,
                        "target": gene,
                        "score": best.overall_score,
                        "efficacy_mult": best.efficacy_multiplier,
                        "pocket_idx": best.pocket_index,
                        "notes": best.notes,
                    })

    # Sort by docking score
    docking_results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n  Top {top_n_drugs} Drug-Target Matches:")
    print(f"  {'Drug':<35} {'Target':<10} {'Score':<8} {'Eff.Mult':<10} {'Notes'}")
    print(f"  {'-'*80}")
    for r in docking_results[:top_n_drugs]:
        print(f"  {r['drug']:<35} {r['target']:<10} {r['score']:<8.3f} "
              f"{r['efficacy_mult']:<10.3f} {r['notes']}")

    # ── Step 4: ODE simulation with structural modifiers ──────────
    print("\n▸ Step 4: Running ODE simulations with structural modifiers...")

    params_class = DISEASE_PARAMS.get(disease, ExtendedParams)
    profiler = ComplexityProfiler()

    # Baseline simulation (no structural modification)
    baseline_params = params_class()
    baseline_ode = ComplexAttractorODE(params=baseline_params)
    baseline_result = baseline_ode.solve(t_span=(0, 100), dt_eval=1.0)
    baseline_phi = profiler.profile(baseline_result['z'], dt=1.0)

    # Healthy reference
    healthy_ode = ComplexAttractorODE(params=ExtendedParams())
    healthy_result = healthy_ode.solve(t_span=(0, 100), dt_eval=1.0)
    healthy_phi = profiler.profile(healthy_result['z'], dt=1.0)

    baseline_dist = np.linalg.norm(
        np.array(baseline_phi.phi_vector) - np.array(healthy_phi.phi_vector)
    )

    print(f"\n  Baseline Φ-distance from healthy: {baseline_dist:.4f}")
    print(f"  Baseline Φ-vector: {[round(v, 4) for v in baseline_phi.phi_vector]}")

    # Simulate with structurally-modified parameters
    # Apply the top vulnerability gene's modifier to amplify disease params
    structural_results = {}
    if profile.gene_modifiers:
        # Get the highest-impact gene
        top_gene = max(
            profile.gene_modifiers.items(),
            key=lambda x: x[1].ode_parameter_multiplier
        )
        top_gene_name, top_modifier = top_gene

        # Create modified params by scaling the key metabolic parameters
        modified_params = params_class()
        multiplier = top_modifier.ode_parameter_multiplier

        # Scale glucose-related params by structural impact
        modified_params.glucose_uptake *= multiplier
        if hasattr(modified_params, 'glycolysis_flux'):
            modified_params.glycolysis_flux *= (0.5 + 0.5 * multiplier)

        modified_ode = ComplexAttractorODE(params=modified_params)
        modified_result = modified_ode.solve(t_span=(0, 100), dt_eval=1.0)
        modified_phi = profiler.profile(modified_result['z'], dt=1.0)

        modified_dist = np.linalg.norm(
            np.array(modified_phi.phi_vector) - np.array(healthy_phi.phi_vector)
        )

        structural_results = {
            "top_gene": top_gene_name,
            "ode_multiplier": multiplier,
            "baseline_phi_distance": round(float(baseline_dist), 4),
            "modified_phi_distance": round(float(modified_dist), 4),
            "structural_impact_delta": round(float(modified_dist - baseline_dist), 4),
        }

        print(f"\n  Structural Modification ({top_gene_name}, mult={multiplier:.2f}):")
        print(f"    Modified Φ-distance: {modified_dist:.4f}")
        print(f"    Δ from baseline:     {modified_dist - baseline_dist:+.4f}")

    # ── Step 5: Summary ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY — {disease}")
    print(f"{'='*70}")
    print(f"  Genes analyzed:              {len(profile.gene_modifiers)}")
    print(f"  Aggregate vulnerability:     {profile.aggregate_vulnerability:.3f}")
    print(f"  Top drug targets:            {', '.join(profile.top_drug_targets[:3])}")
    print(f"  Baseline Φ-distance:         {baseline_dist:.4f}")
    if structural_results:
        print(f"  Structural impact (top gene): {structural_results.get('structural_impact_delta', 0):+.4f}")
    print(f"  Drug-target matches scored:  {len(docking_results)}")
    print()

    # ── Save results ──────────────────────────────────────────────
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "disease": disease,
        "timestamp": datetime.now().isoformat(),
        "mode": "mock" if use_mock else "live",
        "genes_analyzed": len(profile.gene_modifiers),
        "aggregate_vulnerability": round(profile.aggregate_vulnerability, 4),
        "top_drug_targets": profile.top_drug_targets,
        "gene_profiles": {
            g: m.to_dict() for g, m in profile.gene_modifiers.items()
        },
        "top_drug_matches": [r for r in docking_results[:top_n_drugs]],
        "phi_analysis": {
            "healthy_phi": [round(v, 4) for v in healthy_phi.phi_vector],
            "baseline_phi": [round(v, 4) for v in baseline_phi.phi_vector],
            "baseline_phi_distance": round(float(baseline_dist), 4),
        },
        "structural_impact": structural_results,
    }

    result_file = output_path / f"{disease.lower()}_alphafold_scan.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {result_file}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="AlphaFold × Confluence Disease Scanner"
    )
    parser.add_argument(
        "--disease", type=str, default=None,
        help=f"Disease to scan. Available: {', '.join(DISEASE_PANELS.keys())}"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Scan all diseases in the panel"
    )
    parser.add_argument(
        "--mock", action="store_true", default=True,
        help="Use mock structures (no API calls, default)"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Use live AlphaFold API (requires internet)"
    )
    parser.add_argument(
        "--top-drugs", type=int, default=5,
        help="Number of top drug matches to display (default: 5)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/alphafold_scan",
        help="Output directory for results"
    )
    args = parser.parse_args()

    use_mock = not args.live

    if args.all:
        diseases = list(DISEASE_PANELS.keys())
    elif args.disease:
        diseases = [args.disease]
    else:
        # Default: scan diseases that have ODE params
        diseases = list(DISEASE_PARAMS.keys())

    all_results = {}
    for disease in diseases:
        try:
            results = run_disease_scan(
                disease=disease,
                use_mock=use_mock,
                top_n_drugs=args.top_drugs,
                output_dir=args.output_dir,
            )
            all_results[disease] = results
        except Exception as e:
            logger.error(f"Failed to scan {disease}: {e}")
            import traceback
            traceback.print_exc()

    # Cross-disease comparison
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  CROSS-DISEASE COMPARISON")
        print(f"{'='*70}")
        print(f"  {'Disease':<15} {'Vulnerability':<15} {'Φ-distance':<12} {'Top Target'}")
        print(f"  {'-'*55}")

        sorted_diseases = sorted(
            all_results.items(),
            key=lambda x: x[1].get('aggregate_vulnerability', 0),
            reverse=True,
        )
        for name, res in sorted_diseases:
            vuln = res.get('aggregate_vulnerability', 0)
            phi_d = res.get('phi_analysis', {}).get('baseline_phi_distance', 0)
            top_t = res.get('top_drug_targets', ['—'])[0]
            print(f"  {name:<15} {vuln:<15.3f} {phi_d:<12.4f} {top_t}")

        print()


if __name__ == "__main__":
    main()
