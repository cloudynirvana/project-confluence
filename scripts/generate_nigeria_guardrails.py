"""
Generate Nigeria Clinical Guardrails — Project Confluence
==========================================================

One-time script that parses the Nigeria Clinical Guidelines Dataset
(NSTG 2022, 270 conditions) and auto-generates a Nigeria-specific
guardrails JSON for use by the Adaptive Therapy Controller.

Usage:
    python scripts/generate_nigeria_guardrails.py

Output:
    validation/nigeria_clinical_guardrails_auto.json

Data Source:
    chisomrutherford/nigeria-clinical-guidelines-dataset (HuggingFace)
    License: CC-BY-4.0 (Creative Commons Attribution 4.0 International)
    Attribution: Federal Ministry of Health, Nigeria (2022)
                 Dataset curated by Chisom Rutherford
"""

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from agents.nigeria_guideline_retriever import NigeriaGuidelineRetriever
except ImportError:
    print("Error: Cannot import NigeriaGuidelineRetriever.")
    print("Make sure you're running from the project root.")
    sys.exit(1)


def extract_all_drugs(retriever: NigeriaGuidelineRetriever) -> dict:
    """Extract all drugs mentioned across all conditions."""
    drug_registry = defaultdict(lambda: {
        "conditions": [],
        "dosages": [],
        "routes": set(),
        "adverse_reactions": [],
    })

    for cond_name, condition in retriever.conditions.items():
        protocols = condition.get("treatment_protocols", [])
        for tp in protocols:
            if not isinstance(tp, dict):
                continue
            drugs = tp.get("drug", tp.get("drugs", []))
            if not isinstance(drugs, list):
                continue
            for drug in drugs:
                if isinstance(drug, dict):
                    name = drug.get("name", drug.get("drug", ""))
                    if not name:
                        continue
                    name_key = name.strip().upper()
                    drug_registry[name_key]["conditions"].append(cond_name)
                    dosage = drug.get("dosage", drug.get("dose", ""))
                    if dosage:
                        drug_registry[name_key]["dosages"].append(dosage)
                    route = drug.get("route", "")
                    if route:
                        drug_registry[name_key]["routes"].add(route)
                adverse = tp.get("adverse_reactions_and_cautions", [])
                if isinstance(adverse, list):
                    for a in adverse:
                        if isinstance(a, str):
                            drug_registry[name_key]["adverse_reactions"].append(a)

    # Convert sets to lists for JSON serialization
    for key in drug_registry:
        drug_registry[key]["routes"] = list(drug_registry[key]["routes"])
        drug_registry[key]["conditions"] = list(set(drug_registry[key]["conditions"]))
        drug_registry[key]["adverse_reactions"] = list(set(
            drug_registry[key]["adverse_reactions"]
        ))

    return dict(drug_registry)


def extract_oncology_conditions(retriever: NigeriaGuidelineRetriever) -> dict:
    """Extract oncology-specific conditions and their protocols."""
    return retriever.extract_oncology_constraints()


def build_guardrails(retriever: NigeriaGuidelineRetriever) -> dict:
    """Build the complete Nigeria guardrails JSON."""
    print("Extracting drug registry...")
    drug_registry = extract_all_drugs(retriever)
    print(f"  Found {len(drug_registry)} unique drugs")

    print("Extracting oncology protocols...")
    oncology = extract_oncology_conditions(retriever)
    print(f"  Found {len(oncology)} oncology-related conditions")

    guardrails = {
        "meta": {
            "version": "1.0-auto",
            "framework": "Project Confluence — Nigeria Extension (Auto-generated)",
            "standard": "Nigeria Standard Treatment Guidelines (NSTG) 2022",
            "source": "Federal Ministry of Health, Nigeria",
            "license": "CC-BY-4.0",
            "attribution": (
                "Dataset curated by Chisom Rutherford. "
                "Source: chisomrutherford/nigeria-clinical-guidelines-dataset (HuggingFace)"
            ),
            "n_conditions": len(retriever.conditions),
            "n_drugs_extracted": len(drug_registry),
            "n_oncology_conditions": len(oncology),
        },
        "drug_registry": drug_registry,
        "oncology_protocols": oncology,
        "all_conditions": list(retriever.conditions.keys()),
    }

    return guardrails


def main():
    print("=" * 60)
    print("Generate Nigeria Clinical Guardrails")
    print("=" * 60)
    print()

    # Initialize retriever (will download dataset or use mock)
    print("Loading NSTG 2022 dataset...")
    retriever = NigeriaGuidelineRetriever()
    print(f"  {retriever}")
    print()

    # Build guardrails
    guardrails = build_guardrails(retriever)

    # Write output
    output_path = PROJECT_ROOT / "validation" / "nigeria_clinical_guardrails_auto.json"
    with open(output_path, "w") as f:
        json.dump(guardrails, f, indent=2, default=str)

    print()
    print(f"✅ Guardrails written to: {output_path}")
    print(f"   Conditions: {guardrails['meta']['n_conditions']}")
    print(f"   Drugs: {guardrails['meta']['n_drugs_extracted']}")
    print(f"   Oncology: {guardrails['meta']['n_oncology_conditions']}")


if __name__ == "__main__":
    main()
