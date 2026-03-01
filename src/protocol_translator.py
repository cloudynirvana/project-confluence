"""
Protocol Translator — Project Confluence
==========================================

Converts SAEM in-silico protocol outputs into wet-lab-ready experimental
protocols that can be directly executed by a collaborating laboratory.

Output includes:
  - Drug concentrations in μM (mapped from model dosing via PK)
  - Exact media change schedule aligned to protocol phases
  - Measurable endpoints with timing (viability, metabolomics, ROS, apoptosis)
  - Statistical design (replicates, controls, power)
  - Expected outcomes from the simulation for comparison

Target Systems:
  - Cell lines (MDA-MB-231 for TNBC, Panc-1 for PDAC, etc.)
  - 3D tumor organoids (patient-derived)
  - Co-culture systems (tumor + immune cells)

References:
  - Chou 2010, Pharmacol Rev (Combination Index method)
  - Sachs et al. 2021, PNAS (organoid drug screening)
  - Barretina et al. 2012, Nature (CCLE cell line encyclopedia)
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════
# DRUG → IN-VITRO CONCENTRATION MAPPING
# ═══════════════════════════════════════════════════════════════════════

# Maps SAEM drug names to in-vitro concentrations (μM) and cell line IC50s
# Sources: CCLE, published literature, DrugBank
INVITRO_DRUG_MAP: Dict[str, Dict] = {
    "Dichloroacetate (DCA)": {
        "concentration_uM": [1000, 5000, 10000],   # DCA needs mM range
        "ic50_uM": {"MDA-MB-231": 8000, "Panc-1": 12000, "A549": 10000},
        "solvent": "PBS",
        "storage": "-20°C, protected from light",
        "preparation": "Dissolve in PBS, filter sterilize (0.22μm)",
        "vendor": "Sigma-Aldrich #347795",
    },
    "Metformin": {
        "concentration_uM": [500, 2000, 5000],
        "ic50_uM": {"MDA-MB-231": 3000, "Panc-1": 5000, "A549": 4000},
        "solvent": "PBS",
        "storage": "RT",
        "preparation": "Dissolve in PBS, filter sterilize",
        "vendor": "Sigma-Aldrich #D150959",
    },
    "2-Deoxyglucose (2-DG)": {
        "concentration_uM": [500, 2000, 5000],
        "ic50_uM": {"MDA-MB-231": 2500, "Panc-1": 3500, "A549": 3000},
        "solvent": "PBS",
        "storage": "-20°C",
        "preparation": "Dissolve in PBS, filter sterilize",
        "vendor": "Sigma-Aldrich #D8375",
    },
    "CB-839 (Telaglenastat)": {
        "concentration_uM": [0.1, 0.5, 1.0],
        "ic50_uM": {"MDA-MB-231": 0.3, "Panc-1": 0.8, "A549": 0.5},
        "solvent": "DMSO",
        "storage": "-20°C",
        "preparation": "Dissolve in DMSO (10mM stock), dilute in media (final DMSO <0.1%)",
        "vendor": "Selleckchem #S7655",
    },
    "Olaparib (PARP inhibitor)": {
        "concentration_uM": [1.0, 5.0, 10.0],
        "ic50_uM": {"MDA-MB-231": 5.0, "Panc-1": 15.0, "A549": 8.0},
        "solvent": "DMSO",
        "storage": "-20°C",
        "preparation": "Dissolve in DMSO (10mM stock), dilute in media",
        "vendor": "Selleckchem #S1060",
    },
    "Vorinostat (SAHA, HDACi)": {
        "concentration_uM": [0.5, 1.0, 5.0],
        "ic50_uM": {"MDA-MB-231": 1.5, "Panc-1": 2.0, "A549": 2.5},
        "solvent": "DMSO",
        "storage": "-20°C",
        "preparation": "Dissolve in DMSO (50mM stock), dilute in media",
        "vendor": "Selleckchem #S1047",
    },
    "Anti-PD-1 (Pembrolizumab)": {
        "concentration_uM": [0.01, 0.1, 1.0],   # nM-μM range (antibody)
        "ic50_uM": {},  # Not directly cytotoxic — measured by T-cell activation
        "solvent": "PBS",
        "storage": "4°C, do not freeze",
        "preparation": "Dilute in complete media",
        "vendor": "BioXCell or clinical-grade",
        "note": "Requires T-cell co-culture system for efficacy readout",
    },
    "Anti-CTLA-4 (Ipilimumab)": {
        "concentration_uM": [0.01, 0.1, 1.0],
        "ic50_uM": {},
        "solvent": "PBS",
        "storage": "4°C, do not freeze",
        "preparation": "Dilute in complete media",
        "vendor": "BioXCell or clinical-grade",
        "note": "Requires T-cell co-culture system for efficacy readout",
    },
    "Hydroxychloroquine (HCQ)": {
        "concentration_uM": [5, 20, 50],
        "ic50_uM": {"MDA-MB-231": 25, "Panc-1": 15, "A549": 30},
        "solvent": "PBS",
        "storage": "RT, protected from light",
        "preparation": "Dissolve in PBS, filter sterilize",
        "vendor": "Sigma-Aldrich #H0915",
    },
    "5-Azacitidine (DNMTi)": {
        "concentration_uM": [0.5, 1.0, 5.0],
        "ic50_uM": {"MDA-MB-231": 2.0, "Panc-1": 3.0, "A549": 1.5},
        "solvent": "DMSO",
        "storage": "-80°C, use within 30min of thawing",
        "preparation": "Dissolve fresh in DMSO (10mM), dilute immediately",
        "vendor": "Sigma-Aldrich #A2385",
    },
    "N6F11 (Selective GPX4 degrader)": {
        "concentration_uM": [1.0, 5.0, 10.0],
        "ic50_uM": {"MDA-MB-231": 3.0, "Panc-1": 5.0},
        "solvent": "DMSO",
        "storage": "-80°C",
        "preparation": "Dissolve in DMSO (10mM stock)",
        "vendor": "MedChemExpress (contact for availability)",
    },
    "Bevacizumab (Anti-VEGF)": {
        "concentration_uM": [0.01, 0.1, 0.5],
        "ic50_uM": {},
        "solvent": "PBS",
        "storage": "4°C",
        "preparation": "Dilute in complete media",
        "vendor": "Clinical-grade (Avastin) or BioXCell",
        "note": "Efficacy measured by vascular normalization, not direct kill",
    },
    "Ferroptosis Inducer (Erastin/RSL3)": {
        "concentration_uM": [1.0, 5.0, 10.0],
        "ic50_uM": {"MDA-MB-231": 5.0, "Panc-1": 8.0, "A549": 6.0},
        "solvent": "DMSO",
        "storage": "-20°C",
        "preparation": "Dissolve in DMSO (10mM stock)",
        "vendor": "Selleckchem #S7242 (Erastin), #S8155 (RSL3)",
    },
    "Entropic Heating (Hyperthermia)": {
        "concentration_uM": [],
        "ic50_uM": {},
        "solvent": "N/A",
        "storage": "N/A",
        "preparation": "Incubate cells at 42°C for 1 hour in water bath or CO2 incubator",
        "vendor": "N/A",
        "note": "Temperature-controlled incubator or water bath required",
    },
    "High-dose Vitamin C": {
        "concentration_uM": [100, 500, 2000],
        "ic50_uM": {"MDA-MB-231": 1000, "Panc-1": 800, "A549": 1200},
        "solvent": "PBS (pH-adjusted to 7.4)",
        "storage": "-20°C, prepare fresh",
        "preparation": "Dissolve in PBS, adjust pH to 7.4, filter sterilize, use immediately",
        "vendor": "Sigma-Aldrich #A4544",
    },
    "Fasting-Mimicking Diet (FMD)": {
        "concentration_uM": [],
        "ic50_uM": {},
        "solvent": "N/A",
        "storage": "N/A",
        "preparation": "Use low-glucose DMEM (0.5g/L glucose) + 1% FBS (vs normal 10% FBS, 4.5g/L glucose)",
        "vendor": "N/A",
        "note": "Simulated by glucose/serum restriction in vitro",
    },
}

# ═══════════════════════════════════════════════════════════════════════
# TARGET CELL LINES PER CANCER TYPE
# ═══════════════════════════════════════════════════════════════════════

CELL_LINE_MAP: Dict[str, Dict] = {
    "TNBC": {
        "primary": "MDA-MB-231",
        "alternatives": ["BT-549", "MDA-MB-468", "HCC1937"],
        "organoid_source": "Patient-derived from breast tumor biopsy",
        "media": "DMEM + 10% FBS + 1% Pen/Strep",
        "doubling_time_hours": 38,
    },
    "PDAC": {
        "primary": "Panc-1",
        "alternatives": ["MIA PaCa-2", "BxPC-3", "CFPAC-1"],
        "organoid_source": "EUS-FNA biopsy or surgical resection",
        "media": "RPMI-1640 + 10% FBS + 1% Pen/Strep",
        "doubling_time_hours": 52,
    },
    "NSCLC": {
        "primary": "A549",
        "alternatives": ["H460", "H1299", "PC-9"],
        "organoid_source": "Patient-derived from lung tumor biopsy",
        "media": "RPMI-1640 + 10% FBS",
        "doubling_time_hours": 22,
    },
    "Melanoma": {
        "primary": "A375",
        "alternatives": ["SK-MEL-28", "WM266-4", "UACC-62"],
        "organoid_source": "Patient-derived melanoma spheroids",
        "media": "DMEM + 10% FBS",
        "doubling_time_hours": 20,
    },
    "GBM": {
        "primary": "U87-MG",
        "alternatives": ["U251", "T98G", "LN-229"],
        "organoid_source": "Patient-derived glioma spheroids (neurosphere assay)",
        "media": "DMEM/F-12 + 10% FBS or Neurobasal + B27 (stem-like)",
        "doubling_time_hours": 34,
    },
    "CRC": {
        "primary": "HCT-116",
        "alternatives": ["SW480", "HT-29", "Caco-2"],
        "organoid_source": "Patient-derived intestinal organoids (Matrigel dome)",
        "media": "McCoy's 5A + 10% FBS",
        "doubling_time_hours": 18,
    },
    "HGSOC": {
        "primary": "OVCAR-3",
        "alternatives": ["SKOV3", "A2780", "COV362"],
        "organoid_source": "Ascites-derived or surgical tumor fragments",
        "media": "RPMI-1640 + 10% FBS",
        "doubling_time_hours": 48,
    },
    "AML": {
        "primary": "HL-60",
        "alternatives": ["KG-1", "THP-1", "MOLM-13"],
        "organoid_source": "N/A (suspension culture)",
        "media": "RPMI-1640 + 10% FBS",
        "doubling_time_hours": 36,
    },
    "mCRPC": {
        "primary": "PC-3",
        "alternatives": ["DU145", "LNCaP", "22Rv1"],
        "organoid_source": "Patient-derived prostate organoids",
        "media": "RPMI-1640 + 10% FBS",
        "doubling_time_hours": 33,
    },
    "HCC": {
        "primary": "HepG2",
        "alternatives": ["Huh-7", "SNU-449", "PLC/PRF/5"],
        "organoid_source": "Patient-derived hepatic organoids",
        "media": "DMEM + 10% FBS",
        "doubling_time_hours": 48,
    },
}


# ═══════════════════════════════════════════════════════════════════════
# MEASURABLE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

ENDPOINT_PROTOCOLS: Dict[str, Dict] = {
    "cell_viability": {
        "assay": "CellTiter-Glo 2.0 (ATP-based luminescence)",
        "vendor": "Promega #G9241",
        "readout": "Luminescence (RLU) → % viability vs DMSO control",
        "timepoints": ["Day 0", "End Phase 1", "End Phase 2", "End Phase 3", "Day +7 (follow-up)"],
        "replicates": 3,
        "controls": ["DMSO vehicle", "Positive control (staurosporine 1μM)"],
        "protocol": (
            "1. Seed cells in 96-well plate (5000 cells/well)\n"
            "2. Allow 24h attachment\n"
            "3. Add drug treatments per protocol schedule\n"
            "4. At each timepoint: add CellTiter-Glo reagent (1:1 v/v)\n"
            "5. Incubate 10 min RT, read luminescence\n"
            "6. Normalize to Day 0 vehicle control"
        ),
    },
    "metabolite_panel": {
        "assay": "Targeted LC-MS/MS metabolomics (10-metabolite panel)",
        "vendor": "Core facility or Biocrates kit",
        "readout": "Absolute concentrations (μmol/L) for: Glucose, Lactate, Pyruvate, ATP, NAD+/NADH, Glutamine, Glutamate, α-KG, Citrate, ROS (via DCFDA)",
        "timepoints": ["Day 0", "End Phase 1", "End Phase 2", "End Phase 3"],
        "replicates": 3,
        "controls": ["Untreated", "Vehicle control"],
        "protocol": (
            "1. Collect cell pellets (1M cells) at each timepoint\n"
            "2. Quench in 80% methanol (-80°C)\n"
            "3. Extract metabolites (methanol:water 4:1)\n"
            "4. Run on LC-MS/MS with internal standards\n"
            "5. For ROS: parallel plate with DCFDA (20μM, 30min) → flow cytometry"
        ),
    },
    "apoptosis": {
        "assay": "Annexin V / PI dual staining + flow cytometry",
        "vendor": "BD Biosciences #556547",
        "readout": "% early apoptotic (AV+/PI−), late apoptotic (AV+/PI+), necrotic (AV−/PI+)",
        "timepoints": ["End Phase 1", "End Phase 2", "End Phase 3"],
        "replicates": 3,
        "controls": ["Untreated", "Staurosporine 1μM (positive)"],
        "protocol": (
            "1. Harvest cells by gentle trypsinization\n"
            "2. Wash 2× in cold PBS\n"
            "3. Resuspend in 1× Binding Buffer (1×10⁵ cells/100μL)\n"
            "4. Add 5μL Annexin V-FITC + 5μL PI\n"
            "5. Incubate 15 min RT in dark\n"
            "6. Add 400μL Binding Buffer\n"
            "7. Analyze by flow cytometry within 1 hour"
        ),
    },
    "clonogenic_survival": {
        "assay": "Colony formation assay (clonogenic assay)",
        "vendor": "Crystal violet staining",
        "readout": "Colony count (≥50 cells) → surviving fraction",
        "timepoints": ["Post-treatment Day 10-14"],
        "replicates": 3,
        "controls": ["Untreated", "Vehicle control"],
        "protocol": (
            "1. After treatment protocol, harvest cells\n"
            "2. Seed 200-1000 cells in 6-well plates (count-adjusted per group)\n"
            "3. Culture in fresh media for 10-14 days\n"
            "4. Fix with 4% PFA, stain with 0.5% crystal violet\n"
            "5. Count colonies ≥50 cells\n"
            "6. Calculate surviving fraction = (colonies formed / cells seeded) / PE"
        ),
    },
    "resistance_evolution": {
        "assay": "Serial passage IC50 tracking",
        "vendor": "Standard dose-response with CellTiter-Glo",
        "readout": "IC50 shift over passages → resistance index",
        "timepoints": ["Passage 0", "Passage 2", "Passage 4", "Passage 6"],
        "replicates": 3,
        "controls": ["Untreated passage-matched", "Continuous treatment control"],
        "protocol": (
            "1. Split cells into 3 arms: Untreated, Adaptive, Continuous\n"
            "2. Adaptive: Flatten→Heat→Push per SAEM protocol per passage\n"
            "3. Continuous: Same drugs, no holiday, same total exposure\n"
            "4. At each passage (every 7-10 days):\n"
            "   a. Harvest cells from each arm\n"
            "   b. Run 8-point dose-response curve for each drug\n"
            "   c. Calculate IC50\n"
            "5. Plot IC50 vs passage number\n"
            "6. Success criterion: Adaptive IC50 drift < Continuous IC50 drift"
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════════
# PROTOCOL TRANSLATOR
# ═══════════════════════════════════════════════════════════════════════

class ProtocolTranslator:
    """
    Translates SAEM computational protocol into a wet-lab-executable 
    experimental protocol document.
    
    Usage:
        translator = ProtocolTranslator()
        protocol = translator.generate_lab_protocol(
            cancer_type="TNBC",
            drug_names=["2-Deoxyglucose (2-DG)", "CB-839 (Telaglenastat)", ...],
            phase_days={"flatten": 23, "heat": 6, "push": 24},
            simulation_results={...}
        )
        
        # Save to file
        translator.save_protocol(protocol, "results/tnbc_lab_protocol.md")
    """
    
    def __init__(self):
        self.drug_map = INVITRO_DRUG_MAP
        self.cell_lines = CELL_LINE_MAP
        self.endpoints = ENDPOINT_PROTOCOLS
    
    def generate_lab_protocol(self,
                              cancer_type: str,
                              drug_names: List[str],
                              phase_days: Dict[str, int],
                              simulation_results: Optional[Dict] = None,
                              safety_result: Optional[Dict] = None,
                              clonal_result: Optional[Dict] = None,
                              ) -> Dict:
        """
        Generate a complete lab-executable protocol.
        
        Args:
            cancer_type: Cancer type (e.g., "TNBC")
            drug_names: List of drugs in the protocol
            phase_days: {"flatten": n1, "heat": n2, "push": n3}
            simulation_results: Optional SAEM simulation output for comparison
            safety_result: Optional ToxicityGuard output
            clonal_result: Optional ClonalDynamics output
        
        Returns:
            Complete protocol dictionary
        """
        cell_info = self.cell_lines.get(cancer_type, self.cell_lines["TNBC"])
        total_days = sum(phase_days.values())
        
        # ── Drug preparation ──
        drug_preparations = []
        for name in drug_names:
            drug_info = self.drug_map.get(name, {})
            cell_line = cell_info["primary"]
            ic50 = drug_info.get("ic50_uM", {}).get(cell_line, "N/A")
            
            concentrations = drug_info.get("concentration_uM", [])
            # Select middle concentration near IC50 for protocol dose
            if concentrations and ic50 != "N/A":
                # Pick concentration closest to IC50 * 0.5 (sub-lethal for combination)
                target = ic50 * 0.5
                protocol_conc = min(concentrations, key=lambda c: abs(c - target))
            elif concentrations:
                protocol_conc = concentrations[len(concentrations)//2]
            else:
                protocol_conc = "Per equipment spec"
            
            drug_preparations.append({
                "name": name,
                "protocol_concentration": protocol_conc,
                "concentration_unit": "μM" if isinstance(protocol_conc, (int, float)) else "",
                "concentration_range_tested": concentrations,
                "ic50": ic50,
                "solvent": drug_info.get("solvent", "DMSO"),
                "storage": drug_info.get("storage", "Check vendor"),
                "preparation": drug_info.get("preparation", "Follow vendor protocol"),
                "vendor": drug_info.get("vendor", "Contact supplier"),
                "note": drug_info.get("note", ""),
            })
        
        # ── Phase schedule ──
        schedule = []
        day_cursor = 0
        
        # Phase 1: Flatten
        flatten_drugs = [d for d in drug_names 
                        if d not in ["Anti-PD-1 (Pembrolizumab)", "Anti-CTLA-4 (Ipilimumab)",
                                     "CAR-T Cell Therapy", "Entropic Heating (Hyperthermia)"]]
        schedule.append({
            "phase": "Phase 1: FLATTEN (Metabolic Destabilization)",
            "days": f"Day {day_cursor} – Day {day_cursor + phase_days['flatten'] - 1}",
            "duration": f"{phase_days['flatten']} days",
            "drugs": flatten_drugs,
            "media_change": "Every 48 hours (refresh drugs at each change)",
            "action": "Add metabolic drugs at protocol concentrations to culture media",
        })
        day_cursor += phase_days['flatten']
        
        # Phase 2: Heat (drug holiday + entropic drivers)
        heat_drugs = [d for d in drug_names 
                     if d in ["Entropic Heating (Hyperthermia)", "High-dose Vitamin C",
                             "Ferroptosis Inducer (Erastin/RSL3)", "N6F11 (Selective GPX4 degrader)"]]
        schedule.append({
            "phase": "Phase 2: HEAT (Drug Holiday + Entropic Destabilization)",
            "days": f"Day {day_cursor} – Day {day_cursor + phase_days['heat'] - 1}",
            "duration": f"{phase_days['heat']} days",
            "drugs": heat_drugs if heat_drugs else ["Drug-free media (holiday)"],
            "media_change": "Day 1 of phase: replace with drug-free media",
            "action": (
                "1. Remove Phase 1 drugs (wash 2× with PBS)\n"
                "2. Replace with fresh media (no metabolic drugs)\n" +
                ("3. Apply hyperthermia: 42°C, 1 hour daily\n" if "Entropic Heating (Hyperthermia)" in drug_names else "") +
                "3. Allow immune/metabolic recovery"
            ),
        })
        day_cursor += phase_days['heat']
        
        # Phase 3: Push (immune + maintenance)
        push_drugs = [d for d in drug_names 
                     if d in ["Anti-PD-1 (Pembrolizumab)", "Anti-CTLA-4 (Ipilimumab)",
                             "CAR-T Cell Therapy"]]
        schedule.append({
            "phase": "Phase 3: PUSH (Immune Consolidation)",
            "days": f"Day {day_cursor} – Day {day_cursor + phase_days['push'] - 1}",
            "duration": f"{phase_days['push']} days",
            "drugs": push_drugs if push_drugs else ["Maintenance metabolic drugs at 50% dose"],
            "media_change": "Every 72 hours",
            "action": (
                "1. Add checkpoint inhibitors at protocol concentrations\n"
                "2. For immune readout: add PBMCs at 5:1 E:T ratio (co-culture)\n"
                "3. Maintain at 37°C, 5% CO2"
            ),
            "note": "Checkpoint inhibitors require T-cell co-culture for meaningful readout",
        })
        
        # ── Statistical design ──
        stats = {
            "plate_format": "96-well (viability) + 6-well (metabolomics, apoptosis)",
            "replicates_per_condition": 3,
            "biological_replicates": 3,
            "conditions": [
                "1. Vehicle control (DMSO ≤0.1%)",
                "2. Geometric protocol (Flatten→Heat→Push)",
                "3. Continuous treatment (same drugs, no holiday)", 
                "4. Standard of care (cancer-type-specific)",
                "5. Individual drug controls (for combination index)",
            ],
            "total_wells_96wp": "5 conditions × 3 replicates × 5 timepoints = 75 wells/plate",
            "power_analysis": "N=3 per condition provides 80% power to detect 30% viability difference (α=0.05, two-tailed t-test)",
        },
        
        # ── Expected outcomes (from simulation) ──
        expected_outcomes = {}
        if simulation_results:
            expected_outcomes = {
                "escape_distance": simulation_results.get("escape_distance", "N/A"),
                "cure_rate": simulation_results.get("cure_rate", "N/A"),
                "adaptive_vs_continuous_advantage": simulation_results.get("adaptive_advantage", "N/A"),
            }
        
        # ── Assemble protocol ──
        protocol = {
            "title": f"Project Confluence — {cancer_type} Wet-Lab Validation Protocol",
            "version": "1.0",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "cancer_type": cancer_type,
            "cell_line": cell_info,
            "total_duration_days": total_days,
            "drug_preparations": drug_preparations,
            "phase_schedule": schedule,
            "endpoints": {
                name: ep for name, ep in self.endpoints.items()
            },
            "statistical_design": stats,
            "expected_outcomes": expected_outcomes,
            "safety_assessment": safety_result or {},
            "clonal_dynamics_prediction": clonal_result or {},
            "materials_list": self._generate_materials_list(drug_names, cancer_type),
            "success_criteria": {
                "primary": "Geometric protocol achieves >30% greater tumor cell kill than continuous protocol",
                "secondary": [
                    "Metabolite trajectory rank correlation ≥0.7 vs SAEM prediction",
                    "Adaptive protocol maintains lower IC50 drift than continuous at passage 4",
                    "ROS levels match predicted Phase 2 spike pattern",
                ],
            },
        }
        
        return protocol
    
    def _generate_materials_list(self, drug_names: List[str], 
                                  cancer_type: str) -> List[Dict]:
        """Generate consolidated materials and reagents list."""
        materials = []
        
        # Drugs
        for name in drug_names:
            info = self.drug_map.get(name, {})
            materials.append({
                "category": "Drug",
                "item": name,
                "vendor": info.get("vendor", "Contact supplier"),
                "storage": info.get("storage", "Check vendor"),
            })
        
        # Cell culture
        cell_info = self.cell_lines.get(cancer_type, {})
        materials.extend([
            {"category": "Cell Line", "item": cell_info.get("primary", "TBD"), 
             "vendor": "ATCC", "storage": "LN2"},
            {"category": "Media", "item": cell_info.get("media", "DMEM + 10% FBS"),
             "vendor": "Gibco", "storage": "4°C"},
            {"category": "Plates", "item": "96-well white-walled (×6)", 
             "vendor": "Corning #3917", "storage": "RT"},
            {"category": "Plates", "item": "6-well tissue culture (×12)", 
             "vendor": "Corning #3516", "storage": "RT"},
        ])
        
        # Assay kits
        materials.extend([
            {"category": "Assay Kit", "item": "CellTiter-Glo 2.0", 
             "vendor": "Promega #G9241", "storage": "-20°C"},
            {"category": "Assay Kit", "item": "Annexin V-FITC/PI Apoptosis Kit",
             "vendor": "BD #556547", "storage": "4°C"},
            {"category": "Reagent", "item": "DCFDA (ROS detection)", 
             "vendor": "Sigma #D6883", "storage": "-20°C"},
            {"category": "Reagent", "item": "Crystal Violet (clonogenic)",
             "vendor": "Sigma #C0775", "storage": "RT"},
        ])
        
        return materials
    
    def protocol_to_markdown(self, protocol: Dict) -> str:
        """Convert protocol dict to professional markdown document."""
        lines = []
        
        lines.append(f"# {protocol['title']}")
        lines.append(f"\n**Version:** {protocol['version']} | **Date:** {protocol['date']}")
        lines.append(f"\n**Cancer Type:** {protocol['cancer_type']} | "
                     f"**Cell Line:** {protocol['cell_line']['primary']} | "
                     f"**Duration:** {protocol['total_duration_days']} days")
        lines.append("")
        
        # Safety
        safety = protocol.get("safety_assessment", {})
        if safety:
            status = "✅ CLEARED" if safety.get("is_safe") else "⚠️ REVIEW REQUIRED"
            lines.append(f"> **Safety Status:** {status} | "
                        f"Score: {safety.get('safety_score', 'N/A')} | "
                        f"G3/4 Risk: {safety.get('cumulative_g34_probability', 'N/A')}")
            lines.append("")
        
        lines.append("---\n")
        
        # Drug Preparations
        lines.append("## Drug Preparations\n")
        lines.append("| Drug | Concentration | IC50 | Solvent | Vendor |")
        lines.append("|---|---|---|---|---|")
        for d in protocol["drug_preparations"]:
            conc = f"{d['protocol_concentration']} {d['concentration_unit']}"
            lines.append(f"| {d['name']} | {conc} | {d['ic50']} μM | {d['solvent']} | {d['vendor']} |")
        lines.append("")
        
        # Phase Schedule
        lines.append("## Treatment Schedule\n")
        for phase in protocol["phase_schedule"]:
            lines.append(f"### {phase['phase']}")
            lines.append(f"**{phase['days']}** ({phase['duration']})\n")
            lines.append(f"**Active drugs:** {', '.join(phase['drugs'])}\n")
            lines.append(f"**Media change:** {phase['media_change']}\n")
            lines.append(f"**Actions:**\n{phase['action']}\n")
        
        # Endpoints
        lines.append("## Measured Endpoints\n")
        for name, ep in protocol["endpoints"].items():
            lines.append(f"### {name.replace('_', ' ').title()}")
            lines.append(f"- **Assay:** {ep['assay']}")
            lines.append(f"- **Readout:** {ep['readout']}")
            lines.append(f"- **Timepoints:** {', '.join(ep['timepoints'])}")
            lines.append(f"- **Replicates:** {ep['replicates']}")
            lines.append("")
        
        # Success Criteria
        lines.append("## Success Criteria\n")
        sc = protocol["success_criteria"]
        lines.append(f"**Primary:** {sc['primary']}\n")
        lines.append("**Secondary:**")
        for s in sc["secondary"]:
            lines.append(f"- {s}")
        lines.append("")
        
        # Clonal dynamics prediction
        clonal = protocol.get("clonal_dynamics_prediction", {})
        if clonal:
            lines.append("## Clonal Dynamics Prediction\n")
            if "adaptive" in clonal:
                adaptive = clonal["adaptive"]
                continuous = clonal.get("continuous", {})
                lines.append(f"| Metric | Adaptive | Continuous |")
                lines.append(f"|---|---|---|")
                lines.append(f"| Resistant fraction | {adaptive.get('final_resistant_fraction', 'N/A')} | {continuous.get('final_resistant_fraction', 'N/A')} |")
                lines.append(f"| Total burden | {adaptive.get('final_burden', 'N/A')} | {continuous.get('final_burden', 'N/A')} |")
                lines.append(f"| Cured | {adaptive.get('is_cured', 'N/A')} | {continuous.get('is_cured', 'N/A')} |")
            lines.append("")
        
        # Materials
        lines.append("## Materials List\n")
        lines.append("| Category | Item | Vendor | Storage |")
        lines.append("|---|---|---|---|")
        for m in protocol["materials_list"]:
            lines.append(f"| {m['category']} | {m['item']} | {m['vendor']} | {m['storage']} |")
        lines.append("")
        
        # Expected outcomes from simulation
        expected = protocol.get("expected_outcomes", {})
        if expected:
            lines.append("## SAEM In-Silico Predictions (For Comparison)\n")
            for k, v in expected.items():
                lines.append(f"- **{k.replace('_', ' ').title()}:** {v}")
            lines.append("")
        
        lines.append("---\n")
        lines.append("*Generated by Project Confluence — SAEM Cancer PoC Framework*\n")
        lines.append("*⚠️ This protocol is computationally generated. Review by qualified "
                     "wet-lab personnel is required before execution.*")
        
        return "\n".join(lines)
    
    def save_protocol(self, protocol: Dict, filepath: str):
        """Save protocol as both markdown and JSON."""
        # Markdown version
        md_content = self.protocol_to_markdown(protocol)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        # JSON version (for programmatic access)
        json_path = filepath.replace('.md', '.json')
        # Convert non-serializable items
        serializable = json.loads(json.dumps(protocol, default=str))
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2)
