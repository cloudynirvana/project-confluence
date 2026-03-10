"""
Bioinformatics Miner — Project Confluence (UCP Module 4)
=========================================================

Extracts omics data from public repositories and maps to ODE parameters.

Pipeline Position:
    Public APIs (cBioPortal, TCGA, GEO) → [BioinformaticsMiner] → Retrospective Cohorts

Capabilities:
    - cBioPortal REST API: mutations, CNA, mRNA expression
    - Gene → ODE parameter mapping via gene_to_parameter_map.json
    - Cohort extraction for retrospective validation (Phase 2)

Note: Requires network access and API configuration for full functionality.
      Provides mock data mode for offline testing.

References:
    Cerami et al. (2012) — The cBio Cancer Genomics Portal
    Gao et al. (2013) — Integrative Analysis of Complex Cancer Genomics
"""

import json
import warnings
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    warnings.warn("requests not installed. Run: pip install requests")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

CBIOPORTAL_BASE_URL = "https://www.cbioportal.org/api"

# Common TCGA study IDs
TCGA_STUDIES = {
    "LUAD": "luad_tcga_pan_can_atlas_2018",
    "BRCA": "brca_tcga_pan_can_atlas_2018",
    "TNBC": "brca_tcga_pan_can_atlas_2018",  # Filtered by subtype
    "PDAC": "paad_tcga_pan_can_atlas_2018",
    "GBM":  "gbm_tcga_pan_can_atlas_2018",
    "CRC":  "coadread_tcga_pan_can_atlas_2018",
    "NSCLC": "luad_tcga_pan_can_atlas_2018",
    "Melanoma": "skcm_tcga_pan_can_atlas_2018",
    "HCC":  "lihc_tcga_pan_can_atlas_2018",
    "AML":  "laml_tcga_pan_can_atlas_2018",
    "HGSOC": "ov_tcga_pan_can_atlas_2018",
}

# Key genes for ODE parameter mapping
METABOLIC_GENES = [
    "HK2", "PKM", "LDHA", "LDHB", "PDK1", "G6PD",       # Glycolysis
    "GLS", "GLUL", "SLC1A5",                                # Glutamine
    "IDH1", "IDH2", "CS", "ACO2",                           # TCA cycle
    "NDUFS1", "SDHA", "UQCRC1", "COX5A", "ATP5F1A",       # OXPHOS
    "SOD1", "SOD2", "CAT", "GPX4",                           # ROS defense
    "AMPK", "MTOR", "ULK1", "BECN1",                        # Autophagy/mTOR
]


@dataclass
class PatientSample:
    """A single patient sample from a cohort."""
    patient_id: str
    study_id: str
    cancer_type: str
    mutations: List[str] = field(default_factory=list)
    cna_events: Dict[str, int] = field(default_factory=dict)
    expression: Dict[str, float] = field(default_factory=dict)
    clinical: Dict = field(default_factory=dict)
    # Derived
    parameter_shifts: Dict[str, float] = field(default_factory=dict)


@dataclass
class RetrospectiveCohort:
    """Collection of patient samples for validation."""
    study_id: str
    cancer_type: str
    samples: List[PatientSample] = field(default_factory=list)

    @property
    def n_patients(self) -> int:
        return len(self.samples)

    def to_json(self, path: Optional[str] = None) -> str:
        """Export cohort data."""
        data = {
            "study_id": self.study_id,
            "cancer_type": self.cancer_type,
            "n_patients": self.n_patients,
            "samples": [
                {
                    "patient_id": s.patient_id,
                    "n_mutations": len(s.mutations),
                    "mutations": s.mutations[:10],
                    "parameter_shifts": s.parameter_shifts,
                    "clinical": s.clinical,
                }
                for s in self.samples
            ],
        }
        json_str = json.dumps(data, indent=2)
        if path:
            with open(path, 'w') as f:
                f.write(json_str)
        return json_str


# ═══════════════════════════════════════════════════════════════════════════
# GENE → PARAMETER MAPPER
# ═══════════════════════════════════════════════════════════════════════════

class GeneParameterMapper:
    """
    Maps gene alterations to ODE parameter modifications.

    Uses gene_to_parameter_map.json for the mapping rules.
    """

    def __init__(self, map_path: Optional[str] = None):
        if map_path and Path(map_path).exists():
            with open(map_path) as f:
                self.mapping = json.load(f)
        else:
            self.mapping = self._default_mapping()

    def _default_mapping(self) -> Dict:
        """Built-in gene → parameter mapping."""
        return {
            # Glycolysis genes
            "HK2":  {"parameter": "glucose_uptake", "mutation_effect": -0.3,
                     "amplification_effect": -0.5, "description": "Hexokinase 2"},
            "PKM":  {"parameter": "glycolysis_flux", "mutation_effect": 0.2,
                     "amplification_effect": 0.4, "description": "Pyruvate kinase M"},
            "LDHA": {"parameter": "pyruvate_to_lactate", "mutation_effect": 0.1,
                     "amplification_effect": 0.3, "description": "Lactate dehydrogenase A"},
            "PDK1": {"parameter": "pyruvate_to_atp", "mutation_effect": -0.2,
                     "amplification_effect": -0.3, "description": "PDH kinase → blocks OXPHOS"},
            # Glutamine genes
            "GLS":  {"parameter": "glutamine_utilization", "mutation_effect": -0.2,
                     "amplification_effect": -0.4, "description": "Glutaminase"},
            "SLC1A5": {"parameter": "glutaminolysis", "mutation_effect": 0.15,
                       "amplification_effect": 0.3, "description": "Glutamine transporter"},
            # ROS defense
            "SOD2": {"parameter": "ros_clearance", "mutation_effect": 0.3,
                     "deletion_effect": 0.5, "description": "Superoxide dismutase 2"},
            "GPX4": {"parameter": "ros_clearance", "mutation_effect": 0.4,
                     "deletion_effect": 0.6, "description": "Glutathione peroxidase 4"},
            # DNA repair (indirect metabolic effect)
            "BRCA1": {"parameter": "ros_atp_damage", "mutation_effect": -0.15,
                      "description": "DNA repair → metabolic stress sensitivity"},
            "TP53":  {"parameter": "atp_turnover", "mutation_effect": -0.1,
                      "description": "p53 → metabolic reprogramming"},
        }

    def map_patient(self, mutations: List[str],
                    cna: Optional[Dict[str, int]] = None) -> Dict[str, float]:
        """
        Map a patient's alterations to parameter shifts.

        Parameters
        ----------
        mutations : list of str
            Gene names with mutations.
        cna : dict, optional
            Gene → CNA level (-2=HomDel, -1=HetDel, 1=Gain, 2=Amp).

        Returns
        -------
        shifts : dict
            Parameter name → cumulative shift value.
        """
        shifts = {}
        cna = cna or {}

        for gene in mutations:
            if gene in self.mapping:
                m = self.mapping[gene]
                param = m["parameter"]
                shift = m.get("mutation_effect", 0.0)
                shifts[param] = shifts.get(param, 0.0) + shift

        for gene, level in cna.items():
            if gene in self.mapping:
                m = self.mapping[gene]
                param = m["parameter"]
                if level >= 2:
                    shift = m.get("amplification_effect", 0.0)
                elif level <= -2:
                    shift = m.get("deletion_effect", 0.0)
                else:
                    shift = 0.0
                shifts[param] = shifts.get(param, 0.0) + shift

        return shifts


# ═══════════════════════════════════════════════════════════════════════════
# BIOINFORMATICS MINER
# ═══════════════════════════════════════════════════════════════════════════

class BioinformaticsMiner:
    """
    UCP Module 4: Bioinformatics Miner.

    Extracts omics data from public databases and creates retrospective
    cohorts for validation.

    Usage:
        miner = BioinformaticsMiner()
        cohort = miner.extract_cohort("LUAD", max_patients=50)
        cohort.to_json("tcga_luad_cohort.json")
    """

    def __init__(self, base_url: str = CBIOPORTAL_BASE_URL,
                 gene_map_path: Optional[str] = None,
                 use_cache: bool = True):
        self.base_url = base_url
        self.mapper = GeneParameterMapper(gene_map_path)
        self.use_cache = use_cache
        self._cache: Dict = {}

    def _api_get(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make GET request to cBioPortal API."""
        if not HAS_REQUESTS:
            warnings.warn("requests library not available. Using mock data.")
            return None

        url = f"{self.base_url}/{endpoint}"
        cache_key = f"{url}:{json.dumps(params or {}, sort_keys=True)}"

        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            resp = requests.get(url, params=params, timeout=30,
                                headers={"Accept": "application/json"})
            resp.raise_for_status()
            data = resp.json()
            if self.use_cache:
                self._cache[cache_key] = data
            return data
        except Exception as e:
            warnings.warn(f"cBioPortal API error: {e}")
            return None

    def list_studies(self) -> List[Dict]:
        """List available studies from cBioPortal."""
        data = self._api_get("studies")
        if data is None:
            return [{"studyId": v, "cancer_type": k}
                    for k, v in TCGA_STUDIES.items()]
        return data

    def get_mutations(self, study_id: str,
                      gene_list: Optional[List[str]] = None) -> List[Dict]:
        """Fetch mutation data for a study."""
        genes = gene_list or METABOLIC_GENES

        data = self._api_get(
            f"molecular-profiles/{study_id}_mutations/mutations",
            params={"projection": "SUMMARY"}
        )

        if data is None:
            return self._mock_mutations(study_id, genes)

        # Filter to our genes of interest
        return [m for m in data
                if m.get("gene", {}).get("hugoGeneSymbol", "") in genes]

    def get_clinical_data(self, study_id: str) -> List[Dict]:
        """Fetch clinical data (OS, stage, etc.) for a study."""
        data = self._api_get(
            f"studies/{study_id}/clinical-data",
            params={"clinicalDataType": "PATIENT", "projection": "SUMMARY"}
        )

        if data is None:
            return self._mock_clinical(study_id)

        return data

    def extract_cohort(self, cancer_type: str,
                       max_patients: int = 100,
                       gene_list: Optional[List[str]] = None) -> RetrospectiveCohort:
        """
        Extract a retrospective cohort from TCGA via cBioPortal.

        Parameters
        ----------
        cancer_type : str
            Cancer type key (e.g., "LUAD", "BRCA").
        max_patients : int
            Maximum patients to include.
        gene_list : list of str, optional
            Genes to query. Defaults to METABOLIC_GENES.

        Returns
        -------
        cohort : RetrospectiveCohort
        """
        study_id = TCGA_STUDIES.get(cancer_type)
        if not study_id:
            warnings.warn(f"No TCGA study mapped for {cancer_type}. "
                          f"Available: {list(TCGA_STUDIES.keys())}")
            return RetrospectiveCohort(study_id="unknown", cancer_type=cancer_type)

        # Fetch mutations
        mutations_data = self.get_mutations(study_id, gene_list)

        # Group mutations by patient
        patient_mutations: Dict[str, List[str]] = {}
        for m in mutations_data:
            pid = m.get("patientId", m.get("sampleId", "unknown"))
            gene = m.get("gene", {}).get("hugoGeneSymbol",
                         m.get("hugoGeneSymbol", "?"))
            patient_mutations.setdefault(pid, []).append(gene)

        # Fetch clinical data
        clinical_data = self.get_clinical_data(study_id)
        clinical_map: Dict[str, Dict] = {}
        for c in clinical_data:
            pid = c.get("patientId", "unknown")
            attr = c.get("clinicalAttributeId", "")
            val = c.get("value", "")
            clinical_map.setdefault(pid, {})[attr] = val

        # Build samples
        samples = []
        for pid in list(patient_mutations.keys())[:max_patients]:
            muts = patient_mutations[pid]
            param_shifts = self.mapper.map_patient(muts)

            sample = PatientSample(
                patient_id=pid,
                study_id=study_id,
                cancer_type=cancer_type,
                mutations=muts,
                parameter_shifts=param_shifts,
                clinical=clinical_map.get(pid, {}),
            )
            samples.append(sample)

        return RetrospectiveCohort(
            study_id=study_id,
            cancer_type=cancer_type,
            samples=samples,
        )

    # ── MOCK DATA (for offline testing) ──

    def _mock_mutations(self, study_id: str,
                        genes: List[str]) -> List[Dict]:
        """Generate synthetic mutation data for testing."""
        rng = np.random.RandomState(hash(study_id) % 2**31)
        mocks = []
        for i in range(50):
            n_muts = rng.poisson(3)
            for _ in range(n_muts):
                gene = rng.choice(genes)
                mocks.append({
                    "patientId": f"TCGA-{i:04d}",
                    "sampleId": f"TCGA-{i:04d}-01",
                    "hugoGeneSymbol": gene,
                    "gene": {"hugoGeneSymbol": gene},
                    "mutationType": rng.choice(["Missense", "Nonsense",
                                                 "Frame_Shift"]),
                })
        return mocks

    def _mock_clinical(self, study_id: str) -> List[Dict]:
        """Generate synthetic clinical data for testing."""
        rng = np.random.RandomState(hash(study_id) % 2**31)
        mocks = []
        for i in range(50):
            pid = f"TCGA-{i:04d}"
            os_months = rng.exponential(36)
            os_status = "DECEASED" if rng.random() < 0.4 else "LIVING"
            mocks.extend([
                {"patientId": pid, "clinicalAttributeId": "OS_MONTHS",
                 "value": f"{os_months:.1f}"},
                {"patientId": pid, "clinicalAttributeId": "OS_STATUS",
                 "value": f"0:{os_status}" if os_status == "LIVING" else f"1:{os_status}"},
                {"patientId": pid, "clinicalAttributeId": "CANCER_TYPE_DETAILED",
                 "value": study_id.split("_")[0].upper()},
            ])
        return mocks
