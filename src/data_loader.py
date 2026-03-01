"""
Metabolomics Data Loader — Project Confluence
===============================================

Loads and normalizes cancer metabolomics data from public databases,
mapping real metabolite measurements into SAEM's 10-metabolite space.

Supported sources:
  - DepMap/CCLE metabolomics (CSV from depmap.org)
  - Kaggle lung cancer metabolomics
  - Custom CSV with metabolite columns

Usage:
    loader = CCLELoader()
    profiles = loader.load("data/raw/ccle_metabolomics.csv")
    mapper = MetaboliteMapper()
    saem_profiles = mapper.map_to_saem(profiles, cancer_type="TNBC")
"""

import os
import csv
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# SAEM's canonical 10 metabolites (same order as tnbc_ode.py)
SAEM_METABOLITES = [
    "Glucose", "Lactate", "Pyruvate", "ATP", "NADH",
    "Glutamine", "Glutamate", "aKG", "Citrate", "ROS",
]

# Cancer type → representative cell lines (for filtering CCLE data)
CANCER_CELL_LINES: Dict[str, List[str]] = {
    "TNBC":     ["MDA-MB-231", "MDA-MB-468", "HCC1937", "BT-549", "SUM149"],
    "PDAC":     ["PANC-1", "MIA PaCa-2", "BxPC-3", "AsPC-1", "Capan-1"],
    "NSCLC":    ["A549", "H1299", "H460", "H1975", "PC-9"],
    "Melanoma": ["A375", "SK-MEL-28", "WM266-4", "COLO 829", "A2058"],
    "GBM":      ["U87", "U251", "T98G", "LN-229", "A172"],
    "CRC":      ["HCT116", "SW480", "HT-29", "DLD-1", "SW620"],
    "HGSOC":    ["SKOV3", "OVCAR3", "A2780", "CAOV3", "ES-2"],
    "mCRPC":    ["PC3", "DU145", "LNCaP", "22Rv1", "VCaP"],
    "AML":      ["HL-60", "MOLM-13", "OCI-AML3", "THP-1", "KG-1"],
    "HCC":      ["HepG2", "Huh7", "SNU-449", "SK-HEP-1", "PLC/PRF/5"],
}


@dataclass
class MappingResult:
    """Result of mapping real metabolites to SAEM space."""
    saem_profile: np.ndarray          # (10,) normalized profile
    confidence: Dict[str, str]        # metabolite → 'direct' | 'proxy' | 'missing'
    source_columns: Dict[str, str]    # metabolite → original column name used
    n_direct: int = 0
    n_proxy: int = 0
    n_missing: int = 0


class MetaboliteMapper:
    """
    Map real metabolite names/IDs to SAEM's 10-metabolite space.
    
    Handles:
      - Fuzzy name matching (e.g., "L-Lactic acid" → "Lactate")
      - HMDB ID matching
      - Proxy computation for ATP, NADH, ROS
    """
    
    # Canonical mappings: SAEM name → list of alternative names in databases
    SYNONYMS: Dict[str, List[str]] = {
        "Glucose": [
            "glucose", "d-glucose", "dextrose", "blood sugar",
            "HMDB0000122", "alpha-d-glucose", "beta-d-glucose",
        ],
        "Lactate": [
            "lactate", "l-lactic acid", "lactic acid", "l-lactate",
            "HMDB0000190", "2-hydroxypropanoic acid",
        ],
        "Pyruvate": [
            "pyruvate", "pyruvic acid", "2-oxopropanoic acid",
            "HMDB0000243",
        ],
        "ATP": [
            "atp", "adenosine triphosphate", "adenosine 5'-triphosphate",
            "HMDB0000538",
        ],
        "NADH": [
            "nadh", "nad+", "nicotinamide adenine dinucleotide",
            "HMDB0001487", "HMDB0000902",
        ],
        "Glutamine": [
            "glutamine", "l-glutamine", "gln",
            "HMDB0000641",
        ],
        "Glutamate": [
            "glutamate", "l-glutamic acid", "glutamic acid", "glu",
            "HMDB0000148",
        ],
        "aKG": [
            "alpha-ketoglutarate", "alpha-ketoglutaric acid",
            "2-oxoglutarate", "2-oxoglutaric acid", "a-kg", "akg",
            "HMDB0000208",
        ],
        "Citrate": [
            "citrate", "citric acid", "2-hydroxypropane-1,2,3-tricarboxylic acid",
            "HMDB0000094",
        ],
        "ROS": [
            # ROS is not a single metabolite — use proxies
            "ros", "reactive oxygen species",
            # GSH/GSSG ratio as proxy (lower = more ROS)
            "glutathione", "gsh", "gssg", "oxidized glutathione",
        ],
    }
    
    # Proxy computation strategies
    ATP_PROXIES = ["atp", "adp", "amp", "adenylate energy charge"]
    NADH_PROXIES = ["nadh", "nad+", "nad", "nicotinamide"]
    ROS_PROXIES = ["gsh", "gssg", "glutathione", "mda", "malondialdehyde",
                   "8-ohdg", "dcfda", "lipid peroxidation"]
    
    def __init__(self):
        """Build reverse lookup: lowercase synonym → SAEM name."""
        self._reverse_map: Dict[str, str] = {}
        for saem_name, synonyms in self.SYNONYMS.items():
            for syn in synonyms:
                self._reverse_map[syn.lower()] = saem_name
    
    def find_column_match(self, column_name: str) -> Optional[str]:
        """
        Try to match a raw column name to a SAEM metabolite.
        
        Uses exact match first, then substring matching.
        Returns SAEM metabolite name or None.
        """
        col_lower = column_name.lower().strip()
        
        # Exact match
        if col_lower in self._reverse_map:
            return self._reverse_map[col_lower]
        
        # Substring match (e.g., "L-Glutamine (HMDB0000641)" → "Glutamine")
        for synonym, saem_name in self._reverse_map.items():
            if synonym in col_lower or col_lower in synonym:
                return saem_name
        
        return None
    
    def map_columns(self, column_names: List[str]) -> Dict[str, str]:
        """
        Map a list of raw column names to SAEM metabolites.
        
        Returns: {SAEM_name: raw_column_name} for matched metabolites.
        """
        mapping: Dict[str, str] = {}
        used_columns: set = set()
        
        for col in column_names:
            saem_name = self.find_column_match(col)
            if saem_name and saem_name not in mapping and col not in used_columns:
                mapping[saem_name] = col
                used_columns.add(col)
        
        return mapping
    
    def map_row_to_saem(
        self,
        row_data: Dict[str, float],
        column_mapping: Dict[str, str],
    ) -> MappingResult:
        """
        Convert a single row of raw metabolomics data to a 10-element SAEM vector.
        
        Args:
            row_data:       {column_name: value} for one sample
            column_mapping: {SAEM_name: raw_column_name} from map_columns()
        
        Returns:
            MappingResult with normalized 10-element profile
        """
        profile = np.zeros(10)
        confidence: Dict[str, str] = {}
        source_cols: Dict[str, str] = {}
        n_direct = n_proxy = n_missing = 0
        
        for i, met_name in enumerate(SAEM_METABOLITES):
            if met_name in column_mapping:
                raw_col = column_mapping[met_name]
                value = row_data.get(raw_col, None)
                if value is not None and not np.isnan(value):
                    profile[i] = float(value)
                    confidence[met_name] = "direct"
                    source_cols[met_name] = raw_col
                    n_direct += 1
                else:
                    confidence[met_name] = "missing"
                    n_missing += 1
            elif met_name == "ROS":
                # Try GSH/GSSG proxy
                gsh_val = self._find_proxy_value(row_data, ["gsh", "glutathione"])
                gssg_val = self._find_proxy_value(row_data, ["gssg", "oxidized glutathione"])
                if gsh_val is not None and gssg_val is not None and gssg_val > 0:
                    # Lower GSH/GSSG ratio = more ROS → invert for SAEM
                    ratio = gsh_val / gssg_val
                    profile[i] = 1.0 / (1.0 + ratio)  # Scale to 0-1
                    confidence[met_name] = "proxy"
                    source_cols[met_name] = "GSH/GSSG ratio"
                    n_proxy += 1
                else:
                    confidence[met_name] = "missing"
                    n_missing += 1
            elif met_name == "ATP":
                # Try ATP/ADP proxy
                atp_val = self._find_proxy_value(row_data, ["atp"])
                adp_val = self._find_proxy_value(row_data, ["adp"])
                if atp_val is not None:
                    profile[i] = atp_val
                    confidence[met_name] = "proxy" if adp_val is None else "direct"
                    source_cols[met_name] = "ATP (or ATP/ADP)"
                    n_proxy += 1
                else:
                    confidence[met_name] = "missing"
                    n_missing += 1
            elif met_name == "NADH":
                nadh_val = self._find_proxy_value(row_data, ["nadh"])
                if nadh_val is not None:
                    profile[i] = nadh_val
                    confidence[met_name] = "direct"
                    source_cols[met_name] = "NADH"
                    n_direct += 1
                else:
                    nad_val = self._find_proxy_value(row_data, ["nad+", "nad"])
                    if nad_val is not None:
                        profile[i] = 1.0 / (1.0 + nad_val)  # Proxy from NAD+
                        confidence[met_name] = "proxy"
                        source_cols[met_name] = "NAD+ (inverted proxy)"
                        n_proxy += 1
                    else:
                        confidence[met_name] = "missing"
                        n_missing += 1
            else:
                confidence[met_name] = "missing"
                n_missing += 1
        
        # Normalize to relative abundances (z-score-like, centered on healthy baseline)
        nonzero = profile[profile != 0]
        if len(nonzero) > 0:
            mean_val = np.mean(np.abs(nonzero))
            if mean_val > 0:
                profile = profile / mean_val  # Scale so mean absolute value ≈ 1
        
        return MappingResult(
            saem_profile=profile,
            confidence=confidence,
            source_columns=source_cols,
            n_direct=n_direct,
            n_proxy=n_proxy,
            n_missing=n_missing,
        )
    
    def _find_proxy_value(
        self, row_data: Dict[str, float], keywords: List[str]
    ) -> Optional[float]:
        """Search row_data keys for a proxy metabolite by keyword matching."""
        for col_name, value in row_data.items():
            col_lower = col_name.lower()
            for kw in keywords:
                if kw in col_lower:
                    if value is not None and not np.isnan(value):
                        return float(value)
        return None


class CCLELoader:
    """
    Load DepMap/CCLE metabolomics data.
    
    Expected CSV format: rows = cell lines, columns = metabolites.
    First column is cell line name/ID.
    """
    
    def __init__(self):
        self.mapper = MetaboliteMapper()
    
    def load(
        self,
        filepath: str,
        cancer_types: Optional[List[str]] = None,
    ) -> Dict[str, List[MappingResult]]:
        """
        Load CCLE data and map to SAEM space.
        
        Args:
            filepath:     Path to CCLE metabolomics CSV
            cancer_types: Which cancer types to extract (default: all 10)
        
        Returns:
            {cancer_type: [MappingResult, ...]} for each matched cell line
        """
        if cancer_types is None:
            cancer_types = list(CANCER_CELL_LINES.keys())
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"CCLE data not found at {filepath}. "
                f"Download from https://depmap.org/portal/ → Downloads → Metabolomics"
            )
        
        # Read CSV
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            
            # Map columns to SAEM metabolites
            column_mapping = self.mapper.map_columns(columns)
            print(f"[CCLELoader] Mapped {len(column_mapping)}/10 SAEM metabolites")
            for saem, raw in column_mapping.items():
                print(f"  {saem:12s} ← {raw}")
            
            # Build reverse lookup: cell line name → cancer type
            cell_to_cancer: Dict[str, str] = {}
            for ctype in cancer_types:
                for cell_line in CANCER_CELL_LINES.get(ctype, []):
                    cell_to_cancer[cell_line.lower()] = ctype
            
            # Read rows and filter by cell line
            results: Dict[str, List[MappingResult]] = {ct: [] for ct in cancer_types}
            matched = 0
            
            for row in reader:
                # Try to identify cell line from first column
                cell_name = list(row.values())[0] if row else ""
                cell_lower = cell_name.lower().strip()
                
                # Check against known cell lines
                matched_type = None
                for known_cell, ctype in cell_to_cancer.items():
                    if known_cell in cell_lower or cell_lower in known_cell:
                        matched_type = ctype
                        break
                
                if matched_type:
                    # Convert values to float
                    row_data: Dict[str, float] = {}
                    for k, v in row.items():
                        try:
                            row_data[k] = float(v)
                        except (ValueError, TypeError):
                            row_data[k] = float('nan')
                    
                    mapping = self.mapper.map_row_to_saem(row_data, column_mapping)
                    results[matched_type].append(mapping)
                    matched += 1
            
            print(f"[CCLELoader] Matched {matched} cell lines across {len(cancer_types)} cancer types")
        
        return results
    
    def get_mean_profile(
        self, results: Dict[str, List[MappingResult]]
    ) -> Dict[str, np.ndarray]:
        """
        Compute mean SAEM profile per cancer type.
        
        Returns: {cancer_type: mean_10d_profile}
        """
        profiles: Dict[str, np.ndarray] = {}
        for ctype, mappings in results.items():
            if mappings:
                all_profiles = np.array([m.saem_profile for m in mappings])
                profiles[ctype] = np.mean(all_profiles, axis=0)
            else:
                profiles[ctype] = np.zeros(10)
        return profiles
    
    def print_coverage_report(self, results: Dict[str, List[MappingResult]]) -> None:
        """Print a summary of data coverage."""
        print("\n═══════════════════════════════════════")
        print("CCLE → SAEM Mapping Coverage Report")
        print("═══════════════════════════════════════")
        
        for ctype, mappings in results.items():
            n = len(mappings)
            if n == 0:
                print(f"  {ctype:10s}: NO DATA (0 cell lines matched)")
                continue
            
            # Average confidence across mappings
            direct_pcts = [m.n_direct / 10 * 100 for m in mappings]
            proxy_pcts = [m.n_proxy / 10 * 100 for m in mappings]
            missing_pcts = [m.n_missing / 10 * 100 for m in mappings]
            
            print(f"  {ctype:10s}: {n} cell lines | "
                  f"direct={np.mean(direct_pcts):.0f}% | "
                  f"proxy={np.mean(proxy_pcts):.0f}% | "
                  f"missing={np.mean(missing_pcts):.0f}%")
