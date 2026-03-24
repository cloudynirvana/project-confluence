"""
Structure-to-ODE Bridge — Project Confluence
===============================================

Maps AlphaFold protein structural features to ODE parameter modulations.
This is the core integration layer between structural biology and the
15D dynamical systems framework.

Three mapping mechanisms:
  1. pLDDT-based stability scoring — structural confidence → mutation impact
  2. Mutation destabilization — missense mutations in ordered vs disordered regions
  3. Drug-target structural affinity → intervention efficacy scaling

References:
    Jumper et al. (2021) - AlphaFold
    Akdel et al. (2022) - A structural biology community assessment of AlphaFold
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import numpy as np

from .alphafold_client import (
    AlphaFoldClient,
    StructureData,
    BindingPocket,
    DISEASE_PANELS,
    GENE_KEY_RESIDUES,
    create_mock_structure,
)

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class StructuralModifiers:
    """
    Structural modifiers for a gene mutation, derived from AlphaFold.

    These modifiers scale the gene_to_parameter_map effects based on
    the structural context of the mutation.
    """
    gene: str
    uniprot_id: str
    mutation_position: int

    # Core scores
    stability_score: float        # 0-1: how structurally impactful (1 = very)
    pocket_accessibility: float   # 0-1: how druggable the region is
    ode_parameter_multiplier: float  # Scales gene_to_parameter_map effect

    # Detail
    local_plddt: float            # pLDDT around the mutation site
    is_in_pocket: bool            # Whether mutation is in a binding pocket
    is_in_active_site: bool       # Whether mutation hits a known active site residue
    nearby_pocket_druggability: float  # Druggability of nearest pocket

    # Derived
    confidence_tier: str = ""     # "very_high", "confident", "low", "disordered"

    def __post_init__(self):
        if not self.confidence_tier:
            if self.local_plddt >= 90:
                self.confidence_tier = "very_high"
            elif self.local_plddt >= 70:
                self.confidence_tier = "confident"
            elif self.local_plddt >= 50:
                self.confidence_tier = "low"
            else:
                self.confidence_tier = "disordered"

    def to_dict(self) -> Dict:
        return {
            "gene": self.gene,
            "uniprot_id": self.uniprot_id,
            "mutation_position": self.mutation_position,
            "stability_score": round(self.stability_score, 3),
            "pocket_accessibility": round(self.pocket_accessibility, 3),
            "ode_parameter_multiplier": round(self.ode_parameter_multiplier, 3),
            "local_plddt": round(self.local_plddt, 1),
            "is_in_pocket": self.is_in_pocket,
            "is_in_active_site": self.is_in_active_site,
            "nearby_pocket_druggability": round(self.nearby_pocket_druggability, 3),
            "confidence_tier": self.confidence_tier,
        }


@dataclass
class DrugTargetAffinity:
    """
    Structural affinity score for a drug-target pair.

    Used to modulate the efficacy of drugs in intervention.py based on
    the structural quality of their binding site.
    """
    drug_name: str
    target_gene: str
    target_uniprot_id: str

    # Scores
    pocket_score: float          # 0-1: pocket quality for this drug
    structural_confidence: float # 0-1: how reliable the pocket prediction is
    efficacy_multiplier: float   # Multiplier for intervention.py drug effects

    # Detail
    best_pocket_volume: float    # Volume of best matching pocket (ų)
    best_pocket_depth: float     # Depth of best pocket (Å)
    n_contact_residues: int      # Residues in drug contact zone

    def to_dict(self) -> Dict:
        return {
            "drug_name": self.drug_name,
            "target_gene": self.target_gene,
            "pocket_score": round(self.pocket_score, 3),
            "structural_confidence": round(self.structural_confidence, 3),
            "efficacy_multiplier": round(self.efficacy_multiplier, 3),
            "best_pocket_volume": round(self.best_pocket_volume, 1),
            "best_pocket_depth": round(self.best_pocket_depth, 1),
            "n_contact_residues": self.n_contact_residues,
        }


@dataclass
class DiseaseStructuralProfile:
    """
    Comprehensive structural profile for a disease.

    Aggregates per-gene structural analysis into a disease-level
    vulnerability map with drug target rankings.
    """
    disease: str
    gene_modifiers: Dict[str, StructuralModifiers]
    drug_affinities: List[DrugTargetAffinity]
    aggregate_vulnerability: float  # 0-1: overall structural vulnerability
    top_drug_targets: List[str]     # Ranked by structural druggability

    def to_dict(self) -> Dict:
        return {
            "disease": self.disease,
            "aggregate_vulnerability": round(self.aggregate_vulnerability, 3),
            "top_drug_targets": self.top_drug_targets,
            "n_genes_analyzed": len(self.gene_modifiers),
            "n_drug_affinities": len(self.drug_affinities),
            "gene_details": {
                g: m.to_dict() for g, m in self.gene_modifiers.items()
            },
            "drug_details": [a.to_dict() for a in self.drug_affinities],
        }


# ══════════════════════════════════════════════════════════════════════
# DRUG → TARGET GENE MAPPING
# ══════════════════════════════════════════════════════════════════════

# Maps intervention.py drug names to their primary gene targets
DRUG_TARGET_MAP: Dict[str, List[str]] = {
    "Dichloroacetate (DCA)": ["PDK1", "PDK2"],
    "Metformin": ["PRKAB1"],  # AMPK beta subunit
    "2-Deoxyglucose (2-DG)": ["HK2"],
    "CB-839 (Telaglenastat)": ["GLS"],
    "Olaparib (PARP inhibitor)": ["PARP1"],
    "Vorinostat (SAHA, HDACi)": ["HDAC1", "HDAC2", "HDAC3"],
    "5-Azacitidine (DNMTi)": ["DNMT1", "DNMT3A"],
    "Hydroxychloroquine (HCQ)": ["ATG5"],  # Autophagy target
    "Anti-PD-1 (Pembrolizumab)": ["PDCD1"],  # PD-1
    "Anti-CTLA-4 (Ipilimumab)": ["CTLA4"],
    "Bevacizumab (Anti-VEGF)": ["VEGFA"],
    "Ferroptosis Inducer (Erastin/RSL3)": ["GPX4", "SLC7A11"],
}

# UniProt IDs for drug target genes
DRUG_TARGET_UNIPROT: Dict[str, str] = {
    "PDK1": "Q15118", "PDK2": "Q15119",
    "PRKAB1": "Q9Y478",
    "HK2": "P52789",
    "GLS": "O94925",
    "PARP1": "P09874",
    "HDAC1": "Q13547", "HDAC2": "Q92769", "HDAC3": "O15379",
    "DNMT1": "P26358", "DNMT3A": "Q9Y6K1",
    "ATG5": "Q9H1Y0",
    "PDCD1": "Q15116",
    "CTLA4": "P16410",
    "VEGFA": "P15692",
    "GPX4": "P36969",
    "SLC7A11": "Q9UPY5",
}

# Approximate molecular weights for drug size heuristics (Da)
DRUG_MOLECULAR_WEIGHTS: Dict[str, float] = {
    "Dichloroacetate (DCA)": 128.9,
    "Metformin": 129.2,
    "2-Deoxyglucose (2-DG)": 164.2,
    "CB-839 (Telaglenastat)": 444.5,
    "Olaparib (PARP inhibitor)": 434.5,
    "Vorinostat (SAHA, HDACi)": 264.3,
    "5-Azacitidine (DNMTi)": 244.2,
    "Hydroxychloroquine (HCQ)": 335.9,
    "Ferroptosis Inducer (Erastin/RSL3)": 500.0,
}


# ══════════════════════════════════════════════════════════════════════
# CORE BRIDGE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

class StructureBridge:
    """
    Maps AlphaFold structural features to Project Confluence ODE parameters.

    The bridge computes three types of modifiers:
      1. Mutation stability impact → scales gene_to_parameter_map effects
      2. Pocket accessibility → drug target druggability
      3. Aggregate disease vulnerability → structural risk profile

    Usage:
        bridge = StructureBridge()
        modifiers = bridge.compute_mutation_impact("TP53", 248, "P04637", structure)
        print(modifiers.ode_parameter_multiplier)
    """

    def __init__(self, gene_param_map_path: Optional[str] = None):
        """
        Initialize bridge with gene-to-parameter map.

        Args:
            gene_param_map_path: Path to gene_to_parameter_map.json.
                                 If None, uses default location.
        """
        if gene_param_map_path is None:
            gene_param_map_path = str(
                Path(__file__).parent.parent / "validation" / "gene_to_parameter_map.json"
            )
        self.gene_param_map = self._load_gene_param_map(gene_param_map_path)

    def _load_gene_param_map(self, path: str) -> Dict:
        """Load gene_to_parameter_map.json."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return data.get('mappings', data)
        except FileNotFoundError:
            logger.warning(f"gene_to_parameter_map.json not found at {path}")
            return {}

    def compute_mutation_impact(
        self,
        gene: str,
        mutation_position: int,
        uniprot_id: str,
        structure: StructureData,
    ) -> StructuralModifiers:
        """
        Compute the structural impact of a mutation at a given position.

        The key insight: a mutation in a well-folded region (pLDDT > 90)
        is structurally more impactful than one in a disordered region
        (pLDDT < 50), because disrupting ordered structure has larger
        functional consequences.

        However, active site mutations are always high-impact regardless
        of pLDDT.

        Args:
            gene: Gene symbol (e.g. "TP53")
            mutation_position: 1-based residue position
            uniprot_id: UniProt accession
            structure: Parsed AlphaFold structure

        Returns:
            StructuralModifiers with stability scores and ODE multiplier
        """
        # Get local pLDDT
        local_plddt = structure.get_local_plddt(mutation_position, window=5)
        point_plddt = structure.get_plddt_at(mutation_position)

        # Check if mutation is in an active site
        key_residues = GENE_KEY_RESIDUES.get(gene, [])
        is_active_site = mutation_position in key_residues

        # Check if mutation is in a binding pocket
        is_in_pocket = False
        nearest_pocket_drug = 0.0
        for pocket in structure.pockets:
            if mutation_position in pocket.residue_indices:
                is_in_pocket = True
                nearest_pocket_drug = max(nearest_pocket_drug, pocket.druggability_score)
            else:
                # Check proximity to pocket
                pocket_residue_coords = [
                    structure.residues[i].coords
                    for i in pocket.residue_indices
                    if i < len(structure.residues)
                ]
                if pocket_residue_coords:
                    mut_coord = None
                    for r in structure.residues:
                        if r.index == mutation_position:
                            mut_coord = r.coords
                            break
                    if mut_coord is not None:
                        min_dist = min(
                            np.linalg.norm(mut_coord - pc)
                            for pc in pocket_residue_coords
                        )
                        if min_dist < 8.0:  # Within 8Å of pocket
                            nearest_pocket_drug = max(
                                nearest_pocket_drug,
                                pocket.druggability_score * (1 - min_dist / 16.0)
                            )

        # Compute stability score
        stability_score = self._compute_stability_score(
            local_plddt, point_plddt, is_active_site, is_in_pocket
        )

        # Compute pocket accessibility
        pocket_accessibility = self._compute_pocket_accessibility(
            structure, mutation_position
        )

        # Compute ODE parameter multiplier
        ode_multiplier = self._compute_ode_multiplier(
            stability_score, is_active_site, local_plddt
        )

        return StructuralModifiers(
            gene=gene,
            uniprot_id=uniprot_id,
            mutation_position=mutation_position,
            stability_score=stability_score,
            pocket_accessibility=pocket_accessibility,
            ode_parameter_multiplier=ode_multiplier,
            local_plddt=local_plddt,
            is_in_pocket=is_in_pocket,
            is_in_active_site=is_active_site,
            nearby_pocket_druggability=nearest_pocket_drug,
        )

    def _compute_stability_score(
        self,
        local_plddt: float,
        point_plddt: float,
        is_active_site: bool,
        is_in_pocket: bool,
    ) -> float:
        """
        Compute how structurally disruptive a mutation is.

        Higher pLDDT = more ordered = mutation is more disruptive.
        Active site mutations are always scored high.
        Pocket mutations get a bonus (functional impact).
        """
        # Base score from pLDDT
        # Well-folded regions: mutations are more disruptive
        # pLDDT > 90: score ~0.9 (very disruptive)
        # pLDDT 70-90: score ~0.6-0.8
        # pLDDT 50-70: score ~0.3-0.5
        # pLDDT < 50: score ~0.1-0.2 (disordered, less impact)
        base_score = point_plddt / 100.0

        # Nonlinear scaling: emphasize high-confidence disruptions
        score = base_score ** 0.7  # Gentle boost for mid-range

        # Active site bonus
        if is_active_site:
            score = max(score, 0.85)
            score = min(1.0, score + 0.1)

        # Pocket bonus
        if is_in_pocket:
            score = min(1.0, score + 0.05)

        return float(np.clip(score, 0.0, 1.0))

    def _compute_pocket_accessibility(
        self,
        structure: StructureData,
        mutation_position: int,
    ) -> float:
        """
        Score how accessible the mutation site is to drugs.

        Based on proximity to detected binding pockets and
        the druggability of those pockets.
        """
        if not structure.pockets:
            return 0.1  # No pockets found → low accessibility

        mut_coord = None
        for r in structure.residues:
            if r.index == mutation_position:
                mut_coord = r.coords
                break

        if mut_coord is None:
            return 0.1

        best_accessibility = 0.0
        for pocket in structure.pockets:
            # Distance from mutation to pocket center
            dist = float(np.linalg.norm(mut_coord - pocket.center))

            # Closer to pocket = more accessible
            if dist < 5.0:
                proximity = 1.0
            elif dist < 15.0:
                proximity = 1.0 - (dist - 5.0) / 10.0
            else:
                proximity = 0.0

            accessibility = proximity * pocket.druggability_score
            best_accessibility = max(best_accessibility, accessibility)

        return float(np.clip(best_accessibility, 0.0, 1.0))

    def _compute_ode_multiplier(
        self,
        stability_score: float,
        is_active_site: bool,
        local_plddt: float,
    ) -> float:
        """
        Compute the ODE parameter effect multiplier.

        This scales the mutation_effect/amplification_effect in
        gene_to_parameter_map.json based on structural context.

        Range: 0.5 (weak structural impact) to 2.0 (devastating structural impact)
        Baseline: 1.0 (use gene_to_parameter_map values as-is)
        """
        # Base multiplier from stability
        multiplier = 0.5 + 1.5 * stability_score

        # Active site mutations get amplified
        if is_active_site:
            multiplier *= 1.3

        # Very high confidence disruptions are amplified
        if local_plddt > 90:
            multiplier *= 1.1

        # Cap at reasonable range
        return float(np.clip(multiplier, 0.3, 2.5))

    def compute_drug_target_affinity(
        self,
        drug_name: str,
        target_structure: StructureData,
    ) -> DrugTargetAffinity:
        """
        Score the structural affinity between a drug and its target.

        Uses pocket geometry and confidence to estimate how well
        a drug can bind to the predicted structure.

        Args:
            drug_name: Drug name (from intervention.py)
            target_structure: AlphaFold structure of the target

        Returns:
            DrugTargetAffinity with pocket score and efficacy multiplier
        """
        # Find best pocket for this drug
        drug_mw = DRUG_MOLECULAR_WEIGHTS.get(drug_name, 300.0)

        best_pocket = None
        best_score = 0.0

        for pocket in target_structure.pockets:
            # Score pocket-drug compatibility
            # Larger drugs need larger pockets
            volume_ratio = pocket.volume / max(drug_mw * 1.5, 100)
            vol_fit = 1.0 - abs(np.log(max(volume_ratio, 0.1)))
            vol_fit = max(0.0, min(1.0, vol_fit))

            # Depth matters for small molecules
            depth_score = min(1.0, pocket.depth / 10.0)

            # Confidence in pocket prediction
            conf_score = pocket.avg_plddt / 100.0

            score = (0.35 * vol_fit +
                     0.25 * depth_score +
                     0.25 * conf_score +
                     0.15 * pocket.druggability_score)

            if score > best_score:
                best_score = score
                best_pocket = pocket

        if best_pocket is None:
            return DrugTargetAffinity(
                drug_name=drug_name,
                target_gene=target_structure.gene_name,
                target_uniprot_id=target_structure.uniprot_id,
                pocket_score=0.1,
                structural_confidence=target_structure.mean_plddt / 100.0,
                efficacy_multiplier=0.5,
                best_pocket_volume=0.0,
                best_pocket_depth=0.0,
                n_contact_residues=0,
            )

        # Efficacy multiplier: good pocket = better drug delivery
        efficacy_mult = 0.5 + best_score  # Range: 0.5 to 1.5

        return DrugTargetAffinity(
            drug_name=drug_name,
            target_gene=target_structure.gene_name,
            target_uniprot_id=target_structure.uniprot_id,
            pocket_score=round(best_score, 3),
            structural_confidence=round(target_structure.mean_plddt / 100.0, 3),
            efficacy_multiplier=round(efficacy_mult, 3),
            best_pocket_volume=best_pocket.volume,
            best_pocket_depth=best_pocket.depth,
            n_contact_residues=len(best_pocket.residue_indices),
        )

    def profile_disease(
        self,
        disease: str,
        structures: Optional[Dict[str, StructureData]] = None,
        use_mock: bool = False,
    ) -> DiseaseStructuralProfile:
        """
        Generate a comprehensive structural profile for a disease.

        Analyzes all genes in the disease panel, computes mutation
        impacts at key residues, and scores drug-target affinities.

        Args:
            disease: Disease name (key in DISEASE_PANELS)
            structures: Pre-fetched structures (None = fetch/mock)
            use_mock: Use mock structures instead of API calls

        Returns:
            DiseaseStructuralProfile with rankings and scores
        """
        if disease not in DISEASE_PANELS:
            raise ValueError(f"Unknown disease: {disease}")

        panel = DISEASE_PANELS[disease]

        # Get structures
        if structures is None:
            if use_mock:
                structures = {}
                for gene, uniprot_id in panel.items():
                    n_res = {
                        "BRCA1": 1863, "TP53": 393, "EGFR": 1210,
                        "IDH1": 414, "PTEN": 403, "SOD1": 154,
                        "SNCA": 140, "APP": 770, "INS": 110,
                    }.get(gene, 400)
                    structures[gene] = create_mock_structure(
                        uniprot_id=uniprot_id,
                        gene_name=gene,
                        n_residues=n_res,
                    )
            else:
                client = AlphaFoldClient()
                structures = client.fetch_panel(disease)

        # Compute per-gene mutation impacts
        gene_modifiers = {}
        for gene, structure in structures.items():
            key_residues = GENE_KEY_RESIDUES.get(gene, [])
            if not key_residues and structure.residues:
                # Use middle residue as representative
                mid = structure.sequence_length // 2
                key_residues = [mid]

            # Use first key residue as representative mutation site
            mut_pos = key_residues[0] if key_residues else 1
            uniprot_id = panel.get(gene, "")

            modifiers = self.compute_mutation_impact(
                gene=gene,
                mutation_position=mut_pos,
                uniprot_id=uniprot_id,
                structure=structure,
            )
            gene_modifiers[gene] = modifiers

        # Compute drug-target affinities
        drug_affinities = []
        for drug_name, target_genes in DRUG_TARGET_MAP.items():
            for target_gene in target_genes:
                if target_gene in structures:
                    affinity = self.compute_drug_target_affinity(
                        drug_name, structures[target_gene]
                    )
                    drug_affinities.append(affinity)

        # Also check disease panel genes as potential drug targets
        for gene, structure in structures.items():
            for drug_name in DRUG_TARGET_MAP:
                targets = DRUG_TARGET_MAP[drug_name]
                if gene not in targets and gene in panel:
                    # Check if any pocket could bind this drug
                    if structure.pockets:
                        affinity = self.compute_drug_target_affinity(
                            drug_name, structure
                        )
                        if affinity.pocket_score > 0.3:
                            drug_affinities.append(affinity)

        # Sort drug affinities by efficacy multiplier
        drug_affinities.sort(key=lambda a: a.efficacy_multiplier, reverse=True)

        # Aggregate vulnerability score
        if gene_modifiers:
            vulnerabilities = [m.stability_score for m in gene_modifiers.values()]
            aggregate = float(np.mean(vulnerabilities))
        else:
            aggregate = 0.0

        # Top drug targets (by druggability)
        top_targets = sorted(
            gene_modifiers.keys(),
            key=lambda g: gene_modifiers[g].pocket_accessibility,
            reverse=True,
        )

        return DiseaseStructuralProfile(
            disease=disease,
            gene_modifiers=gene_modifiers,
            drug_affinities=drug_affinities,
            aggregate_vulnerability=aggregate,
            top_drug_targets=top_targets[:5],
        )

    def get_structural_ode_params(
        self,
        gene: str,
        mutation_position: int,
        structure: StructureData,
    ) -> Dict[str, float]:
        """
        Get structurally-modified ODE parameters for a gene mutation.

        Looks up the gene in gene_to_parameter_map.json, computes
        the structural modifier, and returns scaled parameters.

        Args:
            gene: Gene symbol
            mutation_position: 1-based residue position
            structure: AlphaFold structure

        Returns:
            Dict of parameter_name → structurally-scaled effect
        """
        gene_info = self.gene_param_map.get(gene, {})
        if not gene_info:
            return {}

        uniprot_id = gene_info.get('uniprot_id', '')
        modifiers = self.compute_mutation_impact(
            gene=gene,
            mutation_position=mutation_position,
            uniprot_id=uniprot_id,
            structure=structure,
        )

        param_name = gene_info.get('parameter', '')
        mutation_effect = gene_info.get('mutation_effect', 0.0)
        amplification_effect = gene_info.get('amplification_effect', 0.0)

        result = {}
        if param_name:
            result[f"{param_name}_mutation"] = mutation_effect * modifiers.ode_parameter_multiplier
            if amplification_effect:
                result[f"{param_name}_amplification"] = amplification_effect * modifiers.ode_parameter_multiplier
            result["structural_multiplier"] = modifiers.ode_parameter_multiplier
            result["stability_score"] = modifiers.stability_score
            result["confidence_tier"] = modifiers.confidence_tier

        return result
