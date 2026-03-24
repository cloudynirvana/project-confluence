"""
Structural Docking Module — Project Confluence
=================================================

Lightweight geometric docking — no AutoDock/Vina dependency.
Uses pocket volume, shape complementarity, and AlphaFold confidence
to estimate drug-binding compatibility.

This module provides the docking heuristics that feed into the
DrugEfficiencyEngine's efficacy_at_time() function, adding a
structural layer to the existing PK/PD model.

References:
    Le Guilloux et al. (2009) - Fpocket (pocket detection inspiration)
    Koes & Camacho (2012) - Pharmer (pharmacophore models)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np

from .alphafold_client import StructureData, BindingPocket

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# DRUG PHARMACOPHORE PROFILES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class DrugPharmacophore:
    """
    Simplified pharmacophore profile for a drug.

    Describes what kind of binding pocket a drug prefers,
    without requiring full 3D conformer generation.
    """
    name: str
    molecular_weight: float    # Da
    n_hbond_donors: int        # Hydrogen bond donors
    n_hbond_acceptors: int     # Hydrogen bond acceptors
    logp: float                # Lipophilicity (octanol-water partition)
    n_rotatable_bonds: int     # Flexibility
    preferred_pocket_volume: Tuple[float, float]  # Min, max (ų)
    preferred_pocket_depth: Tuple[float, float]    # Min, max (Å)
    charge_preference: str     # "positive", "negative", "neutral", "amphiphilic"

    @property
    def lipinski_violations(self) -> int:
        """Count Lipinski Rule of Five violations."""
        violations = 0
        if self.molecular_weight > 500:
            violations += 1
        if self.logp > 5:
            violations += 1
        if self.n_hbond_donors > 5:
            violations += 1
        if self.n_hbond_acceptors > 10:
            violations += 1
        return violations

    @property
    def drug_likeness_score(self) -> float:
        """0-1 score of overall drug-likeness."""
        score = 1.0
        score -= 0.25 * self.lipinski_violations
        if self.n_rotatable_bonds > 10:
            score -= 0.1
        return max(0.0, score)


# Pre-built pharmacophore profiles for intervention.py drugs
DRUG_PROFILES: Dict[str, DrugPharmacophore] = {
    "Dichloroacetate (DCA)": DrugPharmacophore(
        name="DCA", molecular_weight=128.9,
        n_hbond_donors=1, n_hbond_acceptors=2, logp=0.92,
        n_rotatable_bonds=1,
        preferred_pocket_volume=(150, 400),
        preferred_pocket_depth=(4, 8),
        charge_preference="negative",
    ),
    "Metformin": DrugPharmacophore(
        name="Metformin", molecular_weight=129.2,
        n_hbond_donors=4, n_hbond_acceptors=3, logp=-1.43,
        n_rotatable_bonds=2,
        preferred_pocket_volume=(150, 400),
        preferred_pocket_depth=(3, 7),
        charge_preference="positive",
    ),
    "2-Deoxyglucose (2-DG)": DrugPharmacophore(
        name="2-DG", molecular_weight=164.2,
        n_hbond_donors=4, n_hbond_acceptors=5, logp=-2.1,
        n_rotatable_bonds=1,
        preferred_pocket_volume=(200, 500),
        preferred_pocket_depth=(4, 8),
        charge_preference="neutral",
    ),
    "CB-839 (Telaglenastat)": DrugPharmacophore(
        name="CB-839", molecular_weight=444.5,
        n_hbond_donors=3, n_hbond_acceptors=6, logp=2.8,
        n_rotatable_bonds=8,
        preferred_pocket_volume=(400, 900),
        preferred_pocket_depth=(6, 12),
        charge_preference="neutral",
    ),
    "Olaparib (PARP inhibitor)": DrugPharmacophore(
        name="Olaparib", molecular_weight=434.5,
        n_hbond_donors=1, n_hbond_acceptors=7, logp=1.87,
        n_rotatable_bonds=5,
        preferred_pocket_volume=(400, 800),
        preferred_pocket_depth=(6, 12),
        charge_preference="neutral",
    ),
    "Vorinostat (SAHA, HDACi)": DrugPharmacophore(
        name="Vorinostat", molecular_weight=264.3,
        n_hbond_donors=3, n_hbond_acceptors=3, logp=1.47,
        n_rotatable_bonds=8,
        preferred_pocket_volume=(300, 700),
        preferred_pocket_depth=(5, 10),
        charge_preference="amphiphilic",
    ),
    "5-Azacitidine (DNMTi)": DrugPharmacophore(
        name="5-Azacitidine", molecular_weight=244.2,
        n_hbond_donors=4, n_hbond_acceptors=7, logp=-2.17,
        n_rotatable_bonds=2,
        preferred_pocket_volume=(250, 550),
        preferred_pocket_depth=(4, 9),
        charge_preference="neutral",
    ),
    "Hydroxychloroquine (HCQ)": DrugPharmacophore(
        name="HCQ", molecular_weight=335.9,
        n_hbond_donors=2, n_hbond_acceptors=4, logp=3.64,
        n_rotatable_bonds=9,
        preferred_pocket_volume=(350, 750),
        preferred_pocket_depth=(5, 11),
        charge_preference="positive",
    ),
    "Ferroptosis Inducer (Erastin/RSL3)": DrugPharmacophore(
        name="Erastin", molecular_weight=500.0,
        n_hbond_donors=2, n_hbond_acceptors=7, logp=3.4,
        n_rotatable_bonds=7,
        preferred_pocket_volume=(500, 1000),
        preferred_pocket_depth=(7, 14),
        charge_preference="neutral",
    ),
}


# ══════════════════════════════════════════════════════════════════════
# POCKET ANALYSIS
# ══════════════════════════════════════════════════════════════════════

@dataclass
class DockingResult:
    """Result of a geometric docking assessment."""
    drug_name: str
    target_gene: str
    pocket_index: int                # Which pocket was matched
    volume_compatibility: float      # 0-1 score
    depth_compatibility: float       # 0-1 score
    confidence_score: float          # pLDDT-based reliability
    shape_complementarity: float     # 0-1 geometric fit
    overall_score: float             # Weighted combination
    efficacy_multiplier: float       # For DrugEfficiencyEngine
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            "drug_name": self.drug_name,
            "target_gene": self.target_gene,
            "pocket_index": self.pocket_index,
            "volume_compatibility": round(self.volume_compatibility, 3),
            "depth_compatibility": round(self.depth_compatibility, 3),
            "confidence_score": round(self.confidence_score, 3),
            "shape_complementarity": round(self.shape_complementarity, 3),
            "overall_score": round(self.overall_score, 3),
            "efficacy_multiplier": round(self.efficacy_multiplier, 3),
            "notes": self.notes,
        }


class PocketAnalyzer:
    """
    Analyzes binding pockets for drug compatibility.

    Uses geometric heuristics to assess whether a pocket's shape,
    volume, and depth are compatible with a drug's pharmacophore.
    """

    @staticmethod
    def analyze_pocket(
        pocket: BindingPocket,
        structure: StructureData,
    ) -> Dict:
        """
        Detailed pocket analysis including charge distribution and shape.

        Returns dict with:
          - hydrophobic_fraction: Fraction of nonpolar residues
          - charged_fraction: Fraction of charged residues
          - polar_fraction: Fraction of polar uncharged residues
          - shape_descriptor: [elongation, sphericity, roughness]
        """
        # Classify residues by chemistry
        hydrophobic = {'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO'}
        charged = {'ASP', 'GLU', 'LYS', 'ARG', 'HIS'}
        polar = {'SER', 'THR', 'CYS', 'TYR', 'ASN', 'GLN'}

        pocket_residues = [
            structure.residues[i] for i in pocket.residue_indices
            if i < len(structure.residues)
        ]

        n_total = max(len(pocket_residues), 1)
        n_hydrophobic = sum(1 for r in pocket_residues if r.name in hydrophobic)
        n_charged = sum(1 for r in pocket_residues if r.name in charged)
        n_polar = sum(1 for r in pocket_residues if r.name in polar)

        # Shape descriptors from pocket coordinates
        pocket_coords = np.array([
            structure.residues[i].coords for i in pocket.residue_indices
            if i < len(structure.residues)
        ])

        if len(pocket_coords) < 3:
            shape = [0.5, 0.5, 0.5]
        else:
            # PCA-based shape (eigenvalue ratios)
            centered = pocket_coords - np.mean(pocket_coords, axis=0)
            cov = np.cov(centered.T)
            eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
            eigvals = np.maximum(eigvals, 1e-10)

            elongation = float(eigvals[0] / eigvals[1]) if eigvals[1] > 0 else 1.0
            sphericity = float(eigvals[2] / eigvals[0]) if eigvals[0] > 0 else 1.0

            # Roughness: deviation from ellipsoid
            dists = np.linalg.norm(centered, axis=1)
            roughness = float(np.std(dists) / (np.mean(dists) + 1e-10))

            shape = [
                min(elongation, 5.0),
                min(sphericity, 1.0),
                min(roughness, 2.0),
            ]

        return {
            "hydrophobic_fraction": round(n_hydrophobic / n_total, 3),
            "charged_fraction": round(n_charged / n_total, 3),
            "polar_fraction": round(n_polar / n_total, 3),
            "shape_descriptor": [round(s, 3) for s in shape],
        }


class DrugTargetMatcher:
    """
    Matches drugs to binding pockets using geometric compatibility.

    For each drug-pocket pair, computes volume fit, depth fit,
    charge complementarity, and shape compatibility. The combined
    score becomes an efficacy multiplier for the PK/PD model.
    """

    def __init__(self):
        self.pocket_analyzer = PocketAnalyzer()

    def dock(
        self,
        drug_name: str,
        target_structure: StructureData,
        drug_profile: Optional[DrugPharmacophore] = None,
    ) -> List[DockingResult]:
        """
        Score all pockets in a target structure against a drug.

        Args:
            drug_name: Drug name (from intervention.py)
            target_structure: AlphaFold structure with detected pockets
            drug_profile: Drug pharmacophore (None = look up from library)

        Returns:
            List of DockingResult, sorted by overall score (best first)
        """
        if drug_profile is None:
            drug_profile = DRUG_PROFILES.get(drug_name)
            if drug_profile is None:
                # Unknown drug → use generic small molecule profile
                drug_profile = DrugPharmacophore(
                    name=drug_name, molecular_weight=300.0,
                    n_hbond_donors=2, n_hbond_acceptors=4, logp=1.5,
                    n_rotatable_bonds=5,
                    preferred_pocket_volume=(300, 700),
                    preferred_pocket_depth=(5, 10),
                    charge_preference="neutral",
                )

        results = []
        for idx, pocket in enumerate(target_structure.pockets):
            result = self._score_pocket(
                drug_name, drug_profile, pocket, target_structure, idx
            )
            results.append(result)

        results.sort(key=lambda r: r.overall_score, reverse=True)
        return results

    def _score_pocket(
        self,
        drug_name: str,
        drug_profile: DrugPharmacophore,
        pocket: BindingPocket,
        structure: StructureData,
        pocket_idx: int,
    ) -> DockingResult:
        """Score a single drug-pocket pair."""

        # 1. Volume compatibility
        vol_min, vol_max = drug_profile.preferred_pocket_volume
        if vol_min <= pocket.volume <= vol_max:
            vol_score = 1.0
        elif pocket.volume < vol_min:
            vol_score = max(0.0, pocket.volume / vol_min)
        else:
            # Too large is less penalized than too small
            vol_score = max(0.2, 1.0 - (pocket.volume - vol_max) / (vol_max * 2))
        vol_score = float(np.clip(vol_score, 0.0, 1.0))

        # 2. Depth compatibility
        depth_min, depth_max = drug_profile.preferred_pocket_depth
        if depth_min <= pocket.depth <= depth_max:
            depth_score = 1.0
        elif pocket.depth < depth_min:
            depth_score = max(0.0, pocket.depth / depth_min)
        else:
            depth_score = max(0.3, 1.0 - (pocket.depth - depth_max) / (depth_max * 2))
        depth_score = float(np.clip(depth_score, 0.0, 1.0))

        # 3. Confidence score (pLDDT reliability)
        conf_score = pocket.avg_plddt / 100.0

        # 4. Shape complementarity (charge matching)
        pocket_analysis = self.pocket_analyzer.analyze_pocket(pocket, structure)
        shape_score = self._charge_compatibility(
            drug_profile.charge_preference, pocket_analysis
        )

        # Combine scores
        overall = (
            0.30 * vol_score +
            0.20 * depth_score +
            0.25 * conf_score +
            0.25 * shape_score
        )

        # Drug-likeness penalty
        overall *= (0.7 + 0.3 * drug_profile.drug_likeness_score)

        # Efficacy multiplier: 0.5 (no pocket match) to 1.5 (perfect match)
        efficacy = 0.5 + overall

        # Generate notes
        notes = []
        if vol_score > 0.8:
            notes.append("Good volume fit")
        if vol_score < 0.3:
            notes.append("Poor volume fit")
        if conf_score > 0.8:
            notes.append("High-confidence pocket")
        if conf_score < 0.5:
            notes.append("Low-confidence pocket")

        return DockingResult(
            drug_name=drug_name,
            target_gene=structure.gene_name,
            pocket_index=pocket_idx,
            volume_compatibility=vol_score,
            depth_compatibility=depth_score,
            confidence_score=conf_score,
            shape_complementarity=shape_score,
            overall_score=round(overall, 3),
            efficacy_multiplier=round(efficacy, 3),
            notes="; ".join(notes),
        )

    def _charge_compatibility(
        self,
        drug_charge: str,
        pocket_analysis: Dict,
    ) -> float:
        """
        Score charge complementarity between drug and pocket.

        Positive drugs prefer negatively charged pockets and vice versa.
        Neutral drugs prefer hydrophobic pockets.
        """
        hydro = pocket_analysis["hydrophobic_fraction"]
        charged = pocket_analysis["charged_fraction"]
        polar = pocket_analysis["polar_fraction"]

        if drug_charge == "positive":
            # Prefers acidic pockets (ASP, GLU)
            return 0.3 + 0.5 * charged + 0.2 * polar
        elif drug_charge == "negative":
            return 0.3 + 0.5 * charged + 0.2 * polar
        elif drug_charge == "neutral":
            # Prefers hydrophobic
            return 0.3 + 0.6 * hydro + 0.1 * polar
        elif drug_charge == "amphiphilic":
            # Works in mixed environments
            balance = 1.0 - abs(hydro - 0.4)
            return 0.4 + 0.6 * max(0, balance)
        else:
            return 0.5


def compute_docking_score(
    drug_name: str,
    target_structure: StructureData,
) -> float:
    """
    Convenience function: compute best docking score for a drug-target pair.

    Returns the best overall docking score (0-1), or 0.0 if no pockets.
    """
    if not target_structure.pockets:
        return 0.0

    matcher = DrugTargetMatcher()
    results = matcher.dock(drug_name, target_structure)

    if results:
        return results[0].overall_score
    return 0.0
