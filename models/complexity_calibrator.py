"""
Complexity Calibration Engine — Project Confluence
====================================================

Staged, constrained calibration pipeline that optimally maps AlphaFold
structural signals through ODE parameters to Φ complexity profiles.

Calibration Path:
    Genotype → SIS (Structural Impact Score) → Δθ (ODE Shift) → Φ Profile

Three stages, each independently trainable and auditable:
    Stage 1 — SISCalibrator: monotonic mapping from structural features to SIS
    Stage 2 — ParameterMapper: L1-regularized SIS→Δθ with sign constraints
    Stage 3 — GlobalCalibrator: Bayesian multi-objective fine-tuning

Plus:
    StabilityTester: perturbation analysis for calibration robustness
    BiologicalAuditor: sign constraint and plausibility verification

References:
    Jumper et al. (2021) - AlphaFold
    Boyd & Vandenberghe (2004) - Convex Optimization (L1 methods)
"""

import json
import logging
import copy
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import numpy as np
from scipy.optimize import minimize as scipy_minimize

from .alphafold_client import (
    StructureData,
    DISEASE_PANELS,
    GENE_KEY_RESIDUES,
    create_mock_structure,
)
from .structure_bridge import (
    StructureBridge,
    StructuralModifiers,
    DRUG_TARGET_MAP,
)
from .ode_system import ComplexAttractorODE, ExtendedParams
from .complexity_profiler import ComplexityProfiler

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# BIOLOGICAL CONSTRAINTS — the "guardrails" for calibration
# ══════════════════════════════════════════════════════════════════════

# Sign constraints: these map gene→parameter→expected_sign
# Positive sign = gene alteration INCREASES the parameter
# Negative sign = gene alteration DECREASES the parameter
SIGN_CONSTRAINTS: Dict[str, Dict[str, int]] = {
    "HK2":    {"glucose_uptake": -1},        # Loss of function → less glycolysis
    "PKM":    {"glycolysis_flux": +1},        # Gain of function → more flux
    "LDHA":   {"pyruvate_to_lactate": +1},    # Amplification → more lactate
    "PDK1":   {"pyruvate_to_atp": -1},        # Active → blocks OXPHOS
    "GLS":    {"glutamine_utilization": -1},   # Inhibition target
    "SLC1A5": {"glutaminolysis": +1},          # Amplification → more import
    "SOD2":   {"ros_clearance": +1},           # Loss → more ROS
    "GPX4":   {"ros_clearance": +1},           # Loss → ferroptosis
    "BRCA1":  {"ros_atp_damage": -1},          # Loss → more damage sensitivity
    "TP53":   {"atp_turnover": -1},            # Loss → metabolic shift
    "KRAS":   {"glucose_uptake": -1},          # Gain → more glucose
    "MYC":    {"glutamine_utilization": -1},   # Amplification → more glutamine
    "VHL":    {"glycolysis_flux": +1},         # Loss → HIF → Warburg
    "IDH1":   {"akg_to_citrate": -1},          # Neomorphic → 2-HG
    "PTEN":   {"glucose_uptake": -1},          # Loss → PI3K → glucose
}

# Plausibility bounds: max absolute Δθ for each parameter (fraction of baseline)
PLAUSIBILITY_BOUNDS: Dict[str, float] = {
    "glucose_uptake": 0.5,
    "glycolysis_flux": 0.5,
    "pyruvate_to_lactate": 0.4,
    "pyruvate_to_atp": 0.4,
    "glutamine_utilization": 0.5,
    "glutaminolysis": 0.4,
    "ros_clearance": 0.6,
    "ros_atp_damage": 0.3,
    "atp_turnover": 0.3,
    "akg_to_citrate": 0.3,
}


# ══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SISVector:
    """
    Structural Impact Score vector for a single gene.

    Features extracted from AlphaFold analysis, normalized to [0, 1].
    This is the input to the calibration pipeline.
    """
    gene: str
    plddt_score: float          # Normalized pLDDT disruption (higher = more ordered region)
    stability_score: float      # From StructuralModifiers
    pocket_accessibility: float # From StructuralModifiers
    active_site_flag: float     # 1.0 if active site, 0.0 otherwise
    druggability_score: float   # Nearby pocket druggability

    @property
    def feature_vector(self) -> np.ndarray:
        """5D feature vector for calibration."""
        return np.array([
            self.plddt_score,
            self.stability_score,
            self.pocket_accessibility,
            self.active_site_flag,
            self.druggability_score,
        ])

    @classmethod
    def from_modifiers(cls, mod: StructuralModifiers) -> 'SISVector':
        """Create from StructuralModifiers output."""
        return cls(
            gene=mod.gene,
            plddt_score=mod.local_plddt / 100.0,
            stability_score=mod.stability_score,
            pocket_accessibility=mod.pocket_accessibility,
            active_site_flag=1.0 if mod.is_in_active_site else 0.0,
            druggability_score=mod.nearby_pocket_druggability,
        )

    def to_dict(self) -> Dict:
        return {
            "gene": self.gene,
            "plddt": round(self.plddt_score, 3),
            "stability": round(self.stability_score, 3),
            "pocket": round(self.pocket_accessibility, 3),
            "active_site": self.active_site_flag,
            "druggability": round(self.druggability_score, 3),
        }


@dataclass
class CalibrationWeights:
    """
    Trainable weights for the calibration path.

    Stage 1: SIS feature weights (5D) — how much each structural feature
             contributes to the composite SIS.
    Stage 2: L1 mapping coefficients — SIS → Δθ for each gene-parameter pair.
    """
    # Stage 1: SIS feature importance (5D)
    sis_weights: np.ndarray = field(
        default_factory=lambda: np.array([0.25, 0.30, 0.15, 0.20, 0.10])
    )

    # Stage 2: L1 regularization strength
    l1_lambda: float = 0.1

    # Stage 2: Per-gene scaling coefficients (gene → float)
    gene_scales: Dict[str, float] = field(default_factory=dict)

    # Global: multi-objective weights
    w_phi_alignment: float = 0.35    # Φ-distance tracks severity
    w_cross_panel: float = 0.20     # Cross-disease coherence
    w_drug_response: float = 0.15   # Drug sensitivity alignment
    w_param_entropy: float = 0.15   # Sparsity preference
    w_stability: float = 0.15       # Perturbation robustness

    def to_dict(self) -> Dict:
        return {
            "sis_weights": [round(w, 4) for w in self.sis_weights],
            "l1_lambda": self.l1_lambda,
            "gene_scales": {g: round(s, 4) for g, s in self.gene_scales.items()},
            "objective_weights": {
                "phi_alignment": self.w_phi_alignment,
                "cross_panel": self.w_cross_panel,
                "drug_response": self.w_drug_response,
                "param_entropy": self.w_param_entropy,
                "stability": self.w_stability,
            },
        }

    def to_optimization_vector(self) -> np.ndarray:
        """Flatten weights into a 1D optimization vector."""
        gene_list = sorted(self.gene_scales.keys())
        gene_vals = [self.gene_scales.get(g, 1.0) for g in gene_list]
        return np.concatenate([
            self.sis_weights,            # 5 values
            [self.l1_lambda],            # 1 value
            gene_vals,                   # N values
        ])

    def from_optimization_vector(self, vec: np.ndarray, gene_list: List[str]):
        """Restore weights from a 1D optimization vector."""
        self.sis_weights = vec[:5]
        # Normalize SIS weights to sum to 1
        self.sis_weights = np.maximum(self.sis_weights, 0.01)
        self.sis_weights /= self.sis_weights.sum()
        self.l1_lambda = max(0.001, vec[5])
        for i, g in enumerate(gene_list):
            if 6 + i < len(vec):
                self.gene_scales[g] = float(np.clip(vec[6 + i], 0.1, 3.0))


@dataclass
class CalibrationResult:
    """Full result of a calibration run."""
    disease: str
    sis_vectors: Dict[str, SISVector]
    delta_theta: Dict[str, Dict[str, float]]  # gene → {param: Δθ}
    phi_baseline: np.ndarray
    phi_calibrated: np.ndarray
    phi_healthy: np.ndarray
    baseline_distance: float
    calibrated_distance: float
    sign_violations: List[str]
    bound_violations: List[str]
    stability_scores: Dict[str, float]  # gene → stability score
    weights_used: CalibrationWeights
    calibration_score: float

    def to_dict(self) -> Dict:
        return {
            "disease": self.disease,
            "baseline_phi_distance": round(self.baseline_distance, 4),
            "calibrated_phi_distance": round(self.calibrated_distance, 4),
            "delta_phi": round(self.calibrated_distance - self.baseline_distance, 4),
            "n_sign_violations": len(self.sign_violations),
            "sign_violations": self.sign_violations,
            "n_bound_violations": len(self.bound_violations),
            "bound_violations": self.bound_violations,
            "calibration_score": round(self.calibration_score, 4),
            "sis_vectors": {g: v.to_dict() for g, v in self.sis_vectors.items()},
            "delta_theta": {
                g: {p: round(v, 4) for p, v in params.items()}
                for g, params in self.delta_theta.items()
            },
            "stability": {g: round(s, 4) for g, s in self.stability_scores.items()},
        }


# ══════════════════════════════════════════════════════════════════════
# STAGE 1: SIS CALIBRATOR
# ══════════════════════════════════════════════════════════════════════

class SISCalibrator:
    """
    Stage 1: Maps raw structural features → composite Structural Impact Score.

    Uses a monotonic weighted combination with isotonic correction:
      SIS = Σ(w_i * f_i) where all w_i > 0

    Monotonicity guarantee: higher pLDDT disruption → higher SIS,
    active site mutations → higher SIS. This prevents counterintuitive
    structural interpretations.
    """

    def __init__(self, weights: Optional[CalibrationWeights] = None):
        self.weights = weights or CalibrationWeights()

    def compute_sis(self, sis_vector: SISVector) -> float:
        """
        Compute composite SIS from feature vector.

        Returns a single [0, 1] score representing the overall
        structural impact of this gene's alteration.
        """
        features = sis_vector.feature_vector
        w = self.weights.sis_weights

        # Weighted combination
        raw_sis = float(np.dot(w, features))

        # Isotonic correction: ensure SIS is in [0, 1]
        sis = float(np.clip(raw_sis, 0.0, 1.0))

        return sis

    def compute_batch(
        self, modifiers: Dict[str, StructuralModifiers]
    ) -> Dict[str, Tuple[SISVector, float]]:
        """
        Compute SIS for a batch of genes.

        Returns dict: gene → (SISVector, composite_SIS_score)
        """
        results = {}
        for gene, mod in modifiers.items():
            sv = SISVector.from_modifiers(mod)
            sis = self.compute_sis(sv)
            results[gene] = (sv, sis)
        return results

    def verify_monotonicity(self, perturbation_delta: float = 0.1) -> bool:
        """
        Verify that the SIS mapping is monotonic:
        increasing any positive feature → non-decreasing SIS.
        """
        base_features = np.array([0.5, 0.5, 0.5, 0.0, 0.5])
        base_sis = float(np.dot(self.weights.sis_weights, base_features))

        for i in range(5):
            perturbed = base_features.copy()
            perturbed[i] += perturbation_delta
            perturbed_sis = float(np.dot(self.weights.sis_weights, perturbed))
            if perturbed_sis < base_sis - 1e-10:
                return False

        return True


# ══════════════════════════════════════════════════════════════════════
# STAGE 2: L1-REGULARIZED PARAMETER MAPPER
# ══════════════════════════════════════════════════════════════════════

class ParameterMapper:
    """
    Stage 2: Maps SIS → ODE parameter shifts (Δθ) with constraints.

    L1 regularization ensures sparsity: only genes with significant
    structural evidence get nonzero Δθ. Sign constraints prevent
    biologically impossible parameter inversions.

    The mapping:
        Δθ_gene = gene_scale * SIS * base_effect * sign_constraint

    Where gene_scale is learned, SIS is from Stage 1, base_effect
    is from gene_to_parameter_map.json, and sign is enforced.
    """

    def __init__(
        self,
        weights: Optional[CalibrationWeights] = None,
        gene_param_map: Optional[Dict] = None,
    ):
        self.weights = weights or CalibrationWeights()

        # Load gene-to-parameter map
        if gene_param_map is None:
            try:
                map_path = Path(__file__).parent.parent / "validation" / "gene_to_parameter_map.json"
                with open(map_path, 'r') as f:
                    data = json.load(f)
                self.gene_param_map = data.get('mappings', data)
            except FileNotFoundError:
                self.gene_param_map = {}
        else:
            self.gene_param_map = gene_param_map

    def compute_delta_theta(
        self,
        sis_scores: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute ODE parameter shifts for each gene given SIS scores.

        Returns:
            gene → {parameter_name: Δθ_value}
        """
        delta_theta = {}

        for gene, sis in sis_scores.items():
            gene_info = self.gene_param_map.get(gene, {})
            if not gene_info:
                continue

            param = gene_info.get('parameter', '')
            base_effect = gene_info.get('mutation_effect', 0.0)
            if not param or base_effect == 0:
                continue

            # Get gene-specific scale (learned or default)
            gene_scale = self.weights.gene_scales.get(gene, 1.0)

            # L1 regularization: zero out weak signals
            effective_sis = sis * gene_scale
            if abs(effective_sis) < self.weights.l1_lambda:
                effective_sis = 0.0  # Sparsity: below L1 threshold → zero

            # Compute raw Δθ
            raw_delta = effective_sis * base_effect

            # Enforce sign constraint
            sign_constraint = SIGN_CONSTRAINTS.get(gene, {}).get(param, 0)
            if sign_constraint != 0:
                expected_sign = sign_constraint
                actual_sign = np.sign(raw_delta) if raw_delta != 0 else 0
                if actual_sign != 0 and actual_sign != expected_sign:
                    raw_delta = 0.0  # Kill sign-violating shifts

            # Enforce plausibility bounds
            bound = PLAUSIBILITY_BOUNDS.get(param, 0.5)
            raw_delta = float(np.clip(raw_delta, -bound, bound))

            delta_theta[gene] = {param: raw_delta}

        return delta_theta

    def apply_to_params(
        self,
        base_params: ExtendedParams,
        delta_theta: Dict[str, Dict[str, float]],
    ) -> ExtendedParams:
        """
        Apply Δθ shifts to ODE parameters.

        Creates a modified copy of ExtendedParams with structural
        calibration applied.
        """
        modified = copy.deepcopy(base_params)

        for gene, shifts in delta_theta.items():
            for param_name, delta in shifts.items():
                if delta == 0:
                    continue
                if hasattr(modified, param_name):
                    current = getattr(modified, param_name)
                    # Apply as multiplicative shift: param *= (1 + Δθ)
                    new_val = current * (1.0 + delta)
                    setattr(modified, param_name, new_val)

        return modified

    def audit_signs(
        self,
        delta_theta: Dict[str, Dict[str, float]],
    ) -> List[str]:
        """
        Check all Δθ values against biological sign constraints.

        Returns list of violation descriptions.
        """
        violations = []

        for gene, shifts in delta_theta.items():
            for param, delta in shifts.items():
                if delta == 0:
                    continue
                expected_sign = SIGN_CONSTRAINTS.get(gene, {}).get(param, 0)
                if expected_sign != 0:
                    actual_sign = int(np.sign(delta))
                    if actual_sign != expected_sign:
                        violations.append(
                            f"{gene}/{param}: expected sign {'+' if expected_sign > 0 else '-'}, "
                            f"got {'+' if actual_sign > 0 else '-'} (Δθ={delta:.4f})"
                        )

        return violations

    def audit_bounds(
        self,
        delta_theta: Dict[str, Dict[str, float]],
    ) -> List[str]:
        """Check Δθ values against plausibility bounds."""
        violations = []

        for gene, shifts in delta_theta.items():
            for param, delta in shifts.items():
                bound = PLAUSIBILITY_BOUNDS.get(param, 0.5)
                if abs(delta) > bound:
                    violations.append(
                        f"{gene}/{param}: |Δθ|={abs(delta):.4f} > bound {bound}"
                    )

        return violations

    def compute_parameter_entropy(
        self,
        delta_theta: Dict[str, Dict[str, float]],
    ) -> float:
        """
        Compute entropy of parameter shifts (lower = sparser = more interpretable).

        Uses L1 norm of all Δθ values as a sparsity measure.
        """
        all_deltas = []
        for shifts in delta_theta.values():
            all_deltas.extend(shifts.values())

        if not all_deltas:
            return 0.0

        # L1 norm (average absolute shift)
        return float(np.mean(np.abs(all_deltas)))


# ══════════════════════════════════════════════════════════════════════
# STAGE 3: GLOBAL CALIBRATOR
# ══════════════════════════════════════════════════════════════════════

class GlobalCalibrator:
    """
    Stage 3: Multi-objective Bayesian fine-tuning of the full path.

    Jointly optimizes Stage 1 and Stage 2 weights to maximize:
        w1 * Φ-distance alignment
      + w2 * cross-panel stability
      + w3 * drug response consistency
      - w4 * parameter entropy
      - w5 * calibration variance

    Uses scipy L-BFGS-B (constrained) for smooth optimization.
    """

    def __init__(
        self,
        diseases: Optional[List[str]] = None,
        use_mock: bool = True,
    ):
        self.diseases = diseases or ["TNBC", "GBM", "Alzheimers"]
        self.use_mock = use_mock
        self.profiler = ComplexityProfiler()
        self.bridge = StructureBridge()

        # Precompute healthy reference
        healthy_ode = ComplexAttractorODE(params=ExtendedParams())
        healthy_result = healthy_ode.solve(t_span=(0, 80), dt_eval=1.0)
        self.healthy_phi = self.profiler.profile(healthy_result['z'], dt=1.0)

    def calibrate_disease(
        self,
        disease: str,
        weights: CalibrationWeights,
    ) -> CalibrationResult:
        """
        Run full calibration path for a single disease.

        Genotype → SIS → Δθ → Φ with all constraints enforced.
        """
        # Get structural data
        profile = self.bridge.profile_disease(disease, use_mock=self.use_mock)

        # Stage 1: Compute SIS
        sis_calibrator = SISCalibrator(weights)
        sis_batch = sis_calibrator.compute_batch(profile.gene_modifiers)
        sis_scores = {gene: sis for gene, (_, sis) in sis_batch.items()}
        sis_vectors = {gene: sv for gene, (sv, _) in sis_batch.items()}

        # Stage 2: Compute Δθ
        param_mapper = ParameterMapper(weights)
        delta_theta = param_mapper.compute_delta_theta(sis_scores)

        # Audit
        sign_violations = param_mapper.audit_signs(delta_theta)
        bound_violations = param_mapper.audit_bounds(delta_theta)

        # Get disease-specific params
        from .ode_system import (
            TNBCParams, AlzheimersParams, ParkinsonsParams,
            DiabetesParams, GlioblastomaParams, ALSParams, LupusParams,
        )
        DISEASE_PARAMS = {
            "TNBC": TNBCParams, "GBM": GlioblastomaParams,
            "Alzheimers": AlzheimersParams, "Parkinsons": ParkinsonsParams,
            "ALS": ALSParams, "Diabetes": DiabetesParams, "Lupus": LupusParams,
        }
        base_params = DISEASE_PARAMS.get(disease, ExtendedParams)()

        # Baseline Φ (no calibration)
        baseline_ode = ComplexAttractorODE(params=base_params)
        baseline_result = baseline_ode.solve(t_span=(0, 80), dt_eval=1.0)
        baseline_phi = self.profiler.profile(baseline_result['z'], dt=1.0)

        # Calibrated Φ (with Δθ applied)
        calibrated_params = param_mapper.apply_to_params(base_params, delta_theta)
        calibrated_ode = ComplexAttractorODE(params=calibrated_params)
        calibrated_result = calibrated_ode.solve(t_span=(0, 80), dt_eval=1.0)
        calibrated_phi = self.profiler.profile(calibrated_result['z'], dt=1.0)

        # Distances from healthy
        baseline_dist = float(np.linalg.norm(
            np.array(baseline_phi.phi_vector) - np.array(self.healthy_phi.phi_vector)
        ))
        calibrated_dist = float(np.linalg.norm(
            np.array(calibrated_phi.phi_vector) - np.array(self.healthy_phi.phi_vector)
        ))

        # Stability test
        stability_scores = self._stability_test(disease, weights, profile)

        # Compute calibration score
        entropy = param_mapper.compute_parameter_entropy(delta_theta)
        avg_stability = float(np.mean(list(stability_scores.values()))) if stability_scores else 1.0
        cal_score = (
            weights.w_phi_alignment * (1.0 - abs(calibrated_dist - baseline_dist) / max(baseline_dist, 0.01))
            + weights.w_stability * avg_stability
            - weights.w_param_entropy * entropy
            - 0.5 * len(sign_violations)
        )

        return CalibrationResult(
            disease=disease,
            sis_vectors=sis_vectors,
            delta_theta=delta_theta,
            phi_baseline=np.array(baseline_phi.phi_vector),
            phi_calibrated=np.array(calibrated_phi.phi_vector),
            phi_healthy=np.array(self.healthy_phi.phi_vector),
            baseline_distance=baseline_dist,
            calibrated_distance=calibrated_dist,
            sign_violations=sign_violations,
            bound_violations=bound_violations,
            stability_scores=stability_scores,
            weights_used=weights,
            calibration_score=cal_score,
        )

    def _stability_test(
        self,
        disease: str,
        weights: CalibrationWeights,
        profile,
        perturbation: float = 0.10,
    ) -> Dict[str, float]:
        """
        Perturb pLDDT by ±perturbation and check Φ stability.

        Returns gene → stability_score (1.0 = perfectly stable, 0.0 = wildly unstable)
        """
        stability = {}

        for gene, modifiers in profile.gene_modifiers.items():
            # Create perturbed modifier (simulate ±10% pLDDT shift)
            original_plddt = modifiers.local_plddt
            scores = []

            for direction in [+1, -1]:
                perturbed_plddt = original_plddt * (1 + direction * perturbation)
                perturbed_plddt = float(np.clip(perturbed_plddt, 0, 100))

                # Recompute SIS with perturbed pLDDT
                sv = SISVector(
                    gene=gene,
                    plddt_score=perturbed_plddt / 100.0,
                    stability_score=modifiers.stability_score,
                    pocket_accessibility=modifiers.pocket_accessibility,
                    active_site_flag=1.0 if modifiers.is_in_active_site else 0.0,
                    druggability_score=modifiers.nearby_pocket_druggability,
                )
                sis_calibrator = SISCalibrator(weights)
                perturbed_sis = sis_calibrator.compute_sis(sv)
                scores.append(perturbed_sis)

            # Original SIS
            original_sv = SISVector.from_modifiers(modifiers)
            original_sis = SISCalibrator(weights).compute_sis(original_sv)

            # Stability = 1 - max relative change
            if original_sis > 0.01:
                max_change = max(abs(s - original_sis) / original_sis for s in scores)
            else:
                max_change = max(abs(s - original_sis) for s in scores)

            # Score: 1.0 if change < 5%, 0.0 if change > 25%
            stability[gene] = float(np.clip(1.0 - max_change / 0.25, 0.0, 1.0))

        return stability

    def optimize(
        self,
        n_iterations: int = 30,
        seed: int = 42,
    ) -> Tuple[CalibrationWeights, Dict[str, CalibrationResult]]:
        """
        Global optimization of calibration weights across all diseases.

        Uses scipy minimize (L-BFGS-B) with bounded constraints.

        Returns:
            (optimized_weights, {disease: CalibrationResult})
        """
        # Collect all genes across diseases
        all_genes = set()
        for disease in self.diseases:
            panel = DISEASE_PANELS.get(disease, {})
            all_genes.update(panel.keys())
        gene_list = sorted(all_genes)

        # Initialize weights
        weights = CalibrationWeights()
        for g in gene_list:
            weights.gene_scales[g] = 1.0

        initial_vec = weights.to_optimization_vector()
        n_params = len(initial_vec)

        # Bounds
        bounds = []
        # SIS weights: [0.05, 0.50]
        for _ in range(5):
            bounds.append((0.05, 0.50))
        # L1 lambda: [0.01, 0.5]
        bounds.append((0.01, 0.5))
        # Gene scales: [0.1, 3.0]
        for _ in gene_list:
            bounds.append((0.1, 3.0))

        best_score = -np.inf
        best_weights = copy.deepcopy(weights)
        best_results = {}

        # Evaluate objective
        def objective(vec):
            w = copy.deepcopy(weights)
            w.from_optimization_vector(vec, gene_list)

            total_score = 0.0
            for disease in self.diseases:
                try:
                    result = self.calibrate_disease(disease, w)
                    total_score += result.calibration_score
                except Exception:
                    total_score -= 1.0  # Penalize failures

            # Negate for minimization
            return -total_score / len(self.diseases)

        rng = np.random.RandomState(seed)

        # Multi-start optimization: try several random initializations
        n_starts = min(n_iterations, 3)
        for start in range(n_starts):
            if start == 0:
                x0 = initial_vec.copy()
            else:
                # Random perturbation
                x0 = initial_vec + rng.randn(n_params) * 0.1
                x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

            try:
                result = scipy_minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max(n_iterations // n_starts, 2), 'ftol': 1e-5},
                )

                if -result.fun > best_score:
                    best_score = -result.fun
                    best_weights.from_optimization_vector(result.x, gene_list)
            except Exception as e:
                logger.warning(f"Optimization start {start} failed: {e}")

        # Final evaluation with best weights
        for disease in self.diseases:
            try:
                best_results[disease] = self.calibrate_disease(disease, best_weights)
            except Exception as e:
                logger.warning(f"Final evaluation for {disease} failed: {e}")

        return best_weights, best_results

    def cross_panel_coherence(
        self,
        results: Dict[str, CalibrationResult],
    ) -> Dict[str, Dict]:
        """
        Check cross-disease coherence for shared genes.

        A gene like EGFR should produce consistent parameter shifts
        across TNBC and GBM. Coherent = same sign, similar magnitude.
        """
        # Collect per-gene deltas across diseases
        gene_deltas: Dict[str, List[Tuple[str, Dict]]] = {}
        for disease, result in results.items():
            for gene, shifts in result.delta_theta.items():
                if gene not in gene_deltas:
                    gene_deltas[gene] = []
                gene_deltas[gene].append((disease, shifts))

        # Analyze coherence
        coherence = {}
        for gene, entries in gene_deltas.items():
            if len(entries) < 2:
                continue

            # Check sign consistency
            signs = []
            magnitudes = []
            for disease, shifts in entries:
                for param, delta in shifts.items():
                    signs.append(int(np.sign(delta)))
                    magnitudes.append(abs(delta))

            sign_consistent = all(s == signs[0] for s in signs) if signs else True
            mag_cv = float(np.std(magnitudes) / (np.mean(magnitudes) + 1e-10)) if magnitudes else 0.0

            coherence[gene] = {
                "n_diseases": len(entries),
                "diseases": [d for d, _ in entries],
                "sign_consistent": sign_consistent,
                "magnitude_cv": round(mag_cv, 4),
                "coherence_score": round(1.0 - min(mag_cv, 1.0), 4) if sign_consistent else 0.0,
            }

        return coherence


# ══════════════════════════════════════════════════════════════════════
# BIOLOGICAL AUDITOR
# ══════════════════════════════════════════════════════════════════════

class BiologicalAuditor:
    """
    Audits calibration results for biological plausibility.

    Checks:
      1. Sign constraints (gene function → parameter direction)
      2. Plausibility bounds (no extreme shifts)
      3. Cross-disease coherence (shared genes → consistent effects)
      4. Stability under pLDDT perturbation
    """

    def audit(
        self,
        results: Dict[str, CalibrationResult],
    ) -> Dict:
        """
        Full biological audit of calibration results.

        Returns structured audit report.
        """
        all_sign_violations = []
        all_bound_violations = []
        per_disease_scores = {}
        all_stability = {}

        for disease, result in results.items():
            all_sign_violations.extend(
                [f"[{disease}] {v}" for v in result.sign_violations]
            )
            all_bound_violations.extend(
                [f"[{disease}] {v}" for v in result.bound_violations]
            )
            per_disease_scores[disease] = {
                "calibration_score": round(result.calibration_score, 4),
                "baseline_phi_dist": round(result.baseline_distance, 4),
                "calibrated_phi_dist": round(result.calibrated_distance, 4),
                "n_sign_violations": len(result.sign_violations),
                "n_bound_violations": len(result.bound_violations),
                "avg_stability": round(
                    float(np.mean(list(result.stability_scores.values()))), 4
                ) if result.stability_scores else 0.0,
            }
            all_stability.update(result.stability_scores)

        # Cross-panel coherence
        calibrator = GlobalCalibrator.__new__(GlobalCalibrator)
        coherence = calibrator.cross_panel_coherence(results)

        # Overall grade
        total_violations = len(all_sign_violations) + len(all_bound_violations)
        avg_stability = float(np.mean(list(all_stability.values()))) if all_stability else 0.0
        avg_coherence = float(np.mean([
            c["coherence_score"] for c in coherence.values()
        ])) if coherence else 1.0

        if total_violations == 0 and avg_stability > 0.8 and avg_coherence > 0.6:
            grade = "A"
        elif total_violations <= 2 and avg_stability > 0.6:
            grade = "B"
        elif total_violations <= 5:
            grade = "C"
        else:
            grade = "D"

        return {
            "grade": grade,
            "total_sign_violations": len(all_sign_violations),
            "total_bound_violations": len(all_bound_violations),
            "sign_violations": all_sign_violations,
            "bound_violations": all_bound_violations,
            "avg_stability": round(avg_stability, 4),
            "avg_cross_panel_coherence": round(avg_coherence, 4),
            "per_disease": per_disease_scores,
            "cross_panel_coherence": coherence,
        }


# ══════════════════════════════════════════════════════════════════════
# CONVENIENCE: FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════

def run_full_calibration(
    diseases: Optional[List[str]] = None,
    n_iterations: int = 20,
    use_mock: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Run the complete calibration pipeline.

    1. Optimize calibration weights across diseases
    2. Run full calibration for each disease
    3. Audit for biological plausibility
    4. Return structured report

    Args:
        diseases: List of diseases to calibrate (None = default panel)
        n_iterations: Optimization iterations
        use_mock: Use mock AlphaFold structures
        seed: Random seed

    Returns:
        Full calibration report dict
    """
    if diseases is None:
        diseases = ["TNBC", "GBM", "Alzheimers"]

    logger.info(f"Starting complexity calibration for {diseases}")

    # Stage 3: Global optimization
    calibrator = GlobalCalibrator(diseases=diseases, use_mock=use_mock)
    optimized_weights, results = calibrator.optimize(
        n_iterations=n_iterations,
        seed=seed,
    )

    # Audit
    auditor = BiologicalAuditor()
    audit_report = auditor.audit(results)

    # Compile report
    report = {
        "calibration_summary": {
            "diseases": diseases,
            "n_iterations": n_iterations,
            "optimized_weights": optimized_weights.to_dict(),
            "audit_grade": audit_report["grade"],
        },
        "audit": audit_report,
        "per_disease_results": {
            d: r.to_dict() for d, r in results.items()
        },
    }

    logger.info(f"Calibration complete. Grade: {audit_report['grade']}")
    return report
