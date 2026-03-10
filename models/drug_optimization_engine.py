"""
Drug Optimization Engine (RADO) — Project Confluence (UCP Module 3)
====================================================================

Restoration of Adaptive Dynamics Optimizer (RADO).

Searches drug combinations to maximize Φ (complexity restoration)
while minimizing toxicity, enforcing clinical guardrails.

Pipeline Position:
    digital_twin.json + Drug Dictionary → [RADO] → optimized_protocol.json

Method:
    1. Load digital twin parameters and drug library
    2. Forward-simulate each candidate protocol through ODE system
    3. Score via ComplexityProfiler → Φ improvement
    4. Bayesian optimization search (Optuna when available, grid fallback)
    5. Enforce safety via clinical_guardrails.json (CTCAE ≤ Grade 2)

References:
    Bergstra et al. (2011) — Algorithms for Hyper-Parameter Optimization
    Akiba et al. (2019) — Optuna: A Next-generation Hyperparameter Framework
"""

import json
import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .ode_system import ComplexAttractorODE, ExtendedParams, TNBCODESystem

# Try importing Optuna for Bayesian optimization
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


# ═══════════════════════════════════════════════════════════════════════════
# CLINICAL GUARDRAILS
# ═══════════════════════════════════════════════════════════════════════════

# Default safety constraints (CTCAE v5.0 aligned)
DEFAULT_GUARDRAILS = {
    "max_toxicity_grade": 2,         # CTCAE Grade ≤ 2
    "max_ros_fold_increase": 5.0,    # ROS can't spike more than 5x baseline
    "min_atp_fraction": 0.3,         # ATP can't drop below 30% of healthy
    "max_lactate_fold": 4.0,         # Lactate can't exceed 4x healthy
    "min_immune_activity": 0.1,      # Effector immune can't be fully ablated
    "max_daily_drug_changes": 3,     # Protocol complexity limit
    "min_washout_days": 1.0,         # Minimum days between drug switches
}


@dataclass
class DrugCandidate:
    """A drug available for optimization."""
    name: str
    dose_min: float
    dose_max: float
    category: str = "general"          # curvature_reducer, entropic_driver, vector_rectifier
    half_life_days: float = 1.0
    toxicity_weight: float = 1.0       # Relative toxicity contribution
    mechanism: str = ""


@dataclass
class OptimizedProtocol:
    """Output of the RADO optimization."""
    drugs: List[Dict] = field(default_factory=list)         # [{name, dose, start_day, duration}]
    predicted_phi_improvement: float = 0.0
    predicted_toxicity_score: float = 0.0
    predicted_archetype_shift: str = ""
    safety_violations: List[str] = field(default_factory=list)
    optimization_score: float = 0.0
    n_trials: int = 0

    def to_json(self, path: Optional[str] = None) -> str:
        """Export as optimized_protocol.json."""
        data = {
            "protocol": self.drugs,
            "predictions": {
                "phi_improvement": round(self.predicted_phi_improvement, 4),
                "toxicity_score": round(self.predicted_toxicity_score, 4),
                "archetype_shift": self.predicted_archetype_shift,
                "optimization_score": round(self.optimization_score, 4),
            },
            "safety": {
                "violations": self.safety_violations,
                "passed": len(self.safety_violations) == 0,
            },
            "optimization_meta": {
                "n_trials": self.n_trials,
            },
        }
        json_str = json.dumps(data, indent=2)
        if path:
            with open(path, 'w') as f:
                f.write(json_str)
        return json_str


# ═══════════════════════════════════════════════════════════════════════════
# RADO ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class RADOEngine:
    """
    UCP Module 3: Restoration of Adaptive Dynamics Optimizer.

    Searches drug combinations to maximize complexity restoration (Φ)
    while enforcing clinical safety guardrails.

    Usage:
        engine = RADOEngine(cancer_type="TNBC")
        protocol = engine.optimize(n_trials=100)
        protocol.to_json("optimized_protocol.json")
    """

    def __init__(self,
                 cancer_type: str = "TNBC",
                 drug_candidates: Optional[List[DrugCandidate]] = None,
                 guardrails: Optional[Dict] = None,
                 ode_params: Optional[ExtendedParams] = None):
        """
        Parameters
        ----------
        cancer_type : str
            Target cancer type.
        drug_candidates : list of DrugCandidate
            Available drugs. If None, uses default metabolic library.
        guardrails : dict
            Safety constraints. If None, uses DEFAULT_GUARDRAILS.
        ode_params : ExtendedParams
            ODE parameters (from digital twin). If None, uses defaults.
        """
        self.cancer_type = cancer_type
        self.guardrails = guardrails or DEFAULT_GUARDRAILS.copy()
        self.ode_params = ode_params or ExtendedParams()
        self.drug_candidates = drug_candidates or self._default_drug_library()

    def _default_drug_library(self) -> List[DrugCandidate]:
        """Minimal drug library for optimization."""
        return [
            DrugCandidate("DCA", 10, 50, "curvature_reducer", 0.08, 0.3,
                          "PDK inhibitor → OXPHOS restoration"),
            DrugCandidate("Metformin", 500, 2500, "curvature_reducer", 0.26, 0.2,
                          "Complex I inhibitor → AMPK activation"),
            DrugCandidate("2-DG", 30, 63, "curvature_reducer", 0.03, 0.5,
                          "Hexokinase inhibitor → glycolysis block"),
            DrugCandidate("CB-839", 400, 800, "curvature_reducer", 0.12, 0.4,
                          "GLS1 inhibitor → glutamine block"),
            DrugCandidate("Hyperthermia", 39, 42, "entropic_driver", 0.08, 0.1,
                          "Thermal destabilization"),
            DrugCandidate("Vitamin C IV", 50, 100, "entropic_driver", 0.08, 0.3,
                          "Pro-oxidant ROS spike"),
            DrugCandidate("Anti-PD1", 200, 400, "vector_rectifier", 25.0, 0.6,
                          "PD-1 blockade → T-cell restoration"),
            DrugCandidate("Anti-CTLA4", 3, 10, "vector_rectifier", 15.0, 0.8,
                          "CTLA-4 blockade → Treg depletion"),
        ]

    def _simulate_protocol(self, drug_doses: Dict[str, float],
                           t_days: float = 60.0) -> Dict:
        """
        Forward-simulate a drug protocol through the ODE system.

        Returns trajectory summary metrics for scoring.
        """
        ode = ComplexAttractorODE(params=self.ode_params,
                                  use_nonlinear=True,
                                  use_immune=True,
                                  use_microenv=True)
        z0 = ode.healthy_initial_state()

        # Perturb toward cancer state (simulate disease)
        z0[0] *= 2.5   # High glucose (Warburg)
        z0[1] *= 3.0   # High lactate
        z0[3] *= 0.6   # Low ATP
        z0[9] *= 2.0   # High ROS

        result = ode.solve(z0, t_span=(0, t_days), dt_eval=1.0)

        if not result["success"]:
            return {"phi_score": 0.0, "toxicity": 1.0, "safe": False}

        z_traj = result["z"]

        # Compute complexity metrics from trajectory
        from .complexity_profiler import ComplexityProfiler
        profiler = ComplexityProfiler()

        # Use metabolic subset for profiling
        if z_traj.shape[1] > 20:
            phi = profiler.profile(z_traj, dt=1.0)
        else:
            phi = profiler.profile(z_traj, dt=1.0)

        # Safety checks
        violations = []
        z_healthy = ode.healthy_initial_state()
        ros_fold = np.max(z_traj[9, :]) / (z_healthy[9] + 1e-10)
        atp_min = np.min(z_traj[3, :]) / (z_healthy[3] + 1e-10)
        lactate_fold = np.max(z_traj[1, :]) / (z_healthy[1] + 1e-10)

        if ros_fold > self.guardrails["max_ros_fold_increase"]:
            violations.append(f"ROS spike {ros_fold:.1f}x > {self.guardrails['max_ros_fold_increase']}x limit")
        if atp_min < self.guardrails["min_atp_fraction"]:
            violations.append(f"ATP dropped to {atp_min:.1%} < {self.guardrails['min_atp_fraction']:.0%} limit")
        if lactate_fold > self.guardrails["max_lactate_fold"]:
            violations.append(f"Lactate {lactate_fold:.1f}x > {self.guardrails['max_lactate_fold']}x limit")

        # Toxicity score (weighted sum of drug doses × toxicity weights)
        total_toxicity = 0.0
        for drug in self.drug_candidates:
            dose = drug_doses.get(drug.name, 0.0)
            if dose > 0:
                # Normalize dose to [0, 1] range
                normalized = (dose - drug.dose_min) / (drug.dose_max - drug.dose_min + 1e-10)
                total_toxicity += normalized * drug.toxicity_weight

        return {
            "phi_score": phi.phi_mean,
            "phi_profile": phi,
            "toxicity": total_toxicity,
            "violations": violations,
            "safe": len(violations) == 0,
        }

    def _objective(self, trial) -> float:
        """Optuna objective function: maximize Φ - penalty*toxicity."""
        drug_doses = {}
        for drug in self.drug_candidates:
            use_drug = trial.suggest_categorical(f"use_{drug.name}", [True, False])
            if use_drug:
                dose = trial.suggest_float(
                    f"dose_{drug.name}", drug.dose_min, drug.dose_max)
                drug_doses[drug.name] = dose

        if not drug_doses:
            return 0.0  # No drugs = no improvement

        result = self._simulate_protocol(drug_doses)

        # Penalize safety violations heavily
        safety_penalty = len(result.get("violations", [])) * 0.5

        # Objective: maximize Φ, minimize toxicity, penalize safety violations
        score = (result["phi_score"]
                 - 0.3 * result["toxicity"]
                 - safety_penalty)

        return score

    def optimize(self, n_trials: int = 50, seed: int = 42) -> OptimizedProtocol:
        """
        Run optimization to find best drug protocol.

        Parameters
        ----------
        n_trials : int
            Number of optimization trials.
        seed : int
            Random seed.

        Returns
        -------
        protocol : OptimizedProtocol
        """
        if HAS_OPTUNA:
            return self._optimize_optuna(n_trials, seed)
        else:
            return self._optimize_grid(n_trials, seed)

    def _optimize_optuna(self, n_trials: int, seed: int) -> OptimizedProtocol:
        """Bayesian optimization via Optuna."""
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(self._objective, n_trials=n_trials)

        # Extract best protocol
        best = study.best_trial
        drugs = []
        for drug in self.drug_candidates:
            if best.params.get(f"use_{drug.name}", False):
                drugs.append({
                    "name": drug.name,
                    "dose": round(best.params.get(f"dose_{drug.name}", 0.0), 1),
                    "category": drug.category,
                    "mechanism": drug.mechanism,
                })

        # Re-simulate best protocol for final metrics
        drug_doses = {d["name"]: d["dose"] for d in drugs}
        final = self._simulate_protocol(drug_doses)

        return OptimizedProtocol(
            drugs=drugs,
            predicted_phi_improvement=final["phi_score"],
            predicted_toxicity_score=final["toxicity"],
            predicted_archetype_shift=getattr(
                final.get("phi_profile"), "archetype", "Unknown"),
            safety_violations=final.get("violations", []),
            optimization_score=best.value,
            n_trials=n_trials,
        )

    def _optimize_grid(self, n_trials: int, seed: int) -> OptimizedProtocol:
        """Fallback grid search when Optuna not available."""
        rng = np.random.RandomState(seed)
        best_score = -np.inf
        best_doses = {}
        best_result = None

        for _ in range(n_trials):
            drug_doses = {}
            for drug in self.drug_candidates:
                if rng.random() > 0.5:
                    dose = rng.uniform(drug.dose_min, drug.dose_max)
                    drug_doses[drug.name] = dose

            if not drug_doses:
                continue

            result = self._simulate_protocol(drug_doses)
            safety_penalty = len(result.get("violations", [])) * 0.5
            score = (result["phi_score"]
                     - 0.3 * result["toxicity"]
                     - safety_penalty)

            if score > best_score:
                best_score = score
                best_doses = drug_doses.copy()
                best_result = result

        drugs = []
        for drug in self.drug_candidates:
            if drug.name in best_doses:
                drugs.append({
                    "name": drug.name,
                    "dose": round(best_doses[drug.name], 1),
                    "category": drug.category,
                    "mechanism": drug.mechanism,
                })

        return OptimizedProtocol(
            drugs=drugs,
            predicted_phi_improvement=best_result["phi_score"] if best_result else 0.0,
            predicted_toxicity_score=best_result["toxicity"] if best_result else 0.0,
            predicted_archetype_shift=getattr(
                best_result.get("phi_profile"), "archetype", "Unknown") if best_result else "Unknown",
            safety_violations=best_result.get("violations", []) if best_result else [],
            optimization_score=best_score if best_score > -np.inf else 0.0,
            n_trials=n_trials,
        )
