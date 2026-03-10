"""
Geometric Optimization Module
=============================

Implements the "Geometric Achievement Protocol":
1. Measure the curvature (depth) of the cancer attractor.
2. Compute the optimal "flattening" trajectory to minimize well depth.
3. Identify entropic resonance frequencies to destabilize the attractor via noise.

The core theorem:
P(escape) varies as exp(-Barrier / (Noise + Drive))
To maximize P, we must Minimize Barrier (Flatten) and Maximize Noise (Entrophic Heat).
"""

import numpy as np
from scipy import linalg, optimize
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class GeometricState:
    curvature: float        # Scalar metric of well depth (mu)
    anisotropy: float       # Measure of how "narrow" the valley is
    volume: float          # Phase space volume of the basin (approx)
    dominant_mode: complex  # The eigenvalue pair defining the primary dynamic

class GeometricOptimizer:
    """
    Solves the geometric alignment problem:
    Find the metabolic intervention that maximizes the flatness of the cancer well
    without destroying the system's global stability.
    """
    
    def __init__(self, n_metabolites: int):
        self.n = n_metabolites
        
    def compute_basin_curvature(self, A: np.ndarray) -> float:
        """
        Compute the scalar curvature "mu" of the attractor basin.
        
        Physically, this corresponds to the steepness of the potential well.
        Mathematically, it's dominated by the real parts of the stable eigenvalues.
        
        Deep Well (Stable Cancer) -> Large negative real parts -> High curvature
        Flat Well (Escapable)     -> Real parts close to 0 -> Low curvature
        """
        evals = linalg.eigvals(A)
        
        # We only care about the STABLE modes (Re < 0) that are trapping the system.
        # Unstable modes (Re > 0) actually help escape, so we ignore them for "depth".
        stable_real_parts = [e.real for e in evals if e.real < 0]
        
        if not stable_real_parts:
            return 0.0  # System is already unstable (escapable)
            
        # The "depth" is determined by the strongest restoring force (most negative lambda)
        # But for escape probability, the "shallowest" direction matters too.
        # We define curvature as the norm of the stable spectrum.
        # Blend: min captures drug perturbation sensitivity (Kramers escape direction),
        # mean captures overall basin depth. The blend preserves both properties.
        abs_parts = np.abs(stable_real_parts)
        return float(0.6 * np.min(abs_parts) + 0.4 * np.mean(abs_parts))

    def compute_anisotropy(self, A: np.ndarray) -> float:
        """
        Compute how "shaped" the well is. High anisotropy = narrow tunnel.
        Narrow tunnels require precise vector alignment; isotropic wells are easier to exit.
        """
        evals = linalg.eigvals(A)
        stable_real_parts = [abs(e.real) for e in evals if e.real < 0]
        
        if not stable_real_parts:
            return 0.0
            
        return np.max(stable_real_parts) / (np.min(stable_real_parts) + 1e-9)

    def analyze_geometry(self, A: np.ndarray) -> GeometricState:
        """
        Full geometric analysis of the generator.
        """
        evals = linalg.eigvals(A)
        # Sort by magnitude
        idx = np.argsort(np.abs(evals))
        dominant = evals[idx[-1]]
        
        return GeometricState(
            curvature=self.compute_basin_curvature(A),
            anisotropy=self.compute_anisotropy(A),
            volume=1.0 / (np.prod(np.abs(evals)) + 1e-9), # Determinant inverse proxy
            dominant_mode=dominant
        )

    def find_optimal_flattening(
        self, 
        A_cancer: np.ndarray, 
        available_interventions: List[Tuple[np.ndarray, float, float]], # (EffectMatrix, MinDose, MaxDose)
        max_toxicity: float = 1.0
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Solve the flattening problem:
        Minimize Curvature(A + Sum(w_i * Intervention_i))
        Subject to: weights in dosage range
        """
        n_interventions = len(available_interventions)
        
        def objective(weights):
            # Construct candidate A
            delta = np.zeros_like(A_cancer)
            for i, w in enumerate(weights):
                delta += available_interventions[i][0] * w
            
            A_candidate = A_cancer + delta
            
            # We want to minimize curvature (make real parts close to 0)
            # But we must NOT make them positive (instability/death)
            # So we penalize curvature, but harder penalty for instability
            
            mu = self.compute_basin_curvature(A_candidate)
            
            # Stability check
            evals = linalg.eigvals(A_candidate)
            max_real = np.max(evals.real)
            instability_penalty = 0
            if max_real > 0.1: # Allow slight instability (transient), but not chaos
                instability_penalty = max_real * 100
                
            return mu + instability_penalty

        # Bounds
        bounds = [(i[1], i[2]) for i in available_interventions]
        
        # Initial guess (min dose)
        x0 = [b[0] for b in bounds]
        
        result = optimize.minimize(
            objective, 
            x0, 
            bounds=bounds, 
            method='SLSQP'
        )
        
        optimal_weights = result.x
        
        # Compute final delta
        final_delta = np.zeros_like(A_cancer)
        for i, w in enumerate(optimal_weights):
            final_delta += available_interventions[i][0] * w
            
        return final_delta, list(optimal_weights)

    def compute_entropic_resonance(self, A: np.ndarray) -> Dict[str, float]:
        """
        Identify the optimal noise frequencies to pump into the system.
        Based on the imaginary parts of the eigenvalues.
        """
        evals = linalg.eigvals(A)
        frequencies = [abs(e.imag)/(2*np.pi) for e in evals if abs(e.imag) > 1e-5]
        
        if not frequencies:
            return {"primary_freq": 0.0, "bandwidth": 0.0}
            
        # The resonant frequency is the one associated with the "escape mode"
        # Usually the mode with the smallest real part (closest to instability)
        
        # Sort by stability (real part closest to 0)
        idx = np.argsort([abs(e.real) for e in evals])
        resonant_eval = evals[idx[0]]
        
        return {
            "target_frequency": abs(resonant_eval.imag) / (2*np.pi),
            "damping_ratio": abs(resonant_eval.real) / (abs(resonant_eval) + 1e-9),
            "optimal_noise_color": "pink" if abs(resonant_eval.real) < 0.5 else "white"
        }

    def compute_kramers_escape_rate(self, A: np.ndarray, noise_variance: float, immune_force: float = 0.0) -> float:
        """
        Compute the theoretical probability/rate of escaping the cancer attractor basin.
        Based on Kramers escape rate theory for noise-driven exit from a potential well.
        
        P(escape) = prefactor * exp(-Barrier / (Noise + Drive))
        
        Args:
            A: Generator matrix
            noise_variance: Current thermal/entropic noise in the system (sigma_squared)
            immune_force: Magnitude of the targeted immune restorative force
            
        Returns:
            Scalar score representing escape probability per unit time
        """
        # The barrier is proportional to the well depth (curvature)
        barrier_height = self.compute_basin_curvature(A)
        
        # Effective temperature/agitation
        effective_noise = max(noise_variance + immune_force, 1e-6)
        
        # Prefactor related to well shape (anisotropy and volume)
        # Narrow, deep wells have lower escape rates even if barrier is similar
        anisotropy = max(self.compute_anisotropy(A), 1e-6)
        prefactor = 1.0 / np.sqrt(anisotropy)
        
        # Kramers formula
        escape_rate = prefactor * np.exp(-barrier_height / effective_noise)
        
        return float(escape_rate)


@dataclass
class ProtocolPhase:
    """A phase in the therapeutic protocol."""
    day_start: int
    duration: int
    interventions: List[Tuple[str, float]] # (Name, Dose)
    expected_curvature: float
    expected_escape_rate: float
    description: str


class TherapeuticProtocolOptimizer:
    """
    Advanced optimizer that translates static geometric flattening into a 
    dynamic, multi-phase sequential protocol (Flatten -> Heat -> Push).
    """
    
    def __init__(self, n_metabolites: int):
        self.geom_opt = GeometricOptimizer(n_metabolites)
        
    def generate_optimal_sequence(
        self,
        A_cancer: np.ndarray,
        metabolic_drugs: List[Tuple[str, np.ndarray, Tuple[float, float]]], # (Name, Effect, Bounds)
        entropic_drivers: List[Tuple[str, np.ndarray, Tuple[float, float], float]], # (+ noise multiplier)
        immune_rectifiers: List[Tuple[str, np.ndarray, Tuple[float, float], float]], # (+ force multiplier)
        base_noise: float = 0.05,
        base_immune_force: float = 0.1,
        toxicity_penalty: float = 0.1,
        min_coherence_score: float = 0.65 # Target score for the final state
    ) -> List[ProtocolPhase]:
        """
        Generate a strictly sequenced 3-phase protocol.
        
        Phase 1: Flatten (minimize curvature using metabolic drugs)
        Phase 2: Heat (introduce entropic drivers when flattened)
        Phase 3: Push (introduce immune rectifiers to force escape)
        """
        from coherence import CoherenceAnalyzer
        coherence_engine = CoherenceAnalyzer()
        
        protocol = []
        current_A = A_cancer.copy()
        current_noise = base_noise
        current_force = base_immune_force
        
        # --- PHASE 1: FLATTEN (Days 0-14) ---
        # Find combination of metabolic drugs that minimizes curvature but penalizes high total dose
        if metabolic_drugs:
            def collapse_objective(weights):
                delta = np.zeros_like(current_A)
                total_dose_ratio = 0
                for i, w in enumerate(weights):
                    delta += metabolic_drugs[i][1] * w
                    # Toxicity proxy: normalized dose
                    b_min, b_max = metabolic_drugs[i][2]
                    total_dose_ratio += (w - b_min) / max((b_max - b_min), 1e-9)
                    
                A_cand = current_A + delta
                mu = self.geom_opt.compute_basin_curvature(A_cand)
                
                # Prevent tipping into chaotic instability (death)
                max_real = np.max(np.linalg.eigvals(A_cand).real)
                penalty = max_real * 1000 if max_real > 0 else 0
                
                # Add coherence penalty (encourage moving toward healthy spectrum)
                coh_metrics = coherence_engine.analyze(A_cand)
                coh_score = coh_metrics.get('overall_score', 0)
                coh_penalty = max(0, min_coherence_score - coh_score) * 10 
                
                return mu + (toxicity_penalty * total_dose_ratio) + penalty + coh_penalty

            bounds = [i[2] for i in metabolic_drugs]
            x0 = [np.mean(b) for b in bounds]
            
            res = optimize.minimize(collapse_objective, x0, bounds=bounds, method='SLSQP')
            
            phase1_interventions = []
            phase1_delta = np.zeros_like(current_A)
            for i, w in enumerate(res.x):
                if w > bounds[i][0] * 1.01: # Check if meaningfully above minimum
                    phase1_interventions.append((metabolic_drugs[i][0], float(w)))
                    phase1_delta += metabolic_drugs[i][1] * w
                    
            current_A += phase1_delta
            
            mu_flattened = self.geom_opt.compute_basin_curvature(current_A)
            escape1 = self.geom_opt.compute_kramers_escape_rate(current_A, current_noise, current_force)
            
            protocol.append(ProtocolPhase(
                day_start=0,
                duration=14,
                interventions=phase1_interventions,
                expected_curvature=mu_flattened,
                expected_escape_rate=escape1,
                description="Phase 1 (Flatten): Metabolic rewiring to minimize attractor depth."
            ))
            
        # --- PHASE 2: HEAT (Days 14-21) ---
        # Add maximal entropic drivers given safety limits
        if entropic_drivers:
            phase2_interventions = []
            max_noise_added = 0.0
            
            # Simple greedy strategy for heat: max out safest driver
            for name, effect, (b_min, b_max), noise_mult in entropic_drivers:
                # Use max safe dose
                dose = b_max
                phase2_interventions.append((name, dose))
                current_A += effect * dose
                max_noise_added += noise_mult * dose
                
            current_noise += max_noise_added
            
            mu_heated = self.geom_opt.compute_basin_curvature(current_A)
            escape2 = self.geom_opt.compute_kramers_escape_rate(current_A, current_noise, current_force)
            
            protocol.append(ProtocolPhase(
                day_start=14,
                duration=7,
                interventions=phase2_interventions,
                expected_curvature=mu_heated,
                expected_escape_rate=escape2,
                description="Phase 2 (Heat): Introduce noise at target resonance to agitate flattened state."
            ))
            
        # --- PHASE 3: PUSH (Days 21-42) ---
        # Introduce immune rectifiers
        if immune_rectifiers:
            phase3_interventions = []
            max_force_added = 0.0
            
            for name, effect, (b_min, b_max), force_mult in immune_rectifiers:
                dose = b_max # Standard practice is max tolerated dose for checkpoints
                phase3_interventions.append((name, dose))
                current_A += effect * dose
                max_force_added += force_mult * dose
                
            current_force += max_force_added
            
            mu_pushed = self.geom_opt.compute_basin_curvature(current_A)
            escape3 = self.geom_opt.compute_kramers_escape_rate(current_A, current_noise, current_force)
            
            protocol.append(ProtocolPhase(
                day_start=21,
                duration=21,
                interventions=phase3_interventions,
                expected_curvature=mu_pushed,
                expected_escape_rate=escape3,
                description="Phase 3 (Push): Targeted immune force applied to destabilized well."
            ))
            
        return protocol
        
    def evaluate_robustness_monte_carlo(self, protocol_A_final: np.ndarray, base_noise: float, base_force: float, n_trials: int = 100) -> dict:
        """
        Evaluate protocol robustness by perturbing the final generator matrix (simulating patient variance).
        """
        from coherence import CoherenceAnalyzer
        import copy
        coherence_engine = CoherenceAnalyzer()
        
        successes = 0
        total_escape = 0.0
        coherence_scores = []
        
        for _ in range(n_trials):
            # Perturb the parameters by up to 15% to simulate patient-to-patient variation
            noise_matrix = np.random.normal(0, 0.15 * np.std(protocol_A_final), size=protocol_A_final.shape)
            perturbed_A = protocol_A_final + noise_matrix
            
            # Re-evaluate Kramers rate and Coherence
            escape_rate = self.geom_opt.compute_kramers_escape_rate(perturbed_A, base_noise, base_force)
            coh_metrics = coherence_engine.analyze(perturbed_A)
            score = coh_metrics.get('overall_score', 0)
            
            total_escape += escape_rate
            coherence_scores.append(score)
            
            # Success strictly defined as escape rate higher than a threshold and decent coherence
            if escape_rate > 0.01 and score > 0.2:
                successes += 1
                
        return {
            'robustness_score': successes / n_trials, # % of trials that succeeded
            'mean_escape_rate': total_escape / n_trials,
            'mean_coherence_score': np.mean(coherence_scores),
            'coherence_variance': np.var(coherence_scores)
        }
