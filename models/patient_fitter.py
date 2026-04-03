"""
Patient Fitter — Project Confluence (UCP Module 2)
====================================================

Bayesian inference engine for creating personalized Digital Twins.
Fits the 15D SAEM ODE model parameters (θ) to individual patient data
via MCMC sampling.

Pipeline Position:
    phi_profile.json + Clinical Data → [PatientFitter] → digital_twin.json

Method:
    1. Define parameter priors from literature bounds
    2. Likelihood: Gaussian comparison of simulated vs. observed metrics
    3. MCMC sampling via emcee (ensemble sampler)
    4. Output: posterior distributions, MAP estimate, credible intervals

References:
    Goodman & Weare (2010) — Ensemble samplers with affine invariance
    Raue et al. (2009) — Structural and practical identifiability
"""

import json
import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from .ode_system import ODEParams, TNBCODESystem
from .neural_ode import ComplexityNeuralODE, TORCHDIFFEQ_AVAILABLE
from agents.digital_twin_memory import DigitalTwinMemory

# Try importing PyTorch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Try importing optional deps with graceful fallback
try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False
    warnings.warn("emcee not installed. Run: pip install emcee>=3.1.0")

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False


# ═══════════════════════════════════════════════════════════════════════════
# PARAMETER SPACE DEFINITION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ParameterBounds:
    """Prior bounds for each ODE parameter (biologically plausible ranges)."""
    name: str
    lower: float
    upper: float
    display_name: str = ""
    unit: str = ""

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name


# Default parameter bounds (literature-derived)
DEFAULT_PARAM_BOUNDS = [
    ParameterBounds("glucose_uptake",     -1.50, -0.10, "Glucose uptake rate",      "day⁻¹"),
    ParameterBounds("glycolysis_flux",     0.05,  1.00, "Glycolysis flux",           "day⁻¹"),
    ParameterBounds("lactate_clearance",  -2.00, -0.10, "Lactate clearance rate",    "day⁻¹"),
    ParameterBounds("pyruvate_to_lactate", 0.01,  0.50, "Pyruvate→Lactate flux",     "day⁻¹"),
    ParameterBounds("pyruvate_to_atp",     0.05,  0.80, "Pyruvate→ATP (OXPHOS)",     "day⁻¹"),
    ParameterBounds("nadh_to_atp",         0.05,  0.60, "NADH→ATP coupling",         "day⁻¹"),
    ParameterBounds("atp_turnover",       -0.80, -0.05, "ATP turnover rate",         "day⁻¹"),
    ParameterBounds("nadh_cycling",       -0.80, -0.05, "NADH cycling rate",         "day⁻¹"),
    ParameterBounds("glutamine_utilization", -1.20, -0.05, "Glutamine utilization",  "day⁻¹"),
    ParameterBounds("glutaminolysis",      0.05,  0.80, "Glutaminolysis flux",       "day⁻¹"),
    ParameterBounds("glutamate_to_akg",    0.05,  0.80, "Glutamate→αKG flux",       "day⁻¹"),
    ParameterBounds("akg_to_citrate",      0.05,  0.60, "αKG→Citrate flux",         "day⁻¹"),
    ParameterBounds("ros_clearance",      -2.00, -0.20, "ROS clearance rate",        "day⁻¹"),
    ParameterBounds("nadh_ros_leak",       0.01,  0.40, "NADH→ROS leak rate",        "day⁻¹"),
    ParameterBounds("ros_atp_damage",     -0.40, -0.01, "ROS→ATP damage rate",       "day⁻¹"),
    ParameterBounds("atp_inhibits_glucose",-0.40, -0.01, "ATP⊣Glucose feedback",    "day⁻¹"),
]


# ═══════════════════════════════════════════════════════════════════════════
# CALIBRATION TARGETS (CCLE-derived basin curvatures)
# ═══════════════════════════════════════════════════════════════════════════

CCLE_CURVATURE_TARGETS = {
    "TNBC":     {"curvature": 2.5,  "sigma": 0.5,  "evidence": "CCLE MDA-MB-231"},
    "PDAC":     {"curvature": 4.2,  "sigma": 0.8,  "evidence": "CCLE PANC-1"},
    "NSCLC":    {"curvature": 1.8,  "sigma": 0.4,  "evidence": "CCLE A549"},
    "Melanoma": {"curvature": 1.3,  "sigma": 0.3,  "evidence": "CCLE A375"},
    "GBM":      {"curvature": 3.0,  "sigma": 0.6,  "evidence": "CCLE U87MG"},
    "CRC":      {"curvature": 2.0,  "sigma": 0.5,  "evidence": "CCLE HCT116"},
    "HGSOC":    {"curvature": 2.8,  "sigma": 0.6,  "evidence": "CCLE SKOV3"},
    "mCRPC":    {"curvature": 2.2,  "sigma": 0.5,  "evidence": "CCLE LNCaP"},
    "AML":      {"curvature": 2.6,  "sigma": 0.5,  "evidence": "CCLE HL-60"},
    "HCC":      {"curvature": 3.5,  "sigma": 0.7,  "evidence": "CCLE HepG2"},
}


# ═══════════════════════════════════════════════════════════════════════════
# CALIBRATION RESULT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DigitalTwin:
    """
    Personalized digital twin output from patient fitting.

    Contains the MAP estimate, posterior distributions, and diagnostics.
    """
    patient_id: str = "anonymous"
    cancer_type: str = "Unknown"
    samples: np.ndarray = field(default_factory=lambda: np.array([]))
    log_probs: np.ndarray = field(default_factory=lambda: np.array([]))
    param_names: List[str] = field(default_factory=list)
    param_bounds: List[ParameterBounds] = field(default_factory=list)
    acceptance_fraction: float = 0.0
    n_walkers: int = 0
    n_steps: int = 0

    # Computed summaries
    map_estimate: Optional[np.ndarray] = None
    median_estimate: Optional[np.ndarray] = None
    credible_intervals: Optional[List[Tuple[float, float]]] = None

    def compute_summary(self):
        """Compute MAP, median, and 95% credible intervals."""
        if len(self.samples) == 0:
            return
        self.map_estimate = self.samples[np.argmax(self.log_probs)]
        self.median_estimate = np.median(self.samples, axis=0)
        self.credible_intervals = [
            (float(np.percentile(self.samples[:, i], 2.5)),
             float(np.percentile(self.samples[:, i], 97.5)))
            for i in range(self.samples.shape[1])
        ]

    def add_trajectory(self, trajectory: np.ndarray, timepoints: np.ndarray):
        """Store the continuous trajectory generated by Neural ODE."""
        self.trajectory = trajectory
        self.timepoints = timepoints

    def identifiability_report(self) -> Dict[str, str]:
        """Classify each parameter as identifiable / weakly / sloppy."""
        if self.credible_intervals is None:
            self.compute_summary()
        report = {}
        for i, b in enumerate(self.param_bounds):
            ci_width = self.credible_intervals[i][1] - self.credible_intervals[i][0]
            prior_width = b.upper - b.lower
            ratio = ci_width / (prior_width + 1e-10)
            if ratio < 0.5:
                report[b.name] = "well-identified"
            elif ratio < 0.8:
                report[b.name] = "weakly-identified"
            else:
                report[b.name] = "sloppy"
        return report

    def to_json(self, path: Optional[str] = None) -> str:
        """Export as digital_twin.json."""
        self.compute_summary()
        data = {
            "patient_id": self.patient_id,
            "cancer_type": self.cancer_type,
            "parameters": {
                name: {
                    "map": float(self.map_estimate[i]),
                    "median": float(self.median_estimate[i]),
                    "ci_95": list(self.credible_intervals[i]),
                }
                for i, name in enumerate(self.param_names)
            },
            "diagnostics": {
                "acceptance_fraction": round(self.acceptance_fraction, 4),
                "n_walkers": self.n_walkers,
                "n_steps": self.n_steps,
                "n_samples": len(self.samples),
                "identifiability": self.identifiability_report(),
            },
        }
        json_str = json.dumps(data, indent=2)
        if path:
            with open(path, 'w') as f:
                f.write(json_str)
        return json_str

    def print_summary(self):
        """Print human-readable calibration summary."""
        self.compute_summary()
        print(f"\n{'='*60}")
        print(f"Digital Twin — {self.cancer_type} ({self.patient_id})")
        print(f"{'='*60}")
        print(f"Acceptance fraction: {self.acceptance_fraction:.3f}")
        print(f"Samples: {len(self.samples)}\n")
        report = self.identifiability_report()
        for i, name in enumerate(self.param_names):
            ci_lo, ci_hi = self.credible_intervals[i]
            status = report.get(name, "?")
            print(f"  {name:30s} MAP={self.map_estimate[i]:+.4f}  "
                  f"CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]  [{status}]")


# ═══════════════════════════════════════════════════════════════════════════
# PATIENT FITTER
# ═══════════════════════════════════════════════════════════════════════════

class PatientFitter:
    """
    UCP Module 2: Patient Fitter — Bayesian Digital Twin Creator.

    Fits ODE parameters to individual patient data using MCMC sampling.

    Usage:
        fitter = PatientFitter(cancer_type="TNBC")
        twin = fitter.fit(n_walkers=32, n_steps=2000)
        twin.to_json("digital_twin.json")
    """

    def __init__(self,
                 cancer_type: str = "TNBC",
                 patient_id: str = "anonymous",
                 param_bounds: Optional[List[ParameterBounds]] = None,
                 targets: Optional[Dict[str, Dict]] = None,
                 inference_mode: str = "mcmc",
                 guideline_priors: Optional[Dict] = None):
        self.cancer_type = cancer_type
        self.patient_id = patient_id
        self.param_bounds = param_bounds or DEFAULT_PARAM_BOUNDS
        self.targets = targets or CCLE_CURVATURE_TARGETS
        self.n_params = len(self.param_bounds)
        self.inference_mode = inference_mode
        self.guideline_priors = guideline_priors
        self.memory_controller = DigitalTwinMemory(patient_id=self.patient_id)
        
        # Apply Nigeria-specific guideline priors if provided
        if self.guideline_priors:
            self._apply_guideline_priors()
        
        if self.inference_mode == "neural" and not (HAS_TORCH and TORCHDIFFEQ_AVAILABLE):
             warnings.warn("Neural inference requested but torch/torchdiffeq is not installed. Falling back to MCMC.")
             self.inference_mode = "mcmc"
    
    def _apply_guideline_priors(self):
        """
        Narrow MCMC parameter bounds using Nigeria-specific guideline constraints.
        
        When NSTG 2022 guidelines are provided, certain ODE parameters can be
        constrained to clinically realistic ranges for the Nigerian patient
        population. For example, higher baseline anaemia prevalence in Nigeria
        may inform tighter bounds on ATP-related parameters.
        
        This is called at init and only modifies bounds if guideline_priors
        contains relevant parameter constraints.
        """
        gp = self.guideline_priors
        if not gp:
            return
        
        param_adjustments = gp.get("parameter_adjustments", {})
        for i, bound in enumerate(self.param_bounds):
            if bound.name in param_adjustments:
                adj = param_adjustments[bound.name]
                if "lower" in adj:
                    self.param_bounds[i] = ParameterBounds(
                        name=bound.name,
                        lower=max(bound.lower, adj["lower"]),
                        upper=min(bound.upper, adj.get("upper", bound.upper)),
                        display_name=bound.display_name,
                        unit=bound.unit,
                    )
                    
        # Log guideline prior application
        n_adjusted = sum(
            1 for b in self.param_bounds
            if b.name in param_adjustments
        )
        if n_adjusted > 0:
            print(f"[PatientFitter] Applied Nigeria guideline priors: "
                  f"{n_adjusted} parameters adjusted")

    def _theta_to_params(self, theta: np.ndarray) -> ODEParams:
        """Convert parameter vector to ODEParams object."""
        kwargs = {}
        for i, b in enumerate(self.param_bounds):
            kwargs[b.name] = theta[i]
        return ODEParams(**kwargs)

    def _simulate_curvature(self, theta: np.ndarray) -> float:
        """Run forward model: parameters → basin curvature."""
        try:
            params = self._theta_to_params(theta)
            gen_method = getattr(TNBCODESystem, f"{self.cancer_type.lower()}_generator",
                                 TNBCODESystem.tnbc_generator)
            A_cancer = gen_method(params)
            eigs = np.linalg.eigvals(A_cancer)
            curvature = float(np.sum(np.real(eigs) ** 2))
            return curvature
        except Exception:
            return np.inf

    def log_prior(self, theta: np.ndarray) -> float:
        """Uniform prior: 0 inside bounds, -inf outside."""
        for i, b in enumerate(self.param_bounds):
            if theta[i] < b.lower or theta[i] > b.upper:
                return -np.inf
        return 0.0

    def log_likelihood(self, theta: np.ndarray) -> float:
        """Gaussian likelihood comparing simulated to target curvatures."""
        target = self.targets.get(self.cancer_type)
        if target is None:
            return 0.0
        curvature = self._simulate_curvature(theta)
        if not np.isfinite(curvature):
            return -np.inf
        return -0.5 * ((curvature - target["curvature"]) / target["sigma"]) ** 2

    def log_probability(self, theta: np.ndarray) -> float:
        """Combined log-posterior = log-prior + log-likelihood."""
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def _initialize_walkers(self, n_walkers: int, seed: int) -> np.ndarray:
        """Initialize walkers in a small ball around the prior center."""
        rng = np.random.RandomState(seed)
        center = np.array([
            0.5 * (b.lower + b.upper) for b in self.param_bounds
        ])
        spread = np.array([
            0.05 * (b.upper - b.lower) for b in self.param_bounds
        ])
        return center + spread * rng.randn(n_walkers, self.n_params)

    def fit(self,
            n_walkers: int = 32,
            n_steps: int = 2000,
            n_burnin: int = 1000,
            seed: int = 42,
            progress: bool = True) -> DigitalTwin:
        """
        Run MCMC sampling to create a Digital Twin.

        Parameters
        ----------
        n_walkers : int
            Number of ensemble walkers (must be >= 2 * n_params).
        n_steps : int
            Total MCMC steps (including burn-in).
        n_burnin : int
            Steps to discard as burn-in.
        seed : int
            Random seed.
        progress : bool
            Show progress bar.

        Returns
        -------
        twin : DigitalTwin
            Personalized digital twin with posterior distributions.
        """
        if self.inference_mode == "neural":
             return self.fit_neural(n_steps=n_steps, seed=seed)
             
        if not HAS_EMCEE:
            raise ImportError(
                "emcee is required for MCMC fitting. "
                "Install with: pip install emcee>=3.1.0"
            )

        if n_walkers < 2 * self.n_params:
            n_walkers = 2 * self.n_params + 2
            warnings.warn(f"Adjusted n_walkers to {n_walkers} (>= 2*n_params)")

        p0 = self._initialize_walkers(n_walkers, seed)

        sampler = emcee.EnsembleSampler(
            n_walkers, self.n_params, self.log_probability
        )
        sampler.run_mcmc(p0, n_steps, progress=progress)

        # Extract post-burn-in samples
        samples = sampler.get_chain(discard=n_burnin, flat=True)
        log_probs = sampler.get_log_prob(discard=n_burnin, flat=True)

        twin = DigitalTwin(
            patient_id=self.patient_id,
            cancer_type=self.cancer_type,
            samples=samples,
            log_probs=log_probs,
            param_names=[b.name for b in self.param_bounds],
            param_bounds=self.param_bounds,
            acceptance_fraction=float(np.mean(sampler.acceptance_fraction)),
            n_walkers=n_walkers,
            n_steps=n_steps,
        )
        twin.compute_summary()

        return twin

    def fit_neural(self, 
                   n_steps: int = 200, 
                   seed: int = 42,
                   clinical_time_series: Optional[np.ndarray] = None) -> DigitalTwin:
        """
        Run continuous-time Neural ODE inference to trace the digital twin trajectory.
        Stores the discrete episodes directly into Graphiti/Cognee.
        """
        torch.manual_seed(seed)
        
        # In a real scenario, model would be loaded. Here we initialize a stub.
        neural_ode = ComplexityNeuralODE()
        neural_ode.eval()
        
        # Stub clinical data [batch=1, seq=5, obs=15]
        if clinical_time_series is None:
             clinical_time_series = torch.ones((1, 5, 15)) * 0.1
        else:
             clinical_time_series = torch.tensor(clinical_time_series, dtype=torch.float32).unsqueeze(0)
             
        t_span = torch.linspace(0, n_steps, steps=n_steps)
        
        with torch.no_grad():
             trajectory_tensor = neural_ode(clinical_time_series, t_span)
             
        trajectory_np = trajectory_tensor.squeeze(0).numpy()
        time_np = t_span.numpy()
        
        # Route to Digital Twin Memory Layer (Graphiti + Cognee)
        memory_stats = self.memory_controller.process_neural_trajectory(trajectory_np, time_np)
        
        twin = DigitalTwin(
            patient_id=self.patient_id,
            cancer_type=self.cancer_type,
            n_steps=n_steps,
        )
        
        # Instead of storing raw params, we store the full continuous trajectory
        twin.add_trajectory(trajectory_np, time_np)
        print(f"Neural Fitter mapped patient {self.patient_id} into Memory Layer:")
        print(f"  Graphiti Temp Events: {memory_stats['temporal_events_recorded']}")
        print(f"  Cognee Structural Edges: {memory_stats['unique_correlations_mapped']}")
        
        return twin

    def run_profile_likelihood(self, param_index: int,
                               n_points: int = 30,
                               n_inner_evals: int = 50,
                               seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Compute profile likelihood for identifiability diagnosis."""
        rng = np.random.RandomState(seed)
        b = self.param_bounds[param_index]
        grid = np.linspace(b.lower, b.upper, n_points)
        profile_ll = np.zeros(n_points)

        center = np.array([
            0.5 * (pb.lower + pb.upper) for pb in self.param_bounds
        ])

        for k, val in enumerate(grid):
            best_ll = -np.inf
            for _ in range(n_inner_evals):
                theta = center + 0.1 * (
                    np.array([pb.upper - pb.lower for pb in self.param_bounds])
                    * rng.randn(self.n_params)
                )
                theta[param_index] = val
                for i, pb in enumerate(self.param_bounds):
                    theta[i] = np.clip(theta[i], pb.lower, pb.upper)
                ll = self.log_probability(theta)
                if ll > best_ll:
                    best_ll = ll
            profile_ll[k] = best_ll

        return grid, profile_ll
