"""
Complexity Profiler — Project Confluence (UCP Module 1)
=======================================================

Computes the 5-dimensional Dynamical Complexity vector Φ (Phi) from
clinical and trajectory data, per the Unified Complexity Profile (UCP).

Pipeline Position:
    Raw Clinical Data → [ComplexityProfiler] → phi_profile.json

The 5 Dimensions of Φ:
    Φ_temporal      : Variability over time (HRV, glucose variability, MSE)
    Φ_spatial        : Heterogeneity across scale (cell diversity, D₂)
    Φ_functional     : Response to stress (recovery rate, resilience)
    Φ_informational  : Information processing capacity (SampEn, EEG/ECG)
    Φ_coupling       : Inter-system communication (immune-metabolic corr.)

Pathology Archetypes:
    Chaotic/Decoupled  → Low Coupling, High Chaos     → Metastatic Cancer
    Rigid/Locked       → High Coupling, Low Variability → Autoimmunity
    Collapsed/Exhausted → All Dimensions Low            → Cachexia

Hardware target: i5-3380M, 8GB RAM — all algorithms O(N²) or better.

References:
    Costa et al. (2005) — Multiscale entropy analysis of biological signals
    Goldberger et al. (2002) — Fractal dynamics in health and disease
    Kembro et al. (2014) — Loss of complexity in cancer cell populations
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# 1. SAMPLE ENTROPY (foundation for MSE)
# ═══════════════════════════════════════════════════════════════════════════

def sample_entropy(signal: np.ndarray, m: int = 2, r: Optional[float] = None) -> float:
    """
    Compute sample entropy of a 1D signal.

    SampEn measures the probability that similar patterns in a time series
    remain similar at the next point. Lower = more regular/predictable.

    Parameters
    ----------
    signal : ndarray, shape (N,)
        Input time series.
    m : int
        Embedding dimension (template length).
    r : float, optional
        Tolerance radius. Defaults to 0.2 * std(signal).

    Returns
    -------
    sampen : float
        Sample entropy value. Higher = more complex.
    """
    N = len(signal)
    if N < m + 2:
        return 0.0

    if r is None:
        r = 0.2 * np.std(signal)
        if r < 1e-10:
            return 0.0

    def _count_matches(templates):
        count = 0
        n_templates = len(templates)
        for i in range(n_templates):
            for j in range(i + 1, n_templates):
                if np.max(np.abs(templates[i] - templates[j])) < r:
                    count += 1
        return count

    # Build templates of length m and m+1
    templates_m = np.array([signal[i:i + m] for i in range(N - m)])
    templates_m1 = np.array([signal[i:i + m + 1] for i in range(N - m)])

    B = _count_matches(templates_m)
    A = _count_matches(templates_m1)

    if B == 0:
        return 0.0

    return -np.log(A / B) if A > 0 else float(np.log(B))


# ═══════════════════════════════════════════════════════════════════════════
# 2. MULTISCALE ENTROPY (MSE)
# ═══════════════════════════════════════════════════════════════════════════

def coarse_grain(signal: np.ndarray, scale: int) -> np.ndarray:
    """Coarse-grain a signal at a given scale factor."""
    n = len(signal) // scale
    return np.mean(signal[:n * scale].reshape(n, scale), axis=1)


def multiscale_entropy(signal: np.ndarray, max_scale: int = 20,
                       m: int = 2, r_factor: float = 0.2) -> np.ndarray:
    """
    Compute multiscale entropy (MSE) of a 1D signal.

    MSE evaluates sample entropy across multiple time scales via
    coarse-graining. Healthy signals maintain high entropy across scales.

    Parameters
    ----------
    signal : ndarray, shape (N,)
        Input time series.
    max_scale : int
        Maximum coarse-graining scale.
    m : int
        Embedding dimension for sample entropy.
    r_factor : float
        Tolerance factor (r = r_factor * std(signal)).

    Returns
    -------
    mse : ndarray, shape (max_scale,)
        Sample entropy at each scale (1 to max_scale).
    """
    r = r_factor * np.std(signal)
    mse_values = np.zeros(max_scale)

    for scale in range(1, max_scale + 1):
        cg = coarse_grain(signal, scale)
        if len(cg) > m + 2:
            mse_values[scale - 1] = sample_entropy(cg, m=m, r=r)
        else:
            mse_values[scale - 1] = 0.0

    return mse_values


def mse_mean(signal: np.ndarray, max_scale: int = 20, **kwargs) -> float:
    """Compute mean MSE across all scales — single scalar summary."""
    mse = multiscale_entropy(signal, max_scale=max_scale, **kwargs)
    return float(np.mean(mse[mse > 0])) if np.any(mse > 0) else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 3. CORRELATION DIMENSION D₂
# ═══════════════════════════════════════════════════════════════════════════

def time_delay_embedding(signal: np.ndarray, emb_dim: int = 10,
                         tau: int = 1) -> np.ndarray:
    """
    Construct time-delay embedding of a 1D signal.

    Parameters
    ----------
    signal : ndarray, shape (N,)
    emb_dim : int
        Embedding dimension.
    tau : int
        Time delay.

    Returns
    -------
    embedded : ndarray, shape (N - (emb_dim-1)*tau, emb_dim)
    """
    N = len(signal)
    M = N - (emb_dim - 1) * tau
    if M <= 0:
        raise ValueError(f"Signal too short ({N}) for embedding (dim={emb_dim}, tau={tau})")
    embedded = np.zeros((M, emb_dim))
    for d in range(emb_dim):
        embedded[:, d] = signal[d * tau: d * tau + M]
    return embedded


def correlation_dimension(signal: np.ndarray, emb_dim: int = 10,
                          tau: int = 1, n_epsilons: int = 20,
                          theiler_window: int = 10,
                          max_points: int = 1000) -> float:
    """
    Estimate correlation dimension D₂ via Grassberger-Procaccia algorithm.

    Uses distance distribution of embedded phase-space vectors to estimate
    the scaling exponent of the correlation integral C(ε) ~ ε^D₂.

    Parameters
    ----------
    signal : ndarray, shape (N,)
        Input time series (single variable from trajectory).
    emb_dim : int
        Embedding dimension (should be > 2*D₂).
    tau : int
        Time delay for embedding.
    n_epsilons : int
        Number of epsilon values to sample.
    theiler_window : int
        Minimum temporal separation to avoid autocorrelation bias.
    max_points : int
        Max points to use (subsample for speed on large series).

    Returns
    -------
    D2 : float
        Estimated correlation dimension.
    """
    embedded = time_delay_embedding(signal, emb_dim, tau)

    # Subsample if needed for performance
    if len(embedded) > max_points:
        idx = np.linspace(0, len(embedded) - 1, max_points, dtype=int)
        embedded = embedded[idx]

    N = len(embedded)
    if N < 20:
        return 0.0

    # Compute pairwise distances (upper triangle, respecting Theiler window)
    distances = []
    for i in range(N):
        for j in range(i + theiler_window, N):
            d = np.linalg.norm(embedded[i] - embedded[j])
            if d > 0:
                distances.append(d)

    if len(distances) < 10:
        return 0.0

    distances = np.array(distances)
    d_min, d_max = np.percentile(distances, [5, 95])

    # Log-spaced epsilon values
    epsilons = np.logspace(np.log10(d_min + 1e-10), np.log10(d_max), n_epsilons)

    # Correlation integral
    C_eps = np.zeros(n_epsilons)
    n_pairs = len(distances)
    for k, eps in enumerate(epsilons):
        C_eps[k] = np.sum(distances < eps) / n_pairs

    # Remove zeros for log
    mask = C_eps > 0
    if np.sum(mask) < 5:
        return 0.0

    log_eps = np.log(epsilons[mask])
    log_C = np.log(C_eps[mask])

    # Linear regression on the scaling region (middle 60%)
    n_valid = len(log_eps)
    start = n_valid // 5
    end = 4 * n_valid // 5
    if end - start < 3:
        start, end = 0, n_valid

    slope, _ = np.polyfit(log_eps[start:end], log_C[start:end], 1)

    return float(max(0, slope))


# ═══════════════════════════════════════════════════════════════════════════
# 4. LYAPUNOV EXPONENT (Maximum)
# ═══════════════════════════════════════════════════════════════════════════

def largest_lyapunov_exponent(trajectory: np.ndarray, dt: float = 0.5,
                               emb_dim: int = 7, tau: int = 2,
                               min_separation: int = 20,
                               max_iter: int = 500) -> float:
    """
    Estimate the largest Lyapunov exponent from a multivariate trajectory
    using the Rosenstein et al. (1993) algorithm.

    A positive value indicates chaos (sensitivity to initial conditions).
    A near-zero or negative value indicates regularity.

    Parameters
    ----------
    trajectory : ndarray, shape (n_vars, n_timepoints) or (n_timepoints,)
        Time series. If multivariate, uses the first principal component.
    dt : float
        Time step between samples (days).
    emb_dim : int
        Embedding dimension.
    tau : int
        Time delay for embedding.
    min_separation : int
        Minimum temporal separation for nearest-neighbor search.
    max_iter : int
        Maximum iterations for divergence tracking.

    Returns
    -------
    lambda_max : float
        Estimated largest Lyapunov exponent (bits/day).
    """
    # If multivariate, reduce to 1D via first PC
    if trajectory.ndim == 2:
        if trajectory.shape[0] > trajectory.shape[1]:
            trajectory = trajectory.T
        centered = trajectory - trajectory.mean(axis=1, keepdims=True)
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            signal = Vt[0, :]
        except np.linalg.LinAlgError:
            signal = trajectory[0, :]
    else:
        signal = trajectory

    N = len(signal)
    embedded = time_delay_embedding(signal, emb_dim, tau)
    M = len(embedded)

    if M < min_separation * 2:
        return 0.0

    # Find nearest neighbors (excluding temporal neighbors)
    nn_indices = np.zeros(M, dtype=int)
    nn_distances = np.full(M, np.inf)

    for i in range(M):
        for j in range(M):
            if abs(i - j) > min_separation:
                d = np.linalg.norm(embedded[i] - embedded[j])
                if d < nn_distances[i] and d > 1e-15:
                    nn_distances[i] = d
                    nn_indices[i] = j

    # Track divergence
    n_steps = min(max_iter, M // 4)
    divergences = np.zeros(n_steps)
    counts = np.zeros(n_steps)

    for i in range(M - n_steps):
        j = nn_indices[i]
        if j + n_steps >= M or nn_distances[i] == np.inf:
            continue
        for k in range(n_steps):
            d = np.linalg.norm(embedded[i + k] - embedded[j + k])
            if d > 1e-15:
                divergences[k] += np.log(d)
                counts[k] += 1

    # Average divergence curve
    mask = counts > 0
    if np.sum(mask) < 5:
        return 0.0

    avg_divergence = np.zeros(n_steps)
    avg_divergence[mask] = divergences[mask] / counts[mask]

    # Fit slope to linear region (initial divergence)
    times = np.arange(n_steps) * dt
    valid = mask & (times > 0)
    if np.sum(valid) < 5:
        return 0.0

    n_fit = max(5, np.sum(valid) // 3)
    valid_idx = np.where(valid)[0][:n_fit]

    slope, _ = np.polyfit(times[valid_idx], avg_divergence[valid_idx], 1)

    return float(slope)


# ═══════════════════════════════════════════════════════════════════════════
# 5. POWER SPECTRAL SLOPE (β)
# ═══════════════════════════════════════════════════════════════════════════

def power_spectral_slope(signal: np.ndarray, dt: float = 0.5) -> float:
    """
    Estimate the power spectral density slope β from a time series.

    For healthy 1/f-like dynamics, β ≈ 1.0.
    White noise: β ≈ 0, Brownian noise: β ≈ 2.

    Parameters
    ----------
    signal : ndarray, shape (N,)
        Input time series.
    dt : float
        Sampling interval.

    Returns
    -------
    beta : float
        Negative slope of log(PSD) vs log(freq). β > 0 for 1/f^β.
    """
    N = len(signal)
    if N < 10:
        return 0.0

    signal = signal - np.mean(signal)
    freqs = np.fft.rfftfreq(N, d=dt)[1:]
    psd = np.abs(np.fft.rfft(signal)[1:]) ** 2 / N

    mask = (psd > 0) & (freqs > 0)
    if np.sum(mask) < 5:
        return 0.0

    log_f = np.log10(freqs[mask])
    log_psd = np.log10(psd[mask])
    slope, _ = np.polyfit(log_f, log_psd, 1)

    return float(-slope)


# ═══════════════════════════════════════════════════════════════════════════
# 6. INTER-SYSTEM COUPLING
# ═══════════════════════════════════════════════════════════════════════════

def coupling_score(trajectory: np.ndarray,
                   group_a: Optional[List[int]] = None,
                   group_b: Optional[List[int]] = None) -> float:
    """
    Compute inter-system coupling strength between two subsystems.

    Measures the mean absolute correlation between variables in group_a
    (e.g., metabolic) and group_b (e.g., immune).

    Parameters
    ----------
    trajectory : ndarray, shape (n_vars, n_timepoints)
    group_a : list of int
        Indices for subsystem A (default: metabolic [0:10]).
    group_b : list of int
        Indices for subsystem B (default: immune [10:13]).

    Returns
    -------
    coupling : float
        Mean absolute cross-correlation in [0, 1].
    """
    n_vars = trajectory.shape[0]
    if group_a is None:
        group_a = list(range(min(10, n_vars)))
    if group_b is None:
        group_b = list(range(10, min(13, n_vars)))

    if not group_a or not group_b:
        return 0.5  # Undefined → neutral

    correlations = []
    eps = 1e-12
    for i in group_a:
        for j in group_b:
            if i < n_vars and j < n_vars:
                a = trajectory[i]
                b = trajectory[j]
                if np.std(a) < eps or np.std(b) < eps:
                    continue
                c = np.corrcoef(a, b)[0, 1]
                if np.isfinite(c):
                    correlations.append(abs(c))

    return float(np.mean(correlations)) if correlations else 0.5


# ═══════════════════════════════════════════════════════════════════════════
# 7. COMPOSITE COHERENCE METRIC C(t)
# ═══════════════════════════════════════════════════════════════════════════

HEALTHY_REFERENCE = {
    "D2": 4.5,         "D2_sigma": 1.5,
    "MSE": 0.8,
    "lyap_max": 0.05,  "lyap_sigma": 0.03,
    "beta": 1.0,       "beta_sigma": 0.3,
}

DEFAULT_WEIGHTS = {
    "D2": 0.30,
    "MSE": 0.25,
    "lyap": 0.25,
    "beta": 0.20,
}


def _gaussian_score(value: float, target: float, sigma: float) -> float:
    """Gaussian normalization: 1.0 at target, decays with distance."""
    return float(np.exp(-0.5 * ((value - target) / (sigma + 1e-10)) ** 2))


def _tanh_score(value: float, target: float) -> float:
    """Saturating normalization: approaches 1.0 at target."""
    return float(np.tanh(value / (target + 1e-10)))


def coherence_metric(D2: float, mse_mean_val: float, lyap_max: float,
                     beta: float,
                     weights: Optional[Dict[str, float]] = None,
                     reference: Optional[Dict[str, float]] = None) -> Dict:
    """
    Compute the composite coherence metric C.

    C = w_D·f_D(D₂) + w_S·f_S(MSE) + w_λ·f_λ(λ_max) + w_β·f_β(β)
    """
    w = weights or DEFAULT_WEIGHTS
    ref = reference or HEALTHY_REFERENCE

    D2_score = _gaussian_score(D2, ref["D2"], ref["D2_sigma"])
    MSE_score = _tanh_score(mse_mean_val, ref["MSE"])
    lyap_score = _gaussian_score(lyap_max, ref["lyap_max"], ref["lyap_sigma"])
    beta_score = _gaussian_score(beta, ref["beta"], ref["beta_sigma"])

    C = (w["D2"] * D2_score + w["MSE"] * MSE_score
         + w["lyap"] * lyap_score + w["beta"] * beta_score)

    return {
        "C": round(float(C), 4),
        "components": {
            "D2_score": round(D2_score, 4),
            "MSE_score": round(MSE_score, 4),
            "lyap_score": round(lyap_score, 4),
            "beta_score": round(beta_score, 4),
        },
        "raw": {
            "D2": round(D2, 4),
            "MSE_mean": round(mse_mean_val, 4),
            "lyap_max": round(lyap_max, 4),
            "beta": round(beta, 4),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# 8. PHI PROFILE — 5D DYNAMICAL COMPLEXITY VECTOR
# ═══════════════════════════════════════════════════════════════════════════

# LOINC / SNOMED codes for regulatory alignment
BIOMARKER_CODES = {
    "Phi_temporal": {
        "loinc": "8867-4",  # Heart rate variability (proxy for temporal complexity)
        "snomed": "251670001",
        "description": "Temporal variability — HRV, glucose variability, MSE"
    },
    "Phi_spatial": {
        "loinc": "33747-3",  # Histologic grade (proxy for spatial heterogeneity)
        "snomed": "371469007",
        "description": "Spatial heterogeneity — cell diversity, tumor architecture, D₂"
    },
    "Phi_functional": {
        "loinc": "30525-0",  # Age (proxy for stress response capacity)
        "snomed": "165109007",
        "description": "Functional resilience — stress recovery, perturbation response"
    },
    "Phi_informational": {
        "loinc": "LP99691-0",  # EEG/ECG entropy (proxy)
        "snomed": "251629003",
        "description": "Information processing — signal entropy, network capacity"
    },
    "Phi_coupling": {
        "loinc": "26881-3",  # IL-6 (proxy for immune-metabolic coupling)
        "snomed": "52988006",
        "description": "Inter-system coupling — immune-metabolic correlation, organ synchrony"
    },
}


@dataclass
class PhiProfile:
    """
    5-Dimensional Dynamical Complexity Profile (Φ vector).

    Each dimension ranges from 0.0 (collapsed/absent) to 1.0 (maximal complexity).
    The healthy target is moderate complexity in all dimensions (~0.5-0.8).
    """
    Phi_temporal: float = 0.0
    Phi_spatial: float = 0.0
    Phi_functional: float = 0.0
    Phi_informational: float = 0.0
    Phi_coupling: float = 0.0

    # Archetype classification
    archetype: str = "Unknown"
    archetype_confidence: float = 0.0

    # Coherence metric (backward compatibility)
    coherence_C: float = 0.0

    # Raw metrics used for computation
    raw_metrics: Dict = field(default_factory=dict)
    
    # Memory Layer Features
    memory_features: Dict[str, float] = field(default_factory=dict)

    @property
    def phi_vector(self) -> np.ndarray:
        """Return the 5D Φ vector as numpy array."""
        return np.array([
            self.Phi_temporal, self.Phi_spatial, self.Phi_functional,
            self.Phi_informational, self.Phi_coupling
        ])

    @property
    def phi_magnitude(self) -> float:
        """L2 norm of Φ — overall complexity magnitude."""
        return float(np.linalg.norm(self.phi_vector))

    @property
    def phi_mean(self) -> float:
        """Mean of Φ dimensions — simple overall complexity score."""
        return float(np.mean(self.phi_vector))

    def to_json(self, path: Optional[str] = None) -> str:
        """Export as JSON (phi_profile.json format)."""
        data = {
            "phi_vector": {
                "Phi_temporal": round(self.Phi_temporal, 4),
                "Phi_spatial": round(self.Phi_spatial, 4),
                "Phi_functional": round(self.Phi_functional, 4),
                "Phi_informational": round(self.Phi_informational, 4),
                "Phi_coupling": round(self.Phi_coupling, 4),
            },
            "summary": {
                "phi_magnitude": round(self.phi_magnitude, 4),
                "phi_mean": round(self.phi_mean, 4),
                "coherence_C": round(self.coherence_C, 4),
                "archetype": self.archetype,
                "archetype_confidence": round(self.archetype_confidence, 4),
            },
            "raw_metrics": self.raw_metrics,
            "memory_features": self.memory_features,
            "biomarker_codes": BIOMARKER_CODES,
        }
        json_str = json.dumps(data, indent=2)
        if path:
            with open(path, 'w') as f:
                f.write(json_str)
        return json_str


# ═══════════════════════════════════════════════════════════════════════════
# 9. ARCHETYPE CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════

def classify_archetype(phi: PhiProfile, use_ml: bool = True) -> Tuple[str, float]:
    """
    Classify the pathology archetype from the Φ vector.

    Uses ML classifier (DT + RF ensemble) if available, falls back to
    expanded centroid-distance method.

    Archetypes (8 expanded):
        Healthy Complex     — Moderate all dimensions (target state)
        Chaotic/Decoupled   — Low coupling, high temporal/informational chaos
        Rigid/Locked        — High coupling, low temporal/spatial variability
        Collapsed/Exhausted — All dimensions low
        Warburg Metabolic   — High functional, elevated informational, low coupling
        Immune Evasion      — Moderate temporal, very low coupling
        Transitional/Pre-disease — Mild deviations from healthy
        Mixed Pathology     — Multiple aberrant dimensions

    Returns
    -------
    (archetype_name, confidence)
    """
    # Try ML classifier first
    if use_ml:
        try:
            from models.ml_classifier import get_classifier
            clf = get_classifier()
            archetype, confidence, method = clf.classify(phi)
            return archetype, confidence
        except Exception:
            pass  # Fall through to centroid method

    v = phi.phi_vector

    # Expanded archetype signatures (centroid vectors in Φ-space)
    archetypes = {
        "Healthy Complex":       np.array([0.60, 0.55, 0.75, 0.45, 0.60]),
        "Chaotic/Decoupled":     np.array([0.80, 0.70, 0.30, 0.80, 0.15]),
        "Rigid/Locked":          np.array([0.15, 0.20, 0.20, 0.30, 0.85]),
        "Collapsed/Exhausted":   np.array([0.10, 0.10, 0.10, 0.10, 0.10]),
        "Warburg Metabolic":     np.array([0.35, 0.35, 0.90, 0.65, 0.30]),
        "Immune Evasion":        np.array([0.40, 0.30, 0.80, 0.50, 0.20]),
        "Transitional/Pre-disease": np.array([0.50, 0.45, 0.65, 0.50, 0.45]),
        "Mixed Pathology":       np.array([0.55, 0.50, 0.50, 0.70, 0.35]),
    }

    # Compute distances, pick closest
    distances = {name: float(np.linalg.norm(v - centroid))
                 for name, centroid in archetypes.items()}

    best = min(distances, key=distances.get)
    max_possible = np.sqrt(5)  # max L2 in [0,1]^5
    confidence = 1.0 - (distances[best] / max_possible)

    return best, round(confidence, 4)


# ═══════════════════════════════════════════════════════════════════════════
# 10. COMPLEXITY PROFILER — MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════

class ComplexityProfiler:
    """
    UCP Module 1: Complexity Profiler.

    Computes the 5D Φ vector from trajectory data or clinical measurements.

    Usage:
        profiler = ComplexityProfiler()
        phi = profiler.profile(trajectory, dt=0.5)
        phi.to_json("phi_profile.json")
    """

    def __init__(self,
                 healthy_reference: Optional[Dict] = None,
                 weights: Optional[Dict[str, float]] = None,
                 use_ml_classifier: bool = True):
        """
        Parameters
        ----------
        healthy_reference : dict, optional
            Override healthy reference values for coherence scoring.
        weights : dict, optional
            Override weights for coherence metric components.
        use_ml_classifier : bool
            If True (default), use ML classifier for archetype detection.
            Falls back to centroid-distance if sklearn is not available.
        """
        self.healthy_reference = healthy_reference or HEALTHY_REFERENCE
        self.weights = weights or DEFAULT_WEIGHTS
        self.use_ml_classifier = use_ml_classifier

    def profile(self, trajectory: np.ndarray, dt: float = 0.5,
                metabolic_idx: Optional[List[int]] = None,
                immune_idx: Optional[List[int]] = None,
                perturbation_response: Optional[np.ndarray] = None,
                mse_max_scale: int = 15,
                emb_dim_D2: int = 8,
                emb_dim_lyap: int = 7,
                memory_features: Optional[Dict[str, float]] = None) -> PhiProfile:
        """
        Compute the full 5D Φ profile from a state trajectory.

        Parameters
        ----------
        trajectory : ndarray, shape (n_vars, n_timepoints)
            Full state trajectory from ODE solver.
        dt : float
            Time step between samples (days).
        metabolic_idx : list of int, optional
            Indices for metabolic variables (default: [0:10]).
        immune_idx : list of int, optional
            Indices for immune variables (default: [10:13]).
        perturbation_response : ndarray, optional
            Recovery trajectory after perturbation (for Φ_functional).
        mse_max_scale : int
            Maximum scale for MSE computation.
        emb_dim_D2 : int
            Embedding dimension for correlation dimension.
        emb_dim_lyap : int
            Embedding dimension for Lyapunov exponent.

        Returns
        -------
        phi : PhiProfile
            Complete 5D complexity profile with archetype classification.
        """
        n_vars, n_points = trajectory.shape
        if metabolic_idx is None:
            metabolic_idx = list(range(min(10, n_vars)))
        if immune_idx is None:
            immune_idx = list(range(10, min(13, n_vars)))

        signal = trajectory[0, :]  # Primary signal (Glucose by default)
        raw = {}

        # ── Φ_temporal: Temporal variability (MSE) ──
        mse_val = mse_mean(signal, max_scale=min(mse_max_scale, n_points // 20))
        raw["MSE_mean"] = round(mse_val, 4)
        # Normalize: MSE / healthy_reference → clamp to [0, 1]
        Phi_temporal = float(np.clip(
            mse_val / (self.healthy_reference["MSE"] + 1e-10), 0, 1))

        # ── Φ_spatial: Spatial heterogeneity (D₂) ──
        if n_vars >= 5:
            composite = np.mean(trajectory[:5, :], axis=0)
        else:
            composite = signal
        D2 = correlation_dimension(composite, emb_dim=emb_dim_D2, tau=2,
                                   max_points=800)
        raw["D2"] = round(D2, 4)
        Phi_spatial = float(np.clip(
            D2 / (self.healthy_reference["D2"] + 1e-10), 0, 1))

        # ── Φ_functional: Stress response / resilience ──
        if perturbation_response is not None and len(perturbation_response) > 10:
            # Recovery rate: how fast does the system return toward baseline
            diff = np.abs(perturbation_response - perturbation_response[-1])
            if diff[0] > 1e-10:
                half_recovery_idx = np.argmax(diff < diff[0] * 0.5)
                if half_recovery_idx > 0:
                    recovery_rate = 1.0 / (half_recovery_idx * dt + 1e-10)
                else:
                    recovery_rate = 0.0
            else:
                recovery_rate = 1.0
            Phi_functional = float(np.clip(recovery_rate / 0.5, 0, 1))
            raw["recovery_rate"] = round(recovery_rate, 4)
        else:
            # Fallback: use trajectory variance stability
            seg1_var = np.var(trajectory[:, :n_points // 2])
            seg2_var = np.var(trajectory[:, n_points // 2:])
            var_ratio = min(seg1_var, seg2_var) / (max(seg1_var, seg2_var) + 1e-10)
            Phi_functional = float(np.clip(var_ratio, 0, 1))
            raw["variance_stability"] = round(var_ratio, 4)

        # ── Φ_informational: Information processing (SampEn + λ_max) ──
        lyap = largest_lyapunov_exponent(trajectory, dt=dt,
                                          emb_dim=emb_dim_lyap, tau=2)
        beta = power_spectral_slope(signal, dt=dt)
        raw["lyap_max"] = round(lyap, 4)
        raw["beta"] = round(beta, 4)
        # Combine Lyapunov and spectral slope into informational score
        lyap_score = _gaussian_score(lyap,
                                     self.healthy_reference["lyap_max"],
                                     self.healthy_reference["lyap_sigma"])
        beta_score = _gaussian_score(beta,
                                     self.healthy_reference["beta"],
                                     self.healthy_reference["beta_sigma"])
        Phi_informational = float(np.clip(0.5 * lyap_score + 0.5 * beta_score, 0, 1))

        # ── Φ_coupling: Inter-system communication ──
        Phi_coupling_val = coupling_score(trajectory,
                                          group_a=metabolic_idx,
                                          group_b=immune_idx)
        raw["coupling"] = round(Phi_coupling_val, 4)

        # Build PhiProfile
        C_result = coherence_metric(D2, mse_val, lyap, beta,
                                    weights=self.weights,
                                    reference=self.healthy_reference)

        mem_feats = memory_features or {}

        phi = PhiProfile(
            Phi_temporal=round(Phi_temporal, 4),
            Phi_spatial=round(Phi_spatial, 4),
            Phi_functional=round(Phi_functional, 4),
            Phi_informational=round(Phi_informational, 4),
            Phi_coupling=round(Phi_coupling_val, 4),
            coherence_C=C_result["C"],
            raw_metrics={**raw, **C_result},
            memory_features=mem_feats,
        )

        # Classify archetype
        archetype, confidence = classify_archetype(phi, use_ml=self.use_ml_classifier)
        phi.archetype = archetype
        phi.archetype_confidence = confidence

        return phi


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE: compute_all_metrics (backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════

def compute_all_metrics(trajectory: np.ndarray, dt: float = 0.5,
                        variable_index: int = 0,
                        mse_max_scale: int = 15,
                        emb_dim_D2: int = 8,
                        emb_dim_lyap: int = 7) -> Dict:
    """
    Compute all four complexity metrics from a trajectory.
    Backward-compatible API from complexity_metrics.py.
    """
    signal = trajectory[variable_index, :]
    n_points = len(signal)

    mse_val = mse_mean(signal, max_scale=min(mse_max_scale, n_points // 20))

    if trajectory.shape[0] >= 5:
        composite = np.mean(trajectory[:5, :], axis=0)
    else:
        composite = signal
    D2 = correlation_dimension(composite, emb_dim=emb_dim_D2, tau=2,
                               max_points=800)

    lyap = largest_lyapunov_exponent(trajectory, dt=dt,
                                      emb_dim=emb_dim_lyap, tau=2)
    beta = power_spectral_slope(signal, dt=dt)
    C_result = coherence_metric(D2, mse_val, lyap, beta)

    return {
        **C_result,
        "n_timepoints": n_points,
        "dt": dt,
        "variable_used": variable_index,
    }
