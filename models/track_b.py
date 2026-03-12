"""
Track B Ingestion Utilities — Project Confluence
================================================

Minimal ingestion layer for real longitudinal cohorts:
  - Load cohort JSON
  - Map variables to 15D state
  - Resample to uniform dt
  - Compute Phi profiles
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
import numpy as np

from .ode_system import STATE_NAMES, ComplexAttractorODE, ExtendedParams
from .complexity_profiler import ComplexityProfiler
from .neural_ode import TORCHDIFFEQ_AVAILABLE


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


STATE_INDEX = {_normalize_name(n): i for i, n in enumerate(STATE_NAMES)}


@dataclass
class TrackBPatient:
    patient_id: str
    timepoints: List[float]
    observations: List[List[float]]
    survival_days: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class TrackBCohort:
    cohort_id: str
    disease: str
    time_unit: str
    variables: List[str]
    patients: List[TrackBPatient]


def load_track_b_cohort(path: str) -> TrackBCohort:
    with open(path, "r") as f:
        data = json.load(f)

    patients = [
        TrackBPatient(
            patient_id=p["patient_id"],
            timepoints=p["timepoints"],
            observations=p["observations"],
            survival_days=p.get("survival_days"),
            metadata={k: v for k, v in p.items()
                      if k not in {"patient_id", "timepoints", "observations", "survival_days"}},
        )
        for p in data.get("patients", [])
    ]

    return TrackBCohort(
        cohort_id=data.get("cohort_id", "unknown"),
        disease=data.get("disease", "unknown"),
        time_unit=data.get("time_unit", "days"),
        variables=data.get("variables", []),
        patients=patients,
    )


def _resample_timeseries(times: np.ndarray, values: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(times) < 2:
        return times, values
    t0, t1 = float(times[0]), float(times[-1])
    n_steps = int(max(2, np.ceil((t1 - t0) / dt) + 1))
    t_grid = np.linspace(t0, t1, n_steps)
    resampled = np.zeros((values.shape[0], len(t_grid)))
    for i in range(values.shape[0]):
        resampled[i] = np.interp(t_grid, times, values[i])
    return t_grid, resampled


def build_trajectory(patient: TrackBPatient,
                     variables: List[str],
                     baseline: np.ndarray,
                     resample_dt: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    times = np.array(patient.timepoints, dtype=float)
    obs = np.array(patient.observations, dtype=float)

    if obs.ndim != 2 or obs.shape[0] != len(times):
        raise ValueError(f"Observations must be shape (T, V) for {patient.patient_id}")
    if obs.shape[1] != len(variables):
        raise ValueError(
            f"Observations column count ({obs.shape[1]}) does not match variables ({len(variables)}) "
            f"for {patient.patient_id}"
        )

    # Map provided variables to 15D state indices
    mapped_idx = []
    for v in variables:
        key = _normalize_name(v)
        mapped_idx.append(STATE_INDEX.get(key, None))

    traj = np.tile(baseline.reshape(-1, 1), (1, len(times)))
    for j, idx in enumerate(mapped_idx):
        if idx is None:
            continue
        traj[idx, :] = obs[:, j]

    if len(times) < 2:
        dt = 1.0
        return traj, times, dt

    dt = resample_dt or float(np.median(np.diff(times)))
    if dt <= 0:
        dt = 1.0

    t_grid, traj_resampled = _resample_timeseries(times, traj, dt)
    return traj_resampled, t_grid, dt


def infer_neural_ode(trajectory: np.ndarray,
                     timepoints: np.ndarray) -> Optional[np.ndarray]:
    if not TORCHDIFFEQ_AVAILABLE:
        return None

    try:
        import torch
        from .neural_ode import ComplexityNeuralODE
    except Exception:
        return None

    # Expect full 15D observations
    obs = torch.tensor(trajectory.T, dtype=torch.float32).unsqueeze(0)
    t_span = torch.tensor(timepoints, dtype=torch.float32)
    model = ComplexityNeuralODE(obs_dim=trajectory.shape[0], state_dim=trajectory.shape[0])
    model.eval()
    with torch.no_grad():
        pred = model(obs, t_span)  # [batch, time, state_dim]
    return pred.squeeze(0).numpy().T


def compute_phi_for_patient(trajectory: np.ndarray, dt: float) -> Dict:
    profiler = ComplexityProfiler()
    n_points = trajectory.shape[1]
    # Ensure embedding dimensions are valid for short trajectories
    max_emb = max(2, min(8, int((n_points - 1) / 2) + 1))
    max_lyap = max(2, min(7, int((n_points - 1) / 2) + 1))
    mse_scale = min(15, max(2, n_points // 3))
    phi = profiler.profile(
        trajectory,
        dt=dt,
        mse_max_scale=mse_scale,
        emb_dim_D2=max_emb,
        emb_dim_lyap=max_lyap,
    )
    return {
        "phi_vector": phi.phi_vector,
        "phi_magnitude": float(phi.phi_magnitude),
        "coherence": float(phi.coherence_C),
        "archetype": phi.archetype,
        "archetype_confidence": float(phi.archetype_confidence),
    }


def healthy_reference_phi(dt: float = 0.5) -> np.ndarray:
    ode = ComplexAttractorODE(params=ExtendedParams(), use_nonlinear=True, use_immune=True, use_microenv=True)
    sol = ode.solve(t_span=(0, 200), dt_eval=dt)
    profiler = ComplexityProfiler()
    phi = profiler.profile(sol["z"], dt=dt)
    return np.array(phi.phi_vector)
