# -*- coding: utf-8 -*-
"""
Multi-Pathology Data Generator — Project Confluence
=====================================================

Integrates ALL disease ODE systems (cancer, CVD, diabetes, neurodegeneration)
to generate ODE-derived training data for the ML archetype classifier.

Replaces synthetic Gaussian-noise training data with real ODE trajectories
from 23 disease states across 4 pathology domains.

Usage:
    from models.pathology_integrator import generate_pathology_training_data
    X, y, labels = generate_pathology_training_data()
    clf.train_on_data(X, y, labels)
"""

import numpy as np
import warnings
import sys
import os
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

# Add parent paths for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'saem-cancer-poc', 'src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'saem-cancer-poc', 'tools')))

from models.ml_classifier import engineer_features, ArchetypeMLClassifier, ARCHETYPES

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ===================================================================
# PATHOLOGY ARCHETYPE MAPPING
# ===================================================================
# Maps each disease generator to its expected complexity archetype

PATHOLOGY_MAP = {
    # Cancer — Warburg Metabolic / Mixed Pathology
    "Cancer": {
        "tnbc":     "Warburg Metabolic",
        "pdac":     "Collapsed/Exhausted",      # deepest attractor
        "nsclc":    "Mixed Pathology",           # metabolically flexible
        "melanoma": "Transitional/Pre-disease",  # shallowest cancer
        "gbm":      "Warburg Metabolic",         # high glycolytic
        "aml":      "Immune Evasion",
    },
    # Cardiovascular — Rigid/Locked / Collapsed
    "CVD": {
        "dyslipidemia":   "Transitional/Pre-disease",
        "atherosclerosis": "Rigid/Locked",
        "acs":            "Chaotic/Decoupled",    # acute instability
        "heart_failure":  "Collapsed/Exhausted",
        "hypertension":   "Rigid/Locked",
    },
    # Diabetes — Mixed patterns
    "Diabetes": {
        "prediabetes":   "Transitional/Pre-disease",
        "t2d_early":     "Mixed Pathology",
        "t2d_advanced":  "Collapsed/Exhausted",
        "t1d":           "Immune Evasion",        # autoimmune
    },
    # Neurodegeneration — Chaotic/Decoupled / Collapsed
    "Neuro": {
        "mci_amyloid":  "Transitional/Pre-disease",
        "alzheimers":   "Chaotic/Decoupled",
        "parkinsons":   "Collapsed/Exhausted",
        "als":          "Collapsed/Exhausted",    # deepest neuro attractor
    },
}


def _simulate_10d_ode(A: np.ndarray, x0: np.ndarray, days: int = 200,
                       dt: float = 0.1, noise: float = 0.05,
                       seed: int = 42) -> np.ndarray:
    """
    Simulate a 10D linear ODE: dx/dt = A @ x + noise.

    Returns trajectory of shape (n_steps, 10).
    """
    rng = np.random.RandomState(seed)
    n = A.shape[0]
    n_steps = int(days / dt)
    trajectory = np.zeros((n_steps, n))
    x = x0.copy()

    for i in range(n_steps):
        dx = A @ x * dt + rng.randn(n) * noise * np.sqrt(dt)
        x = np.clip(x + dx, 0.01, 20.0)  # biological bounds
        trajectory[i] = x

    return trajectory


def _trajectory_to_phi(trajectory: np.ndarray) -> np.ndarray:
    """
    Compute a simplified 5D Phi vector from a 10D trajectory.

    Maps the 10D disease-specific ODE trajectory into the universal
    5D complexity space using domain-agnostic metrics.

    Returns: [phi_temporal, phi_spatial, phi_functional,
              phi_informational, phi_coupling]
    """
    T = trajectory.T  # (10, n_steps)
    n_vars, n_steps = T.shape

    # Phi_temporal: normalized temporal variability (CV across time)
    cvs = []
    for i in range(n_vars):
        mean = np.mean(T[i])
        std = np.std(T[i])
        cvs.append(std / (mean + 1e-8))
    phi_temporal = np.clip(np.mean(cvs), 0, 1)

    # Phi_spatial: inter-variable dispersion
    var_means = np.mean(T, axis=1)
    phi_spatial = np.clip(np.std(var_means) / (np.mean(var_means) + 1e-8), 0, 1)

    # Phi_functional: metabolic throughput (mean absolute rate of change)
    diffs = np.diff(T, axis=1)
    rates = np.mean(np.abs(diffs), axis=1)
    phi_functional = np.clip(np.mean(rates) * 5, 0, 1)

    # Phi_informational: signal complexity (approximate entropy proxy)
    # Use the ratio of high-freq to total spectral power
    spectral_ratios = []
    for i in range(min(n_vars, 5)):
        fft = np.abs(np.fft.rfft(T[i]))
        if len(fft) > 2:
            high = np.sum(fft[len(fft)//2:])
            total = np.sum(fft) + 1e-8
            spectral_ratios.append(high / total)
    phi_informational = np.clip(np.mean(spectral_ratios) * 3, 0, 1) if spectral_ratios else 0.5

    # Phi_coupling: mean absolute cross-correlation
    cors = []
    for i in range(min(n_vars, 5)):
        for j in range(i+1, min(n_vars, 5)):
            c = np.corrcoef(T[i], T[j])[0, 1]
            if np.isfinite(c):
                cors.append(abs(c))
    phi_coupling = np.clip(np.mean(cors), 0, 1) if cors else 0.5

    return np.array([phi_temporal, phi_spatial, phi_functional,
                     phi_informational, phi_coupling])


def _load_generators() -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, str]]]:
    """
    Load all disease generators from saem-cancer-poc.

    Returns dict of {domain: {disease: (A_matrix, x0, archetype_label)}}
    """
    generators = {}

    # ── Cancer generators ──
    try:
        from tnbc_ode import TNBCODESystem
        cancer_gens = {}
        ode = TNBCODESystem
        x0_cancer = np.array([2.0, 0.5, 1.0, 4.0, 1.5, 1.5, 0.8, 0.6, 1.0, 0.3])

        cancer_gens["healthy_cancer"] = (ode.healthy_generator(), x0_cancer, "Healthy Complex")
        cancer_gens["tnbc"] = (ode.tnbc_generator(), x0_cancer, PATHOLOGY_MAP["Cancer"]["tnbc"])

        # Try to load pan-cancer generators
        try:
            pan = ode.pan_cancer_generators()
            name_map = {"PDAC": "pdac", "NSCLC": "nsclc", "Melanoma": "melanoma",
                        "GBM": "gbm", "AML": "aml"}
            for name, A in pan.items():
                key = name_map.get(name, name.lower().replace(" ", "_"))
                if key in PATHOLOGY_MAP["Cancer"]:
                    cancer_gens[key] = (A, x0_cancer, PATHOLOGY_MAP["Cancer"][key])
        except Exception:
            pass

        generators["Cancer"] = cancer_gens
    except ImportError:
        pass

    # ── CVD generators ──
    try:
        from cvd_ode import CardiovascularODESystem
        cvd_gens = {}
        cvd = CardiovascularODESystem
        x0_cvd = np.ones(10)

        cvd_gens["healthy_cvd"] = (cvd.healthy_generator(), x0_cvd, "Healthy Complex")
        all_cvd = cvd.all_generators()
        cvd_name_map = {
            "Dyslipidemia": "dyslipidemia",
            "Atherosclerosis": "atherosclerosis",
            "Acute Coronary Syndrome": "acs",
            "Heart Failure (HFrEF)": "heart_failure",
            "Hypertension": "hypertension",
        }
        for name, A in all_cvd.items():
            key = cvd_name_map.get(name, name.lower().replace(" ", "_"))
            if key in PATHOLOGY_MAP.get("CVD", {}):
                cvd_gens[key] = (A, x0_cvd, PATHOLOGY_MAP["CVD"][key])

        generators["CVD"] = cvd_gens
    except ImportError:
        pass

    # ── Diabetes generators ──
    try:
        from diabetes_ode import DiabetesODESystem
        diab_gens = {}
        diab = DiabetesODESystem
        x0_diab = np.array([5.0, 10.0, 0.5, 2.0, 50.0, 0.2, 8.0, 1.0, 5.5, 10.0])

        diab_gens["healthy_diabetes"] = (diab.healthy_generator(), x0_diab, "Healthy Complex")
        all_diab = diab.all_generators()
        diab_name_map = {
            "Pre-diabetes": "prediabetes",
            "T2D Early": "t2d_early",
            "T2D Advanced": "t2d_advanced",
            "Type 1 Diabetes": "t1d",
        }
        for name, A in all_diab.items():
            key = diab_name_map.get(name, name.lower().replace(" ", "_"))
            if key in PATHOLOGY_MAP.get("Diabetes", {}):
                diab_gens[key] = (A, x0_diab, PATHOLOGY_MAP["Diabetes"][key])

        generators["Diabetes"] = diab_gens
    except ImportError:
        pass

    # ── Neurodegeneration generators ──
    try:
        from neuro_ode import NeurodegenerationODESystem
        neuro_gens = {}
        neuro = NeurodegenerationODESystem
        x0_neuro = np.ones(10)

        neuro_gens["healthy_neuro"] = (neuro.healthy_aging_generator(), x0_neuro, "Healthy Complex")
        all_neuro = neuro.all_generators()
        neuro_name_map = {
            "MCI (Amyloid+)": "mci_amyloid",
            "Alzheimer's Disease": "alzheimers",
            "Parkinson's Disease": "parkinsons",
            "ALS": "als",
        }
        for name, A in all_neuro.items():
            key = neuro_name_map.get(name, name.lower().replace(" ", "_"))
            if key in PATHOLOGY_MAP.get("Neuro", {}):
                neuro_gens[key] = (A, x0_neuro, PATHOLOGY_MAP["Neuro"][key])

        generators["Neuro"] = neuro_gens
    except ImportError:
        pass

    return generators


def generate_pathology_training_data(
    n_samples_per_generator: int = 50,
    simulation_days: int = 200,
    noise_range: Tuple[float, float] = (0.02, 0.10),
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
    """
    Generate training data from ALL pathology ODE systems.

    For each disease generator:
        1. Simulate the 10D ODE with varied noise levels
        2. Convert trajectory to 5D Phi vector
        3. Engineer 12D feature vector
        4. Label with archetype

    Returns
    -------
    X : ndarray (N, 12)
        Engineered feature matrix.
    y : ndarray (N,)
        Integer labels.
    class_names : list of str
        Archetype names.
    metadata : dict
        Stats about the generation run.
    """
    rng = np.random.RandomState(seed)
    generators = _load_generators()

    # Collect unique labels
    class_names = list(ARCHETYPES)
    label_to_idx = {name: i for i, name in enumerate(class_names)}

    X_list = []
    y_list = []
    source_list = []

    total_generators = 0
    total_samples = 0

    for domain, disease_gens in generators.items():
        for disease_name, (A, x0, archetype_label) in disease_gens.items():
            total_generators += 1

            for sample_i in range(n_samples_per_generator):
                # Vary noise and initial conditions
                noise = rng.uniform(noise_range[0], noise_range[1])
                x0_perturbed = x0 * (1 + rng.randn(len(x0)) * 0.1)
                x0_perturbed = np.clip(x0_perturbed, 0.01, 20.0)

                sample_seed = seed + total_generators * 1000 + sample_i

                # Simulate
                traj = _simulate_10d_ode(
                    A, x0_perturbed,
                    days=simulation_days,
                    noise=noise,
                    seed=sample_seed,
                )

                # Convert to Phi and features
                phi_vec = _trajectory_to_phi(traj)
                features = engineer_features(phi_vec)

                X_list.append(features)
                y_list.append(label_to_idx[archetype_label])
                source_list.append(f"{domain}/{disease_name}")
                total_samples += 1

    X = np.array(X_list)
    y = np.array(y_list)

    metadata = {
        "domains_loaded": list(generators.keys()),
        "total_generators": total_generators,
        "total_samples": total_samples,
        "samples_per_generator": n_samples_per_generator,
        "class_distribution": {
            class_names[i]: int(np.sum(y == i))
            for i in range(len(class_names))
            if np.sum(y == i) > 0
        },
        "sources": list(set(source_list)),
    }

    return X, y, class_names, metadata


def retrain_classifier_with_pathology_data(
    n_samples_per_generator: int = 50,
    blend_synthetic: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Retrain the ML classifier using real pathology ODE data.

    If blend_synthetic=True, combines ODE-derived data with synthetic
    Gaussian data for classes that might be underrepresented.

    Returns training report.
    """
    from models.ml_classifier import generate_training_data, ArchetypeMLClassifier

    clf = ArchetypeMLClassifier(verbose=verbose)

    # Generate ODE-derived data
    X_ode, y_ode, class_names, meta = generate_pathology_training_data(
        n_samples_per_generator=n_samples_per_generator
    )

    if verbose:
        print(f"\n  ODE data: {X_ode.shape[0]} samples from {meta['total_generators']} generators")
        print(f"  Domains: {meta['domains_loaded']}")
        for cls_name, count in meta['class_distribution'].items():
            print(f"    {cls_name}: {count} samples")

    if blend_synthetic:
        # Add synthetic data for any underrepresented classes
        X_syn, y_syn, _ = generate_training_data(n_per_class=100, noise_std=0.08)
        X = np.vstack([X_ode, X_syn])
        y = np.concatenate([y_ode, y_syn])
        if verbose:
            print(f"  + Synthetic: {X_syn.shape[0]} samples")
            print(f"  Total: {X.shape[0]} samples")
    else:
        X, y = X_ode, y_ode

    # Train
    report = clf.train_on_external_data(X, y, class_names)
    return {**report, "pathology_meta": meta}


# ===================================================================
# CLI
# ===================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Multi-Pathology Training Data Generator")
    print("=" * 60)

    X, y, labels, meta = generate_pathology_training_data(n_samples_per_generator=30)

    print(f"\n  Domains: {meta['domains_loaded']}")
    print(f"  Generators: {meta['total_generators']}")
    print(f"  Total samples: {meta['total_samples']}")
    print(f"\n  Class distribution:")
    for cls_name, count in meta['class_distribution'].items():
        print(f"    {cls_name:<25} {count:>5} samples")

    # Quick RF test
    print(f"\n  Feature matrix shape: {X.shape}")
    print(f"  Label vector shape:  {y.shape}")

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score, StratifiedKFold

        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
        print(f"\n  RF Cross-Val Accuracy: {scores.mean():.1%} (+/- {scores.std():.1%})")
    except ImportError:
        print("\n  scikit-learn not available for validation")

    print("\n  Done!")
