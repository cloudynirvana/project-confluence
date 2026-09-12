"""Tests for the Neural Therapy Brain (models.neural_brain).

These tests exercise the observation encoder, dataset generation, the recurrent
policy network, behavioral-cloning training, the safety-wrapped closed-loop
controller, and the real-data loader's validation gate.

PyTorch is required; the tests skip cleanly if it is unavailable.
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from models.neural_brain import (  # noqa: E402
    OBS_DIM,
    NeuralBrainController,
    NeuralTherapyBrain,
    TherapyDataset,
    encode_observation,
    generate_expert_dataset,
    load_brain,
    run_neural_brain_simulation,
    save_brain,
    train_brain,
)


# ── Observation encoding ────────────────────────────────────────────────────

def test_encode_observation_shape_and_bounds():
    obs = encode_observation(
        sensitive=0.8, resistant=0.2, carrying_capacity=1.0,
        efficacy=1.0, toxicity_fraction=0.0, is_dosing=False, last_dose=0.0,
    )
    assert len(obs) == OBS_DIM
    assert obs[0] == pytest.approx(1.0)          # V_frac
    assert obs[1] == pytest.approx(0.2)          # R_frac
    assert obs[4] == 0.0                         # is_dosing flag
    assert all(v >= 0.0 for v in obs)


# ── Dataset generation ──────────────────────────────────────────────────────

def test_generate_expert_dataset_shapes():
    ds = generate_expert_dataset(n_scenarios=6, total_days=30, dt=0.5, seed=1)
    seq_len = int(30 / 0.5)
    assert ds.observations.shape == (6, seq_len, OBS_DIM)
    assert ds.target_doses.shape == (6, seq_len)
    assert ds.source == "synthetic_expert"
    assert ds.validated is True
    # Doses are within the physical [0, 1] range.
    assert ds.target_doses.min() >= 0.0
    assert ds.target_doses.max() <= 1.0


# ── Network forward pass ────────────────────────────────────────────────────

def test_brain_forward_output_bounds():
    brain = NeuralTherapyBrain(hidden_dim=16, max_dose=0.9)
    x = torch.rand(4, 20, OBS_DIM)
    out = brain(x)
    assert out.shape == (4, 20)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 0.9 + 1e-5


def test_brain_step_carries_hidden_state():
    brain = NeuralTherapyBrain(hidden_dim=16)
    obs = encode_observation(0.7, 0.3, 1.0, 1.0, 0.0, False, 0.0)
    dose1, h1 = brain.step(obs, None)
    dose2, h2 = brain.step(obs, h1)
    assert 0.0 <= dose1 <= brain.max_dose
    assert h1.shape == (1, 16)
    assert h2.shape == (1, 16)


# ── Training ────────────────────────────────────────────────────────────────

def test_training_reduces_loss():
    ds = generate_expert_dataset(n_scenarios=12, total_days=40, dt=0.5, seed=2)
    _, history = train_brain(ds, epochs=40, lr=5e-3, seed=0)
    assert history["train_loss"][-1] < history["train_loss"][0]
    # Behavioral cloning should reach a small MSE on this low-dim signal.
    assert history["train_loss"][-1] < 0.05


# ── Save / load round-trip ──────────────────────────────────────────────────

def test_save_and_load_brain_roundtrip():
    ds = generate_expert_dataset(n_scenarios=6, total_days=30, dt=0.5, seed=3)
    brain, _ = train_brain(ds, epochs=10, seed=0)
    obs = encode_observation(0.6, 0.4, 1.0, 0.8, 0.1, True, 0.5)
    dose_before, _ = brain.step(obs, None)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "brain.pt")
        save_brain(brain, path, meta={"note": "test"})
        loaded = load_brain(path)
    dose_after, _ = loaded.step(obs, None)
    assert dose_before == pytest.approx(dose_after, abs=1e-5)


# ── Closed-loop deployment + safety layer ───────────────────────────────────

def test_closed_loop_simulation_outcome_valid():
    ds = generate_expert_dataset(n_scenarios=12, total_days=40, dt=0.5, seed=4)
    brain, _ = train_brain(ds, epochs=40, seed=0)
    res = run_neural_brain_simulation(brain, "NSCLC", total_days=40, dt=0.5, seed=42)
    o = res["outcome"]
    assert o["final_burden"] >= 0.0
    assert 0.0 <= o["final_resistant_fraction"] <= 1.0
    assert res["controller_summary"]["policy_mode"] == "neural_brain"


def test_safety_layer_caps_dose():
    """The assurance layer must clamp any brain output to the absolute cap."""
    brain = NeuralTherapyBrain(hidden_dim=8, max_dose=0.9)
    controller = NeuralBrainController(brain, cancer_type="NSCLC")
    cap = controller.safety.params.robust_max_dose
    max_seen = 0.0
    for _ in range(200):
        dose = controller.decide(sensitive=0.9, resistant=0.1,
                                 carrying_capacity=1.0, dt=0.5, resistance_efficacy=1.0)
        max_seen = max(max_seen, dose)
    assert max_seen <= cap + 1e-9


def test_brain_contains_resistance_better_than_mtd():
    """Trained brain should not lose to resistant takeover where MTD does."""
    from models.adaptive_controller import (
        PolicyMode, PolicyParams, run_adaptive_simulation,
    )
    ds = generate_expert_dataset(n_scenarios=24, total_days=60, dt=0.5, seed=5)
    brain, _ = train_brain(ds, epochs=80, seed=0)
    brain_res = run_neural_brain_simulation(brain, "NSCLC", total_days=60, dt=0.5, seed=42)

    mtd_params = PolicyParams(
        dose_on_threshold=0.0, dose_off_threshold=0.0, robust_max_dose=1.0,
        max_continuous_dose_days=999, min_holiday_days=0, max_cumulative_toxicity=999,
    )
    mtd_res = run_adaptive_simulation("NSCLC", PolicyMode.THRESHOLD, mtd_params, 60, 0.5, 42)

    assert (brain_res["outcome"]["final_resistant_fraction"]
            < mtd_res["outcome"]["final_resistant_fraction"])
    assert not brain_res["outcome"]["resistant_takeover"]


# ── Real-data hook: validation gate ─────────────────────────────────────────

def test_real_cohort_requires_validation():
    cohort = {
        "validated": False,
        "carrying_capacity": 1.0,
        "patients": [
            {"id": "PT-1", "steps": [
                {"sensitive": 0.8, "resistant": 0.2, "efficacy": 1.0,
                 "toxicity_fraction": 0.0, "dose": 0.0},
                {"sensitive": 0.7, "resistant": 0.25, "efficacy": 0.9,
                 "toxicity_fraction": 0.1, "dose": 0.6},
            ]},
        ],
    }
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cohort.json")
        with open(path, "w") as f:
            json.dump(cohort, f)

        # Unvalidated cohort must be refused by default.
        with pytest.raises(ValueError):
            TherapyDataset.from_real_cohort(path)

        # Explicit research override loads it (clearly labelled).
        ds = TherapyDataset.from_real_cohort(path, require_validated=False)
        assert len(ds) == 1
        assert ds.observations.shape[-1] == OBS_DIM
        assert ds.validated is False
        assert ds.source.startswith("real_cohort:")
