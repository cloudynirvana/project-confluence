"""Downscaled full-brain training smoke (not a 166700-step job)."""

from pathlib import Path

import numpy as np

from confluence.contracts import ALL_EFFECTOR_IDS, FULL_BRAIN_NEURONS
from confluence.controllers import make_controller
from confluence.training.loop import TrainConfig, train


def test_make_controller_f_accepts_full_scale_config():
    ctrl = make_controller("F", n_neurons=FULL_BRAIN_NEURONS, seed=1)
    assert ctrl.letter == "F"
    assert ctrl.network.n_neurons == 166700
    assert tuple(ctrl.drug_ids) == ALL_EFFECTOR_IDS


def test_smoke_train_512(tmp_path: Path):
    ckpt = tmp_path / "ckpt.npz"
    result = train(
        TrainConfig(
            n_neurons=512,
            episodes=2,
            days=12.0,
            dt=0.5,
            archetype="glioblastoma",
            seed=4,
            outer="both",
            checkpoint=str(ckpt),
            embodiment=False,
        )
    )
    assert len(result.episodes) == 2
    assert ckpt.exists()
    assert np.isfinite(result.best_reward)
    assert result.episodes[0].steps > 0
    loaded = np.load(ckpt)
    assert int(loaded["n_neurons"]) == 512
    assert loaded["w_sec"].shape[0] == len(ALL_EFFECTOR_IDS)
