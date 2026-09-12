#!/usr/bin/env python3
"""Train the Neural Therapy Brain — Project Confluence.

Trains a compact recurrent dosing policy (``models.neural_brain``) by
behavioral cloning of the robust adaptive expert across many randomized
biological scenarios (the "ambivalent externalities"), then reports a
robustness comparison against MTD and the expert.

    python scripts/train_neural_brain.py --scenarios 80 --epochs 120

The trained brain is saved to ``results/neural_brain/brain.pt`` by default.

DATA/VALIDATION: training uses SYNTHETIC simulator trajectories. Real clinical
data can be supplied via ``TherapyDataset.from_real_cohort`` but is pending
independent expert review and authentic clinical validation. Research use only.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.neural_brain import (  # noqa: E402
    TORCH_AVAILABLE,
    evaluate_robustness,
    generate_expert_dataset,
    save_brain,
    train_brain,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the Neural Therapy Brain.")
    parser.add_argument("--scenarios", type=int, default=80,
                        help="Number of randomized training scenarios.")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--days", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-scenarios", type=int, default=30)
    parser.add_argument("--out", type=str, default="results/neural_brain/brain.pt")
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is not installed. Install it with "
              "`pip install torch` (CPU build is sufficient).")
        return 1

    print("=" * 64)
    print("PROJECT CONFLUENCE — Neural Therapy Brain training")
    print("=" * 64)

    print(f"\n[1] Generating expert dataset ({args.scenarios} scenarios, "
          f"{args.days:.0f} days @ dt={args.dt})...")
    dataset = generate_expert_dataset(
        n_scenarios=args.scenarios, total_days=args.days, dt=args.dt, seed=args.seed
    )
    print(f"    observations={dataset.observations.shape}  "
          f"source={dataset.source}  validated={dataset.validated}")

    print(f"\n[2] Training brain ({args.epochs} epochs, hidden={args.hidden})...")
    brain, history = train_brain(
        dataset, hidden_dim=args.hidden, epochs=args.epochs, lr=args.lr,
        seed=args.seed, verbose=True,
    )
    print(f"    final train_loss={history['train_loss'][-1]:.5f}  "
          f"val_loss={history['val_loss'][-1]:.5f}")

    meta = {"training": dataset.meta, "final_val_loss": history["val_loss"][-1]}
    save_brain(brain, args.out, meta=meta)
    hist_path = Path(args.out).with_suffix(".history.json")
    hist_path.write_text(json.dumps(history, indent=2))
    print(f"    saved model → {args.out}")
    print(f"    saved history → {hist_path}")

    print(f"\n[3] Robustness evaluation ({args.eval_scenarios} held-out scenarios)...")
    report = evaluate_robustness(
        brain, n_scenarios=args.eval_scenarios, total_days=args.days,
        dt=args.dt, seed=args.seed + 777,
    )
    print(f"\n    {'strategy':12s} | takeover | mean R% | mean burden | ctrl days")
    print("    " + "-" * 60)
    for name, s in report["strategies"].items():
        print(f"    {name:12s} | {s['resistant_takeover_rate']:7.1%} "
              f"| {s['mean_final_resistant_fraction']:6.1%} "
              f"| {s['mean_final_burden']:11.4f} "
              f"| {s['mean_days_under_control']:.1f}")

    report_path = Path(args.out).with_suffix(".robustness.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n    saved robustness report → {report_path}")

    brain_r = report["strategies"]["NeuralBrain"]["resistant_takeover_rate"]
    mtd_r = report["strategies"]["MTD"]["resistant_takeover_rate"]
    print("\n" + "=" * 64)
    if brain_r <= mtd_r:
        print(f">>> Neural Brain contains resistance better than MTD "
              f"({brain_r:.1%} vs {mtd_r:.1%} takeover). TRAINING OK.")
    else:
        print(f">>> WARNING: Neural Brain takeover {brain_r:.1%} exceeds MTD {mtd_r:.1%}.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
