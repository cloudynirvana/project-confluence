#!/usr/bin/env python3
"""Interactive closed-loop simulation of the Neural Therapy Brain.

Runs the trained recurrent dosing brain (``models.neural_brain``) in closed
loop against the mechanistic clonal-dynamics environment and compares it to
Maximum-Tolerated-Dose (MTD) and the hand-crafted robust adaptive expert.

Examples
--------
    # Quick run (trains a small brain on the fly), TNBC, with a trajectory plot
    python scripts/run_neural_brain_sim.py --cancer TNBC --plot

    # Use a pre-trained brain and run the randomized robustness stress test
    python scripts/run_neural_brain_sim.py --model results/neural_brain/brain.pt --robustness

DATA/VALIDATION: synthetic simulator only. Real-data training is pending
independent expert review and authentic clinical validation. Research use only.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402


def _jsonable(o):
    """JSON encoder fallback for numpy scalar types."""
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

from models.adaptive_controller import (  # noqa: E402
    PolicyMode,
    PolicyParams,
    run_adaptive_simulation,
)
from models.neural_brain import (  # noqa: E402
    TORCH_AVAILABLE,
    evaluate_robustness,
    generate_expert_dataset,
    load_brain,
    run_neural_brain_simulation,
    train_brain,
)


def _get_brain(args):
    if args.model and Path(args.model).exists():
        print(f"[brain] loading pre-trained model from {args.model}")
        return load_brain(args.model)
    print(f"[brain] no saved model; training a quick brain "
          f"({args.train_scenarios} scenarios)...")
    dataset = generate_expert_dataset(
        n_scenarios=args.train_scenarios, total_days=args.days, dt=args.dt, seed=args.seed
    )
    brain, hist = train_brain(dataset, epochs=args.train_epochs, seed=args.seed)
    print(f"[brain] trained (final val_loss={hist['val_loss'][-1]:.5f})")
    return brain


def _plot(brain_res, mtd_res, cancer, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bt = brain_res["trajectories"]
    mt = mtd_res["trajectories"]
    tb = np.array(bt["time"]); tm = np.array(mt["time"])

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    fig.suptitle(f"Neural Therapy Brain vs MTD — {cancer}", fontsize=14, fontweight="bold")

    axes[0].plot(tb, bt["burden"], label="Neural Brain", color="#1f77b4", lw=2)
    axes[0].plot(tm, mt["burden"], label="MTD", color="#d62728", lw=2, ls="--")
    axes[0].set_ylabel("Tumour burden (V/K)")
    axes[0].legend(loc="upper right"); axes[0].grid(alpha=0.3)

    def rfrac(tr):
        s = np.array(tr["sensitive"]); r = np.array(tr["resistant"])
        return r / np.maximum(s + r, 1e-9)

    axes[1].plot(tb, rfrac(bt), label="Neural Brain", color="#1f77b4", lw=2)
    axes[1].plot(tm, rfrac(mt), label="MTD", color="#d62728", lw=2, ls="--")
    axes[1].axhline(0.8, color="k", ls=":", alpha=0.5, label="takeover (0.8)")
    axes[1].set_ylabel("Resistant fraction")
    axes[1].legend(loc="upper left"); axes[1].grid(alpha=0.3)

    axes[2].plot(tb[: len(bt["doses"])], bt["doses"], label="Neural Brain dose",
                 color="#1f77b4", lw=1.5)
    axes[2].set_ylabel("Dose (fraction u_max)")
    axes[2].set_xlabel("Time (days)")
    axes[2].legend(loc="upper right"); axes[2].grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"[plot] saved trajectory figure → {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Neural Therapy Brain closed-loop simulation.")
    parser.add_argument("--cancer", type=str, default="TNBC")
    parser.add_argument("--days", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="results/neural_brain/brain.pt")
    parser.add_argument("--train-scenarios", type=int, default=60)
    parser.add_argument("--train-epochs", type=int, default=80)
    parser.add_argument("--robustness", action="store_true",
                        help="Also run the randomized robustness stress test.")
    parser.add_argument("--eval-scenarios", type=int, default=30)
    parser.add_argument("--plot", action="store_true",
                        help="Save a trajectory plot (Neural Brain vs MTD).")
    parser.add_argument("--out", type=str, default="results/neural_brain")
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is not installed. `pip install torch` (CPU is fine).")
        return 1

    print("=" * 64)
    print(f"PROJECT CONFLUENCE — Neural Therapy Brain closed-loop ({args.cancer})")
    print("=" * 64)

    brain = _get_brain(args)

    print(f"\n[1] Single-scenario closed loop ({args.cancer}, {args.days:.0f} days)...")
    brain_res = run_neural_brain_simulation(brain, args.cancer, args.days, args.dt, args.seed)
    mtd_params = PolicyParams(
        dose_on_threshold=0.0, dose_off_threshold=0.0, robust_max_dose=1.0,
        max_continuous_dose_days=999, min_holiday_days=0, max_cumulative_toxicity=999,
    )
    mtd_res = run_adaptive_simulation(args.cancer, PolicyMode.THRESHOLD, mtd_params,
                                      int(args.days), args.dt, args.seed)
    expert_res = run_adaptive_simulation(args.cancer, PolicyMode.ROBUST_ADAPTIVE, None,
                                         int(args.days), args.dt, args.seed)

    print(f"\n    {'strategy':12s} | final burden | final R% | ctrl days | takeover")
    print("    " + "-" * 58)
    for name, res in (("MTD", mtd_res), ("Expert", expert_res), ("NeuralBrain", brain_res)):
        o = res["outcome"]
        print(f"    {name:12s} | {o['final_burden']:12.4f} "
              f"| {o['final_resistant_fraction']:6.1%} "
              f"| {o['days_under_control']:9.1f} "
              f"| {o['resistant_takeover']}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"sim_{args.cancer}.json").write_text(json.dumps(
        {"NeuralBrain": brain_res["outcome"], "MTD": mtd_res["outcome"],
         "Expert": expert_res["outcome"]}, indent=2, default=_jsonable))

    if args.plot:
        _plot(brain_res, mtd_res, args.cancer, str(Path(args.out) / f"trajectory_{args.cancer}.png"))

    if args.robustness:
        print(f"\n[2] Randomized robustness stress test ({args.eval_scenarios} scenarios)...")
        report = evaluate_robustness(brain, n_scenarios=args.eval_scenarios,
                                     total_days=args.days, dt=args.dt, seed=args.seed + 777)
        print(f"\n    {'strategy':12s} | takeover | mean R% | mean burden | ctrl days")
        print("    " + "-" * 60)
        for name, s in report["strategies"].items():
            print(f"    {name:12s} | {s['resistant_takeover_rate']:7.1%} "
                  f"| {s['mean_final_resistant_fraction']:6.1%} "
                  f"| {s['mean_final_burden']:11.4f} "
                  f"| {s['mean_days_under_control']:.1f}")
        (Path(args.out) / "robustness.json").write_text(json.dumps(report, indent=2, default=_jsonable))

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
