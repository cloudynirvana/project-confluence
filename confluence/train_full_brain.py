"""Headless full-brain training CLI.

    python -m confluence.train_full_brain --neurons 2048 --episodes 3
    python -m confluence.train_full_brain --neurons 166700 --episodes N

Research simulation only. Neurons do not synthesize therapeutics.
"""

from __future__ import annotations

import argparse
import json
import sys

from confluence.contracts import FULL_BRAIN_NEURONS, INTERACTIVE_BRAIN_NEURONS
from confluence.training.loop import TrainConfig, train


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a sparse rate-based full-brain controller on the cancer "
            "closed loop (simulated protein + small-molecule effectors)."
        )
    )
    parser.add_argument(
        "--neurons",
        type=int,
        default=INTERACTIVE_BRAIN_NEURONS,
        help=f"Network size (default {INTERACTIVE_BRAIN_NEURONS}; full scale {FULL_BRAIN_NEURONS})",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--days", type=float, default=80.0)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--archetype", default="glioblastoma")
    parser.add_argument("--checkpoint", default="results/full_brain/ckpt.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outer", choices=("da", "es", "both"), default="both")
    parser.add_argument("--embodiment", action="store_true", help="Also step the flybody bridge")
    parser.add_argument("--sigma", type=float, default=0.05, help="ES perturbation scale")
    args = parser.parse_args(argv)

    print("Confluence full-brain training — research simulation only.")
    print("Neurons are not ribosomes; no clinical dosing claim.")
    print(
        f"n_neurons={args.neurons}  episodes={args.episodes}  "
        f"days={args.days}  outer={args.outer}"
    )
    if args.neurons >= FULL_BRAIN_NEURONS:
        print(
            "Full 166700-neuron mode: sparse rate-based stub "
            "(not multicompartment LIF, not a real FlyWire dump)."
        )

    result = train(
        TrainConfig(
            n_neurons=args.neurons,
            episodes=args.episodes,
            days=args.days,
            dt=args.dt,
            archetype=args.archetype,
            seed=args.seed,
            outer=args.outer,
            checkpoint=args.checkpoint,
            embodiment=args.embodiment,
            sigma=args.sigma,
        )
    )
    for ep in result.episodes:
        proteins = ",".join(ep.protein_active) or "-"
        print(
            f"  ep {ep.episode:03d}  R={ep.reward:+.4f}  DA={ep.mean_da:+.3f}  "
            f"burden={ep.mean_burden:.3f}  tox={ep.mean_toxicity:.3f}  "
            f"H_end={ep.final_health:.3f}  proteins={proteins}  "
            f"{'keep' if ep.kept else 'revert'}"
        )
    print(f"best_reward={result.best_reward:+.4f}")
    print(f"checkpoint={result.checkpoint}")
    print(json.dumps({"best_reward": result.best_reward, "n_neurons": args.neurons}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
