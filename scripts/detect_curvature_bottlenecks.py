"""
Detect Forman-Ricci curvature bottlenecks in the BAC coupling network.

This script builds a simulated 5-scale block-Jacobian coupling network,
computes edge-level Forman-Ricci curvature, flags highly negative edges
between the neural/cellular scale and organismal scale, and exports both a
clinical JSON report and a Matplotlib network plot.

Run:
    python scripts/detect_curvature_bottlenecks.py
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

import matplotlib.pyplot as plt

from models.coupling_tensor import CouplingTensorAnalyzer
from models.network_curvature import NetworkCurvatureAnalyzer


RESULTS_DIR = PROJECT_ROOT / "results" / "curvature_bottlenecks"
REPORT_PATH = RESULTS_DIR / "curvature_bottleneck_report.json"
PLOT_PATH = RESULTS_DIR / "curvature_bottleneck_network.png"


def build_simulated_coupling_matrix() -> Tuple[np.ndarray, List[str]]:
    """
    Build a deterministic 5-scale BAC coupling matrix with a deliberately
    stressed cellular-to-organismal channel.

    The matrix is shaped like a block-Jacobian coupling tensor C_ij where
    row i receives influence from column j.
    """
    analyzer = CouplingTensorAnalyzer()
    scale_names = analyzer.scale_names

    # Baseline: dense enough to make structural bottlenecks meaningful.
    C = np.array([
        [1.00, 0.68, 0.56, 0.42, 0.35],
        [0.62, 1.00, 0.74, 0.52, 0.45],
        [0.48, 0.82, 1.00, 0.45, 0.57],
        [0.38, 0.55, 0.46, 1.00, 0.76],
        [0.32, 0.44, 0.58, 0.81, 1.00],
    ])

    # Remove self loops for graph curvature; diagonal coherence is represented
    # in the coupling tensor but is not an inter-scale communication edge.
    np.fill_diagonal(C, 0.0)
    return C, scale_names


def compute_curvature_report(threshold: float = -0.8) -> Dict:
    """Compute curvature and return a structured clinical report."""
    adjacency, scale_names = build_simulated_coupling_matrix()
    neural_idx = scale_names.index("cellular")
    organism_idx = scale_names.index("organism")

    curvature = NetworkCurvatureAnalyzer(node_names=scale_names)
    graph = curvature.build_graph(adjacency)
    graph = curvature.compute_forman_ricci(graph)

    neural_organism_edges = [
        edge for edge in graph["edge_curvatures"]
        if {edge["source"], edge["target"]} == {neural_idx, organism_idx}
    ]
    bottlenecks = [
        edge for edge in neural_organism_edges
        if edge["curvature"] < threshold
    ]
    bottlenecks = sorted(bottlenecks, key=lambda edge: edge["curvature"])

    all_ranked_edges = sorted(
        graph["edge_curvatures"],
        key=lambda edge: edge["curvature"],
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "Forman-Ricci curvature on simulated BAC block-Jacobian coupling network",
        "threshold": threshold,
        "scale_names": scale_names,
        "target_scales": {
            "neural_scale": "cellular",
            "organismal_scale": "organism",
            "note": "Current codebase maps neural-relevant coupling onto cellular-organismal BAC scales.",
        },
        "network": {
            "n_nodes": graph["n_nodes"],
            "n_edges": len(graph["edges"]),
            "adjacency": adjacency.tolist(),
        },
        "bottlenecks": [_clinical_edge(edge) for edge in bottlenecks],
        "top_negative_edges": [_clinical_edge(edge) for edge in all_ranked_edges[:8]],
        "clinical_interpretation": _interpret_bottlenecks(bottlenecks),
    }


def _clinical_edge(edge: Dict) -> Dict:
    """Keep report fields compact and clinically readable."""
    return {
        "source": edge["source_name"],
        "target": edge["target_name"],
        "source_index": int(edge["source"]),
        "target_index": int(edge["target"]),
        "weight": round(float(edge["weight"]), 6),
        "curvature": round(float(edge["curvature"]), 6),
        "curvature_augmented": round(float(edge["curvature_augmented"]), 6),
        "n_triangles": int(edge["n_triangles"]),
        "priority": "high" if edge["curvature"] < -0.8 else "monitor",
    }


def _interpret_bottlenecks(bottlenecks: List[Dict]) -> str:
    if not bottlenecks:
        return (
            "No cellular-organismal edges crossed the high-risk curvature "
            "threshold in this simulation."
        )
    return (
        "Highly negative cellular-organismal curvature suggests a fragile "
        "communication bridge where neural/cellular state regulation is "
        "poorly coupled to organism-level control. Prioritize biomarker panels "
        "that measure immune exhaustion, vascular integrity, ROS, and ATP."
    )


def save_report(report: Dict, path: Path = REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def save_network_plot(report: Dict, path: Path = PLOT_PATH) -> Path:
    """Draw a small directed network colored by Forman-Ricci curvature."""
    path.parent.mkdir(parents=True, exist_ok=True)

    scale_names = report["scale_names"]
    adjacency = np.array(report["network"]["adjacency"])
    graph = NetworkCurvatureAnalyzer(node_names=scale_names).build_graph(adjacency)
    graph = NetworkCurvatureAnalyzer(node_names=scale_names).compute_forman_ricci(graph)

    angles = np.linspace(0, 2 * np.pi, len(scale_names), endpoint=False)
    positions = {
        idx: np.array([np.cos(angle), np.sin(angle)])
        for idx, angle in enumerate(angles)
    }

    curvatures = np.array([edge["curvature"] for edge in graph["edge_curvatures"]])
    vmin = min(-1.0, float(np.min(curvatures)))
    vmax = max(1.0, float(np.max(curvatures)))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title("Forman-Ricci BAC Coupling Bottlenecks", fontsize=14)
    ax.axis("off")

    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    bottleneck_pairs = {
        (edge["source_index"], edge["target_index"])
        for edge in report["bottlenecks"]
    }

    for edge in graph["edge_curvatures"]:
        src = edge["source"]
        dst = edge["target"]
        start = positions[src]
        end = positions[dst]
        delta = end - start
        color = cmap(norm(edge["curvature"]))
        width = 3.5 if (src, dst) in bottleneck_pairs else 1.6
        alpha = 0.95 if (src, dst) in bottleneck_pairs else 0.55
        ax.annotate(
            "",
            xy=end - 0.14 * delta,
            xytext=start + 0.14 * delta,
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "lw": width,
                "alpha": alpha,
                "shrinkA": 6,
                "shrinkB": 6,
            },
        )

    for idx, name in enumerate(scale_names):
        x, y = positions[idx]
        face = "#1f2937" if name in {"cellular", "organism"} else "#f8fafc"
        text = "white" if face == "#1f2937" else "#111827"
        ax.scatter([x], [y], s=1650, color=face, edgecolor="#111827", linewidth=1.4, zorder=3)
        ax.text(x, y, name, ha="center", va="center", color=text, fontsize=10, zorder=4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Forman-Ricci curvature")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> Dict:
    report = compute_curvature_report()
    report_path = save_report(report)
    plot_path = save_network_plot(report)
    print(f"Saved curvature report: {report_path}")
    print(f"Saved curvature plot: {plot_path}")
    print(f"Detected bottlenecks: {len(report['bottlenecks'])}")
    return report


if __name__ == "__main__":
    main()
