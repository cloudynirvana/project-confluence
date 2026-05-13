"""
Network Curvature Module — Project Confluence
===============================================

Computes discrete Ricci curvature on the metabolic-immune generator graph
to identify structural vulnerabilities (fragile edges / bottlenecks).

Two curvature measures are provided:

1. **Forman-Ricci Curvature (FRC)** — Combinatorial curvature based on
   edge weights and neighbour degree.  Computationally cheap (O(E)).
2. **Augmented Forman-Ricci** — Extends FRC by incorporating triangles
   (3-cliques) that pass through each edge, providing a richer
   geometric signal on small networks.

Interpretation (Sandhu et al., 2015):
    Positive curvature → dense, robust, redundant connectivity
    Negative curvature → fragile bridge / information bottleneck

References:
    Sandhu et al. (2015) - Graph curvature for differentiating cancer networks
    Pouryahya et al. (2024) - ORCO tool for biological network robustness
    Ni et al. (2015) - Ricci curvature of the Internet topology
"""

import numpy as np
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default metabolite names from ode_system.py
_DEFAULT_METABOLITE_NAMES = [
    "Glucose", "Lactate", "Pyruvate", "ATP", "NADH",
    "Glutamine", "Glutamate", "aKG", "Citrate", "ROS",
]
_DEFAULT_STATE_NAMES = _DEFAULT_METABOLITE_NAMES + [
    "I_eff", "I_reg", "I_exhaust", "sigma_stromal", "nu_vascular",
]


class NetworkCurvatureAnalyzer:
    """
    Constructs a weighted directed graph from the ODE generator matrix
    and computes edge-level Forman-Ricci curvature.
    """

    def __init__(self, node_names: Optional[List[str]] = None):
        """
        Parameters
        ----------
        node_names : list of str, optional
            Human-readable labels for each node.  Defaults to the 15D
            STATE_NAMES from ode_system.py when the matrix is 10×10 or
            15×15.
        """
        self.node_names = node_names

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def build_graph(self, A_matrix: np.ndarray) -> Dict:
        """
        Convert a generator matrix into a weighted directed graph.

        Parameters
        ----------
        A_matrix : ndarray, shape (N, N)
            Generator or adjacency matrix.  A[i, j] != 0 means an
            influence from variable j onto variable i.

        Returns
        -------
        graph : dict
            Keys: 'nodes', 'edges', 'adj', 'n_nodes'.
        """
        n = A_matrix.shape[0]

        # Auto-assign names
        if self.node_names is None or len(self.node_names) != n:
            if n <= 10:
                self.node_names = _DEFAULT_METABOLITE_NAMES[:n]
            elif n <= 15:
                self.node_names = _DEFAULT_STATE_NAMES[:n]
            else:
                self.node_names = [f"Node_{i}" for i in range(n)]

        edges = []
        for i in range(n):
            for j in range(n):
                if i != j and abs(A_matrix[i, j]) > 1e-10:
                    edges.append({
                        "source": j,
                        "target": i,
                        "source_name": self.node_names[j],
                        "target_name": self.node_names[i],
                        "weight": abs(A_matrix[i, j]),
                        "sign": int(np.sign(A_matrix[i, j])),
                    })

        return {
            "nodes": list(range(n)),
            "edges": edges,
            "adj": A_matrix.copy(),
            "n_nodes": n,
        }

    # ------------------------------------------------------------------
    # Forman-Ricci curvature
    # ------------------------------------------------------------------
    def compute_forman_ricci(self, graph: Dict) -> Dict:
        """
        Compute the Forman-Ricci curvature for every edge.

        For a directed weighted edge e = (u → v) with weight w(e):

            F(e) = w(e) ·[ 2/w(e)
                           − Σ_{e' ∈ star(u)\\e}  w(e) / √(w(e)·w(e'))
                           − Σ_{e' ∈ star(v)\\e}  w(e) / √(w(e)·w(e'))
                           + Σ_{triangles through e} ... ]

        We use the simplified combinatorial form:

            F(e) = w(e) · [2 − d_out(u) − d_in(v)]

        where d_out(u) = weighted out-degree of u, d_in(v) = weighted
        in-degree of v (excluding the edge e itself).
        """
        adj = graph["adj"]
        n = graph["n_nodes"]
        abs_adj = np.abs(adj)

        # Weighted in-degree and out-degree (excluding self-loops)
        in_deg = np.sum(abs_adj, axis=1) - np.diag(abs_adj)
        out_deg = np.sum(abs_adj, axis=0) - np.diag(abs_adj)

        # Count triangles through each edge for augmented curvature
        # triangle_count[u,v] = number of nodes w such that (u->w) and (w->v)
        binary = (abs_adj > 1e-10).astype(float)
        triangle_matrix = binary @ binary  # triangle_matrix[i,j] = #2-paths i→?→j

        edge_curvatures = []
        for e in graph["edges"]:
            u, v = e["source"], e["target"]
            w_e = e["weight"]

            # Basic Forman-Ricci
            frc = w_e * (2.0 - out_deg[u] - in_deg[v])

            # Triangle augmentation: each triangle adds +1 to curvature
            n_triangles = triangle_matrix[u, v]
            frc_aug = frc + w_e * n_triangles

            e_result = {
                **e,
                "curvature": float(frc),
                "curvature_augmented": float(frc_aug),
                "n_triangles": int(n_triangles),
            }
            edge_curvatures.append(e_result)

        graph["edge_curvatures"] = edge_curvatures
        return graph

    # ------------------------------------------------------------------
    # Bottleneck identification
    # ------------------------------------------------------------------
    def identify_bottlenecks(self, graph: Dict, top_k: int = 5) -> List[Dict]:
        """
        Return the top-k edges with the most negative curvature
        (structural bottlenecks → high-priority drug targets).
        """
        if "edge_curvatures" not in graph:
            graph = self.compute_forman_ricci(graph)
        ranked = sorted(graph["edge_curvatures"], key=lambda e: e["curvature"])
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Cancer vs Healthy curvature shift
    # ------------------------------------------------------------------
    def curvature_difference(self, graph_cancer: Dict, graph_healthy: Dict) -> List[Dict]:
        """
        Identify edges whose curvature shifts the most between healthy
        and cancer.  Large negative shifts indicate edges that became
        more fragile in the disease state.
        """
        if "edge_curvatures" not in graph_cancer:
            graph_cancer = self.compute_forman_ricci(graph_cancer)
        if "edge_curvatures" not in graph_healthy:
            graph_healthy = self.compute_forman_ricci(graph_healthy)

        c_map = {(e["source"], e["target"]): e for e in graph_cancer["edge_curvatures"]}
        h_map = {(e["source"], e["target"]): e for e in graph_healthy["edge_curvatures"]}

        shifts = []
        for key, c_edge in c_map.items():
            if key in h_map:
                h_edge = h_map[key]
                shift = c_edge["curvature"] - h_edge["curvature"]
                shifts.append({
                    "source": key[0],
                    "target": key[1],
                    "source_name": c_edge.get("source_name", key[0]),
                    "target_name": c_edge.get("target_name", key[1]),
                    "cancer_curvature": c_edge["curvature"],
                    "healthy_curvature": h_edge["curvature"],
                    "shift": shift,
                    "interpretation": "more fragile" if shift < 0 else "more robust",
                })

        return sorted(shifts, key=lambda s: s["shift"])

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    def generate_report(self, graph: Dict, top_k: int = 5) -> str:
        """Human-readable curvature analysis report."""
        if "edge_curvatures" not in graph:
            graph = self.compute_forman_ricci(graph)

        bottlenecks = self.identify_bottlenecks(graph, top_k=top_k)

        lines = [
            "=" * 60,
            "NETWORK CURVATURE ANALYSIS REPORT",
            "=" * 60,
            f"Nodes: {graph['n_nodes']}   Edges: {len(graph['edges'])}",
            "",
            f"TOP {top_k} STRUCTURAL BOTTLENECKS (most negative curvature):",
        ]
        for i, b in enumerate(bottlenecks):
            lines.append(
                f"  {i+1}. {b['source_name']} → {b['target_name']}  "
                f"(F = {b['curvature']:.3f}, w = {b['weight']:.3f})"
            )

        # Node-level summary: average curvature of edges touching each node
        node_curv = {}
        for e in graph["edge_curvatures"]:
            for n in [e["source"], e["target"]]:
                node_curv.setdefault(n, []).append(e["curvature"])
        lines.append("")
        lines.append("NODE-LEVEL AVERAGE CURVATURE:")
        for n in sorted(node_curv.keys()):
            avg = np.mean(node_curv[n])
            name = self.node_names[n] if n < len(self.node_names) else f"Node_{n}"
            lines.append(f"  {name:20s}  avg F = {avg:+.3f}")

        lines.append("=" * 60)
        return "\n".join(lines)
