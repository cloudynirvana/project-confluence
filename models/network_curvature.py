"""
Network Curvature Module
========================

Computes discrete Ricci curvature on the metabolic-immune generator graph
to identify structural vulnerabilities (fragile edges/bottlenecks).

While Ollivier-Ricci requires solving optimal transport problems, we use
Forman-Ricci curvature here for efficient computation on directed weighted graphs,
which serves the same biological purpose: finding structurally critical nodes.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkCurvatureAnalyzer:
    """
    Constructs a graph from the ODE generator matrix and computes edge curvature.
    """
    
    def __init__(self, node_names: List[str] = None):
        self.node_names = node_names

    def build_graph(self, A_matrix: np.ndarray) -> Dict:
        """
        Converts the generator matrix into a weighted directed graph representation.
        Nodes are state variables, edges are non-zero coupling strengths.
        """
        n_nodes = A_matrix.shape[0]
        if self.node_names is None:
            self.node_names = [f"Node_{i}" for i in range(n_nodes)]
            
        nodes = list(range(n_nodes))
        edges = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j and abs(A_matrix[i, j]) > 1e-10:
                    edges.append({
                        'source': j,  # influence flows from j to i
                        'target': i,
                        'weight': abs(A_matrix[i, j]),
                        'sign': np.sign(A_matrix[i, j])
                    })
                    
        return {'nodes': nodes, 'edges': edges, 'adj': A_matrix}

    def compute_forman_ricci(self, graph: Dict) -> Dict:
        """
        Computes Forman-Ricci curvature for each edge.
        Forman-Ricci looks at parallel edges. For weighted directed networks,
        F(e) = w(e) - sum_{e_in} w(e_in)/sqrt(w(e)*w(e_in)) - sum_{e_out}...
        Simplified heuristic: Highly central edges connecting dense clusters
        have positive curvature, bridging edges (bottlenecks) have negative.
        """
        edges = graph['edges']
        adj = graph['adj']
        n_nodes = len(graph['nodes'])
        
        # Out-degree and in-degree strengths (sum of absolute weights)
        in_strength = np.sum(np.abs(adj), axis=1) - np.diag(np.abs(adj))
        out_strength = np.sum(np.abs(adj), axis=0) - np.diag(np.abs(adj))
        
        edge_curvatures = []
        
        for e in edges:
            u, v = e['source'], e['target']
            w_uv = e['weight']
            
            # Simplified Forman-Ricci formulation for directed edges
            # F(e) = 2 - In_degree(u) - Out_degree(v) (unweighted basis)
            # Weighted: F(e) = 2*w(e) - sum(w(parallel_edges))
            
            # Nodes connected to u (excluding v)
            u_neighbors_out = out_strength[u] - w_uv
            u_neighbors_in = in_strength[u]
            
            # Nodes connected to v (excluding u)
            v_neighbors_in = in_strength[v] - w_uv
            v_neighbors_out = out_strength[v]
            
            # Edge curvature is lower (more negative) if nodes are highly connected to OTHERS
            # but not to each other (bridge-like)
            curvature = 2 * w_uv - (u_neighbors_in + u_neighbors_out + v_neighbors_in + v_neighbors_out)
            
            edge_curvatures.append({
                'source_name': self.node_names[u] if self.node_names else u,
                'target_name': self.node_names[v] if self.node_names else v,
                'source': u,
                'target': v,
                'weight': w_uv,
                'sign': e['sign'],
                'curvature': float(curvature)
            })
            
        graph['edge_curvatures'] = edge_curvatures
        return graph

    def identify_bottlenecks(self, graph: Dict, top_k: int = 5) -> List[Dict]:
        """
        Returns the edges with the most negative curvature (structural bottlenecks).
        These are primary drug target candidates.
        """
        if 'edge_curvatures' not in graph:
            graph = self.compute_forman_ricci(graph)
            
        curvatures = graph['edge_curvatures']
        # Sort by curvature ascending (most negative first)
        bottlenecks = sorted(curvatures, key=lambda x: x['curvature'])
        
        return bottlenecks[:top_k]

    def curvature_difference(self, graph_cancer: Dict, graph_healthy: Dict) -> List[Dict]:
        """
        Identifies edges whose curvature shifts the most between healthy and cancer.
        """
        c_curv = { (e['source'], e['target']): e['curvature'] for e in graph_cancer.get('edge_curvatures', []) }
        h_curv = { (e['source'], e['target']): e['curvature'] for e in graph_healthy.get('edge_curvatures', []) }
        
        shifts = []
        for edge, c_val in c_curv.items():
            if edge in h_curv:
                h_val = h_curv[edge]
                shifts.append({
                    'source': edge[0],
                    'target': edge[1],
                    'cancer_curvature': c_val,
                    'healthy_curvature': h_val,
                    'shift': c_val - h_val
                })
                
        # Sort by largest shift magnitude
        return sorted(shifts, key=lambda x: abs(x['shift']), reverse=True)
