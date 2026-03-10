# -*- coding: utf-8 -*-
"""
Digital Twin Memory Adapters — Project Confluence
=================================================

Provides isolated API clients for Graphiti (Temporal Knowledge Graphs)
and Cognee (Self-Improving Relational Graphs) to manage longitudinal 
patient records.

This module acts as the "Memory Layer" between the continuous-time Neural ODE
and the discrete archetype ML classifier, discretizing the trajectory $z(t)$
as proposed by the local Qwen architecture.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple

# Set up logging
logger = logging.getLogger("ConfluenceMemory")
logger.setLevel(logging.INFO)

# ═══════════════════════════════════════════════════════════════════════════
# TEMPORAL GRAPH (GRAPHITI ADAPTER)
# ═══════════════════════════════════════════════════════════════════════════

class GraphitiAdapter:
    """
    Adapter for Zep's Graphiti.
    Focuses on Φ_temporal: Ingests sampled, discrete 'episodes' from the
    Neural ODE trajectory to model evolving state transitions over time.
    """
    def __init__(self, endpoint: str = "local"):
        self.endpoint = endpoint
        self._mock_graph = []
        logger.info(f"Initialized Graphiti Temporal Graph Adapter at {self.endpoint}")

    def ingest_trajectory_events(self, z_t: np.ndarray, time_points: np.ndarray, 
                                 entropy_threshold: float = 0.5):
        """
        Discretizes the Neural ODE trajectory into specific temporal events
        based on high-curvature or shifts in multiscale entropy.

        Args:
            z_t: (n_steps, 15) numpy array of simulated state
            time_points: (n_steps,) simulation time
            entropy_threshold: threshold to identify an "event"
        """
        events = []
        # A simple curvature check to identify non-linear state shifts
        for i in range(1, len(z_t) - 1):
            grad1 = z_t[i] - z_t[i-1]
            grad2 = z_t[i+1] - z_t[i]
            curvature = np.linalg.norm(grad2 - grad1)
            
            if curvature > entropy_threshold:
                event = {
                    "valid_time": time_points[i],  # Simulation t
                    "transaction_time": 0.0,       # Ingestion current t
                    "state_vector": z_t[i].tolist(),
                    "type": "curvature_shift"
                }
                events.append(event)
                
        self._mock_graph.extend(events)
        logger.info(f"Graphiti adapter ingested {len(events)} discrete non-linear transitions.")
        return len(events)
        
    def query_temporal_sequence(self) -> List[Dict]:
        """
        Retrieves the historical sequence of transitions leading up to a
        potential collapse.
        """
        # In actual Graphiti query: Match (e:Event) return e ORDER BY e.valid_time
        return sorted(self._mock_graph, key=lambda x: x["valid_time"])

    def get_event_summary(self) -> Dict[str, float]:
        """Return a summary of temporal shift events."""
        events = self._mock_graph
        if not events:
            return {"total_events": 0.0, "event_frequency": 0.0}
            
        total_events = float(len(events))
        if total_events > 1:
            times = sorted([e["valid_time"] for e in events])
            duration = times[-1] - times[0]
            freq = total_events / duration if duration > 0 else 0.0
        else:
            freq = 0.0
            
        return {
            "total_events": total_events,
            "event_frequency": float(freq)
        }


# ═══════════════════════════════════════════════════════════════════════════
# STRUCTURAL GRAPH (COGNEE ADAPTER)
# ═══════════════════════════════════════════════════════════════════════════

class CogneeAdapter:
    """
    Adapter for Cognee framework.
    Focuses on Φ_coupling and Φ_informational: Self-improves mappings between
    the metabolic parameters and the 2D immune state.
    """
    def __init__(self, use_vector_store: bool = False):
        self.relations = {}
        logger.info("Initialized Cognee Structural/Relational Graph Adapter.")
        
    def _symbolize_layer(self, state: np.ndarray) -> List[str]:
        """
        Converts the dense 15D state variable into symbolic nodes.
        E.g., mapping z[0] (Glucose) > 10.0 -> 'Hyperglycemia'
        """
        symbols = []
        if state[0] > 10.0: symbols.append("High Glucose Metabolism")
        if state[1] > 8.0: symbols.append("Hyperlactatemia")
        if state[13] > 0.8: symbols.append("High Stromal Rigidity")
        # Immune Exhaustion (z[12])
        if state[12] > 5.0: symbols.append("T-Cell Exhaustion")
            
        return symbols

    def extract_and_map_relations(self, snapshot: np.ndarray, t: float):
        """
        Extracts symbolic entities from a time snapshot and maps structural
        correlations (e.g., High Stromal Rigidity -> T-Cell Exhaustion)
        """
        symbols = self._symbolize_layer(snapshot)
        
        # Build relational edges
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                edge = f"{symbols[i]} --correlated_with--> {symbols[j]}"
                if edge not in self.relations:
                    self.relations[edge] = {"weight": 1.0, "time_first_observed": t}
                else:
                    self.relations[edge]["weight"] += 0.1
                    
        logger.debug(f"Cognee mapped {len(symbols)} structural nodes at t={t}")

    def get_strongest_correlations(self) -> Dict:
        """Returns the self-improved structural relations over time."""
        return {k: v for k, v in sorted(self.relations.items(), 
                                        key=lambda item: item[1]['weight'], reverse=True)}

    def get_network_properties(self) -> Dict[str, float]:
        """Compute network graph properties from extracted relations."""
        if not self.relations:
            return {
                "total_edges": 0.0,
                "total_weight": 0.0,
                "max_weight": 0.0,
                "edge_density": 0.0,
                "average_weight": 0.0,
            }
            
        weights = [v["weight"] for v in self.relations.values()]
        total_edges = float(len(weights))
        total_weight = float(np.sum(weights))
        max_weight = float(np.max(weights))
        average_weight = total_weight / total_edges
        
        # Approximate unique nodes from edge strings (A --correlated_with--> B)
        nodes = set()
        for edge_str in self.relations.keys():
            parts = edge_str.split(" --correlated_with--> ")
            if len(parts) == 2:
                nodes.add(parts[0])
                nodes.add(parts[1])
                
        n_nodes = max(len(nodes), 2)
        max_possible_edges = (n_nodes * (n_nodes - 1)) / 2
        edge_density = total_edges / max_possible_edges if max_possible_edges > 0 else 0.0
        
        # Calculate structural entropy (Shannon entropy over edge weights)
        weights_array = np.array(weights)
        prob = weights_array / total_weight if total_weight > 0 else np.array([1.0])
        structural_entropy = -np.sum(prob * np.log2(prob + 1e-9))
        
        return {
            "total_edges": total_edges,
            "total_weight": total_weight,
            "max_weight": max_weight,
            "edge_density": edge_density,
            "average_weight": average_weight,
            "structural_entropy": float(structural_entropy),
        }


# ═══════════════════════════════════════════════════════════════════════════
# DIGITAL TWIN MEMORY CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════

class DigitalTwinMemory:
    """
    Acts as the main hub connecting PatientFitter output to targeted
    Graph retrieval databases.
    """
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.temporal_memory = GraphitiAdapter()
        self.structural_memory = CogneeAdapter()
        
    def process_neural_trajectory(self, trajectory: np.ndarray, time_span: np.ndarray):
        """
        Takes raw continuous [T, 15] data from ComplexityNeuralODE and discretizes it.
        """
        # 1. Update temporal history of shifts
        self.temporal_memory.ingest_trajectory_events(trajectory, time_span)
        
        # 2. Update structural coupling map iteratively over the trajectory
        # Sample every 10 steps to reduce Cognee bloat
        for i in range(0, len(time_span), 10):
            self.structural_memory.extract_and_map_relations(trajectory[i], time_span[i])
            
        return {
            "status": "success",
            "temporal_events_recorded": len(self.temporal_memory._mock_graph),
            "unique_correlations_mapped": len(self.structural_memory.relations)
        }

    def get_memory_features(self) -> Dict[str, float]:
        """Aggregate temporal and structural memory into a feature dictionary."""
        features = {}
        # Structural 
        struct = self.structural_memory.get_network_properties()
        for k, v in struct.items():
            features[f"structural_{k}"] = v
            
        # Temporal
        temp = self.temporal_memory.get_event_summary()
        for k, v in temp.items():
            features[f"temporal_{k}"] = v
            
        return features

