"""Lightweight graph container used by loaders and the circuit extractor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from confluence.contracts import ConnectomeSubcircuit, FlyWireNeuron, SynapseEdge


@dataclass
class ConnectomeGraph:
    """Directed signed graph with FlyWire_FAFB_v783-compatible metadata."""

    neurons: Dict[int, FlyWireNeuron] = field(default_factory=dict)
    edges: List[SynapseEdge] = field(default_factory=list)
    source: str = "stub"
    version: str = "FlyWire_FAFB_v783"
    materialization_version: Optional[int] = None
    notes: str = ""

    def add_neuron(self, neuron: FlyWireNeuron) -> None:
        self.neurons[neuron.root_id] = neuron

    def add_edge(self, edge: SynapseEdge) -> None:
        self.edges.append(edge)

    def by_class(self, cell_class: str) -> List[FlyWireNeuron]:
        return [n for n in self.neurons.values() if n.cell_class == cell_class]

    def to_subcircuit(self) -> ConnectomeSubcircuit:
        return ConnectomeSubcircuit(
            version=self.version,
            source=self.source,
            n_pn=len(self.by_class("PN")),
            n_kc=len(self.by_class("KC")),
            n_mbon=len(self.by_class("MBON")),
            n_dan=len(self.by_class("DAN")),
            n_apl=len(self.by_class("APL")),
            neurons=list(self.neurons.values()),
            edges=list(self.edges),
            notes=self.notes,
            materialization_version=self.materialization_version,
        )
