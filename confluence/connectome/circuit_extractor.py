"""Extract a mushroom-body subcircuit and compile rate-network weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from confluence.contracts import ConnectomeSubcircuit
from confluence.connectome.schemas import ConnectomeGraph


@dataclass
class CompiledCircuit:
    """Dense matrices for the rate-based MB engine."""

    pn_ids: List[int]
    kc_ids: List[int]
    mbon_ids: List[int]
    dan_ids: List[int]
    w_pn_kc: np.ndarray
    w_kc_mbon: np.ndarray
    w_apl_kc: np.ndarray
    signs_mbon: np.ndarray
    subcircuit: ConnectomeSubcircuit


class CircuitExtractor:
    """Turn a ConnectomeGraph into a ConnectomeSubcircuit + weight matrices."""

    def extract(self, graph: ConnectomeGraph) -> ConnectomeSubcircuit:
        return graph.to_subcircuit()

    def compile(self, graph: ConnectomeGraph) -> CompiledCircuit:
        pn = [n.root_id for n in graph.by_class("PN")]
        kc = [n.root_id for n in graph.by_class("KC")]
        mbon = [n.root_id for n in graph.by_class("MBON")]
        dan = [n.root_id for n in graph.by_class("DAN")]
        pn_ix = {i: k for k, i in enumerate(pn)}
        kc_ix = {i: k for k, i in enumerate(kc)}
        mbon_ix = {i: k for k, i in enumerate(mbon)}

        w_pn_kc = np.zeros((len(kc), len(pn)), dtype=float)
        w_kc_mbon = np.zeros((len(mbon), len(kc)), dtype=float)
        w_apl_kc = np.zeros(len(kc), dtype=float)
        signs = np.ones(len(mbon), dtype=float)

        for i, mid in enumerate(mbon):
            nt = graph.neurons[mid].nt_type
            signs[i] = -1.0 if nt == "GABA" else 1.0

        for edge in graph.edges:
            if edge.pre_root_id in pn_ix and edge.post_root_id in kc_ix:
                w_pn_kc[kc_ix[edge.post_root_id], pn_ix[edge.pre_root_id]] += edge.sign * edge.weight
            elif edge.pre_root_id in kc_ix and edge.post_root_id in mbon_ix:
                w_kc_mbon[mbon_ix[edge.post_root_id], kc_ix[edge.pre_root_id]] += edge.sign * edge.weight
            elif graph.neurons.get(edge.pre_root_id) and graph.neurons[edge.pre_root_id].cell_class == "APL":
                if edge.post_root_id in kc_ix:
                    w_apl_kc[kc_ix[edge.post_root_id]] += abs(edge.weight)

        return CompiledCircuit(
            pn_ids=pn,
            kc_ids=kc,
            mbon_ids=mbon,
            dan_ids=dan,
            w_pn_kc=w_pn_kc,
            w_kc_mbon=w_kc_mbon,
            w_apl_kc=w_apl_kc,
            signs_mbon=signs,
            subcircuit=graph.to_subcircuit(),
        )
