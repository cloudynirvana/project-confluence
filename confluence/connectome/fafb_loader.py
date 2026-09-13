"""FAFB v783 loader with a realistic stub and a documented real-data hook."""

from __future__ import annotations

from typing import Optional

import numpy as np

from confluence.contracts import FlyWireNeuron, SynapseEdge
from confluence.connectome.schemas import ConnectomeGraph

# FlyWire FAFB production datastack name used by CAVEclient.
FAFB_V783_DATASTACK = "flywire_fafb_production"


class FAFBLoader:
    """Build a Kenyon-cell expansion graph.

    Real path (not executed without a token + caveclient):
        from caveclient import CAVEclient
        client = CAVEclient(datastack, auth_token=token)
        # materialization.synapse_query / cellid lookups for
        # PNs (AL/LH), KCs (γ/αβ/α'β'), MBONs, DANs, APL

    The stub reproduces published MB statistics at reduced N:
      * each KC samples ~6–7 PN axons (Caron et al. 2013; Li et al. 2020)
      * PN→KC cholinergic excitatory
      * APL is GABAergic and densely feedback-inhibitory
      * DANs are dopaminergic onto KC and MBON compartments
    """

    def __init__(self, token: Optional[str] = None, datastack: str = FAFB_V783_DATASTACK):
        self.token = token
        self.datastack = datastack

    @property
    def has_credentials(self) -> bool:
        return bool(self.token)

    def load_v783(self, n_kc: int = 256, seed: int = 783) -> ConnectomeGraph:
        if self.has_credentials:
            try:
                return self._try_caveclient(n_kc=n_kc)
            except Exception:
                # Credentials present but caveclient / materialization unavailable.
                graph = self.build_stub(n_kc=n_kc, seed=seed)
                graph.notes += " | caveclient ingestion failed; serving stub."
                return graph
        return self.build_stub(n_kc=n_kc, seed=seed)

    def _try_caveclient(self, n_kc: int) -> ConnectomeGraph:
        """Reserved real FlyWire path. Raises if caveclient is not installed."""
        try:
            from caveclient import CAVEclient  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("caveclient is not installed") from exc
        _ = CAVEclient  # real queries would go here
        raise RuntimeError(
            "Live FlyWire synapse queries are not bundled. "
            "Implement circuit_extractor over CAVEclient tables, then return a graph."
        )

    def build_stub(self, n_kc: int = 256, seed: int = 783) -> ConnectomeGraph:
        """Biologically structured stub with realistic fan-in and NT signs."""
        rng = np.random.default_rng(seed)
        n_kc = int(max(32, n_kc))
        n_pn = 20 if n_kc <= 512 else 40
        n_mbon = 8 if n_kc <= 512 else 16
        n_dan = 4 if n_kc <= 512 else 8

        graph = ConnectomeGraph(
            source="stub",
            version="FlyWire_FAFB_v783",
            materialization_version=None,
            notes=(
                f"Structured stub (n_kc={n_kc}, n_pn={n_pn}, n_mbon={n_mbon}). "
                "Fan-in ~7 PNs/KC, ACH+/GABA−/DOP neuromodulatory. "
                "Not a literal FlyWire dump — swap in FAFBLoader + CAVE_TOKEN later."
            ),
        )

        next_id = 1000000000

        def _add(cell_class: str, nt_type: str, n: int, prefix: str):
            nonlocal next_id
            ids = []
            for i in range(n):
                nid = next_id
                next_id += 1
                graph.add_neuron(
                    FlyWireNeuron(
                        root_id=nid,
                        cell_class=cell_class,
                        nt_type=nt_type,
                        proofread=False,
                        supervoxel_id=nid + 17,
                        soma_x=float(rng.uniform(0, 400_000)),
                        soma_y=float(rng.uniform(0, 400_000)),
                        soma_z=float(rng.uniform(0, 200_000)),
                        hemibrain_type=f"{prefix}_{i:03d}",
                        flow="intrinsic" if cell_class in {"KC", "APL"} else "output" if cell_class == "MBON" else "input",
                        side="left" if i % 2 == 0 else "right",
                    )
                )
                ids.append(nid)
            return ids

        pn_ids = _add("PN", "ACH", n_pn, "PN")
        kc_ids = _add("KC", "ACH", n_kc, "KC")
        mbon_ids = _add("MBON", "GLU", n_mbon, "MBON")
        # Alternate a few MBONs as GABAergic (fly MB has both)
        for i, mid in enumerate(mbon_ids):
            if i % 3 == 0:
                graph.neurons[mid].nt_type = "GABA"
        dan_ids = _add("DAN", "DOP", n_dan, "DAN")
        apl_ids = _add("APL", "GABA", 1, "APL")

        # PN → KC: sparse random, ~7 inputs / KC, cholinergic +
        fan_in = 7 if n_kc >= 64 else 5
        for kc in kc_ids:
            chosen = rng.choice(pn_ids, size=min(fan_in, len(pn_ids)), replace=False)
            for pn in chosen:
                graph.add_edge(
                    SynapseEdge(
                        pre_root_id=int(pn),
                        post_root_id=int(kc),
                        weight=float(rng.uniform(0.4, 1.2)),
                        nt_type="ACH",
                        sign=1,
                        neuropil="MB_calyx",
                    )
                )

        # KC → MBON: moderate expansion, cholinergic/glutamatergic +
        kc_per_mbon = max(8, n_kc // n_mbon)
        for mbon in mbon_ids:
            chosen = rng.choice(kc_ids, size=min(kc_per_mbon, len(kc_ids)), replace=False)
            for kc in chosen:
                graph.add_edge(
                    SynapseEdge(
                        pre_root_id=int(kc),
                        post_root_id=int(mbon),
                        weight=float(rng.uniform(0.05, 0.35)),
                        nt_type="ACH",
                        sign=1,
                        neuropil="MB_lobe",
                    )
                )

        # APL feedback: KC → APL → KC (GABA −)
        apl = apl_ids[0]
        for kc in kc_ids:
            if rng.random() < 0.35:
                graph.add_edge(
                    SynapseEdge(
                        pre_root_id=int(kc),
                        post_root_id=int(apl),
                        weight=float(rng.uniform(0.02, 0.08)),
                        nt_type="ACH",
                        sign=1,
                        neuropil="MB_lobe",
                    )
                )
            graph.add_edge(
                SynapseEdge(
                    pre_root_id=int(apl),
                    post_root_id=int(kc),
                    weight=float(rng.uniform(0.15, 0.45)),
                    nt_type="GABA",
                    sign=-1,
                    neuropil="MB_calyx",
                )
            )

        # DAN → KC / MBON dopaminergic (modulatory; stored as + for DA eligibility)
        for dan in dan_ids:
            for kc in rng.choice(kc_ids, size=min(32, len(kc_ids)), replace=False):
                graph.add_edge(
                    SynapseEdge(
                        pre_root_id=int(dan),
                        post_root_id=int(kc),
                        weight=float(rng.uniform(0.05, 0.2)),
                        nt_type="DOP",
                        sign=1,
                        neuropil="MB_lobe",
                    )
                )
            for mbon in mbon_ids:
                graph.add_edge(
                    SynapseEdge(
                        pre_root_id=int(dan),
                        post_root_id=int(mbon),
                        weight=float(rng.uniform(0.05, 0.2)),
                        nt_type="DOP",
                        sign=1,
                        neuropil="MB_lobe",
                    )
                )

        return graph
