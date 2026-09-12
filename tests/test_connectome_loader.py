"""Smoke tests for the FlyWire stub loader and circuit extractor."""

from confluence.connectome.circuit_extractor import CircuitExtractor
from confluence.connectome.client_stub import FlyWireClient
from confluence.connectome.fafb_loader import FAFBLoader
from confluence.contracts import ConnectomeSubcircuit


def test_stub_graph_has_mb_classes():
    graph = FAFBLoader().build_stub(n_kc=64, seed=1)
    assert len(graph.by_class("PN")) >= 8
    assert len(graph.by_class("KC")) == 64
    assert len(graph.by_class("MBON")) >= 4
    assert len(graph.by_class("DAN")) >= 2
    assert len(graph.by_class("APL")) == 1
    assert graph.edges
    nts = {e.nt_type for e in graph.edges}
    assert {"ACH", "GABA", "DOP"} <= nts
    signs = {e.sign for e in graph.edges}
    assert 1 in signs and -1 in signs


def test_extractor_schema_and_compile():
    graph = FlyWireClient().fetch_mushroom_body(n_kc=48)
    sub = CircuitExtractor().extract(graph)
    assert isinstance(sub, ConnectomeSubcircuit)
    assert sub.version == "FlyWire_FAFB_v783"
    assert sub.source == "stub"
    assert sub.n_kc == 48
    compiled = CircuitExtractor().compile(graph)
    assert compiled.w_pn_kc.shape[0] == 48
    assert compiled.w_kc_mbon.shape[1] == 48


def test_client_without_token_uses_stub():
    client = FlyWireClient(token=None)
    assert client.using_stub
    graph = client.fetch_mushroom_body(n_kc=32)
    assert graph.source == "stub"


def test_loader_with_fake_token_still_returns_graph():
    loader = FAFBLoader(token="not-a-real-cave-token")
    assert loader.has_credentials
    graph = loader.load_v783(n_kc=32)
    assert graph.by_class("KC")
    assert graph.source == "stub"
