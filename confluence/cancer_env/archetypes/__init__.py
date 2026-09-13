"""Cancer archetype parameter sets (tissue + disease-class modes)."""

from confluence.cancer_env.archetypes.glioblastoma import GlioblastomaArchetype
from confluence.cancer_env.archetypes.melanoma_persister import MelanomaPersisterArchetype
from confluence.cancer_env.archetypes.pancreatic_pdac import PancreaticPDACArchetype
from confluence.cancer_env.archetypes.taxonomy import (
    BenignArchetype,
    DormantArchetype,
    MalignantClassArchetype,
    OccultArchetype,
    TerminalArchetype,
)
from confluence.cancer_env.ode_system import ArchetypeParams

ARCHETYPES = {
    "glioblastoma": GlioblastomaArchetype,
    "pancreatic_pdac": PancreaticPDACArchetype,
    "melanoma_persister": MelanomaPersisterArchetype,
    "benign": BenignArchetype,
    "malignant": MalignantClassArchetype,
    "occult": OccultArchetype,
    "dormant": DormantArchetype,
    "terminal": TerminalArchetype,
}

DISPLAY_NAMES = {
    "glioblastoma": "Glioblastoma",
    "pancreatic_pdac": "PDAC / pancreatic",
    "melanoma_persister": "Melanoma (persister)",
    "benign": "Benign (growth-limited)",
    "malignant": "Malignant (aggressive)",
    "occult": "Occult / hidden",
    "dormant": "Dormant (stochastic wake)",
    "terminal": "Terminal (host-failure)",
}


def get_archetype(name: str) -> ArchetypeParams:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "gbm": "glioblastoma",
        "pdac": "pancreatic_pdac",
        "pancreatic": "pancreatic_pdac",
        "melanoma": "melanoma_persister",
        "hidden": "occult",
        "occult_hidden": "occult",
        "dormancy": "dormant",
        "host_failure": "terminal",
    }
    key = aliases.get(key, key)
    if key not in ARCHETYPES:
        raise KeyError(f"Unknown archetype '{name}'. Choose from {list(ARCHETYPES)}")
    return ARCHETYPES[key]()


__all__ = [
    "ARCHETYPES",
    "DISPLAY_NAMES",
    "GlioblastomaArchetype",
    "MelanomaPersisterArchetype",
    "PancreaticPDACArchetype",
    "BenignArchetype",
    "MalignantClassArchetype",
    "OccultArchetype",
    "DormantArchetype",
    "TerminalArchetype",
    "get_archetype",
]
