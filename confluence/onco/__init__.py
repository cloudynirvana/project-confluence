"""OnCo knowledge adapter.

External, read-only client for https://onco.cc/api/v1.

This package does not write CancerODE terms. It maps OnCo identifiers
onto Confluence slots and EvidenceObjects. Data licence: CC BY-NC 4.0.
"""

from confluence.onco.attribution import ATTRIBUTION, assert_attribution
from confluence.onco.client import OncoClient, OncoClientError
from confluence.onco.mapper import Binding, bind, load_bindings
from confluence.onco.schemas import (
    ConstraintObject,
    EvidenceObject,
    HypothesisObject,
    MechanismObject,
    OncoRef,
    ParameterObject,
    PredictionObject,
    refuse_knowledge_as_parameter,
)

__all__ = [
    "ATTRIBUTION",
    "Binding",
    "ConstraintObject",
    "EvidenceObject",
    "HypothesisObject",
    "MechanismObject",
    "OncoClient",
    "OncoClientError",
    "OncoRef",
    "ParameterObject",
    "PredictionObject",
    "assert_attribution",
    "bind",
    "load_bindings",
    "refuse_knowledge_as_parameter",
]
