"""Ontology tools: hierarchy, mapping, and non-classical logic support."""

from philcore.ontology.hierarchy import ConceptHierarchy
from philcore.ontology.mapping import MappingQuery, MappingRegistry
from philcore.ontology.logic import (
    BashoEnvelopment, BashoLevel, CatuskotiEvaluation, DialecticalMoment,
    Koti, ParaconsistentValuation,
)

__all__ = [
    "ConceptHierarchy", "MappingQuery", "MappingRegistry",
    "BashoEnvelopment", "BashoLevel", "CatuskotiEvaluation",
    "DialecticalMoment", "Koti", "ParaconsistentValuation",
]
