"""philcore - Foundation ontology and data models for the digital philosophy ecosystem."""

from philcore.types import Era, LogicFamily, MappingConfidence, RelationType
from philcore.models.concept import Concept, ConceptLabel, FormalProperty, TemporalContext
from philcore.models.argument import Argument, CatuskotiPosition, LogicalForm, Proposition
from philcore.models.tradition import Tradition
from philcore.models.thinker import Thinker
from philcore.models.text import Text, TextPassage
from philcore.models.relation import ConceptRelation, CrossTraditionMapping
from philcore.ontology.hierarchy import ConceptHierarchy
from philcore.ontology.mapping import MappingQuery, MappingRegistry
from philcore.ontology.logic import (
    BashoEnvelopment, BashoLevel, CatuskotiEvaluation, DialecticalMoment,
    Koti, ParaconsistentValuation,
)
from philcore.serialization.rdf import RDFExporter
from philcore.serialization.jsonld import concept_to_jsonld, to_jsonld_string
from philcore.registry import PhilRegistry

__version__ = "0.1.0"

__all__ = [
    "Era", "LogicFamily", "MappingConfidence", "RelationType",
    "Concept", "ConceptLabel", "FormalProperty", "TemporalContext",
    "Argument", "CatuskotiPosition", "LogicalForm", "Proposition",
    "Tradition", "Thinker", "Text", "TextPassage",
    "ConceptRelation", "CrossTraditionMapping",
    "ConceptHierarchy", "MappingQuery", "MappingRegistry",
    "BashoEnvelopment", "BashoLevel", "CatuskotiEvaluation",
    "DialecticalMoment", "Koti", "ParaconsistentValuation",
    "RDFExporter", "concept_to_jsonld", "to_jsonld_string",
    "PhilRegistry",
]
