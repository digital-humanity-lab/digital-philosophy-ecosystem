"""philmap - Cross-cultural philosophical concept alignment and mapping."""

from philmap.concept import (
    AlignmentType, Concept, ConceptDescription, ConceptDiff,
    ConceptMapping, Tradition, AlignmentEvidence,
)
from philmap.alignment.base import AlignmentMethod
from philmap.analysis.analogues import find_analogues
from philmap.analysis.bridge import tradition_bridge
from philmap.analysis.genealogy import GenealogyNode, concept_genealogy

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazy imports for modules requiring sentence-transformers."""
    _lazy = {
        "ConceptEmbedder": "philmap.embedding.embedder",
        "EmbeddingConfig": "philmap.embedding.embedder",
        "FacetedEmbedding": "philmap.embedding.embedder",
        "SemanticAlignment": "philmap.alignment.semantic",
        "StructuralAlignment": "philmap.alignment.structural",
        "TraditionOntology": "philmap.alignment.structural",
        "ArgumentativeAlignment": "philmap.alignment.argumentative",
        "ArgumentSchema": "philmap.alignment.argumentative",
        "ArgumentRole": "philmap.alignment.argumentative",
        "HybridAlignment": "philmap.alignment.hybrid",
        "concept_diff": "philmap.analysis.diff",
    }
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'philmap' has no attribute {name!r}")


__all__ = [
    "AlignmentType", "Concept", "ConceptDescription", "ConceptDiff",
    "ConceptMapping", "Tradition", "AlignmentEvidence",
    "ConceptEmbedder", "EmbeddingConfig", "FacetedEmbedding",
    "AlignmentMethod", "SemanticAlignment",
    "StructuralAlignment", "TraditionOntology",
    "ArgumentativeAlignment", "ArgumentSchema", "ArgumentRole",
    "HybridAlignment",
    "find_analogues", "concept_diff", "tradition_bridge",
    "GenealogyNode", "concept_genealogy",
]
