"""Philosophical data models."""

from philcore.models.concept import Concept, ConceptLabel, FormalProperty, TemporalContext
from philcore.models.argument import Argument, CatuskotiPosition, LogicalForm, Proposition
from philcore.models.tradition import Tradition
from philcore.models.thinker import Thinker
from philcore.models.text import Text, TextPassage
from philcore.models.relation import ConceptRelation, CrossTraditionMapping

__all__ = [
    "Concept", "ConceptLabel", "FormalProperty", "TemporalContext",
    "Argument", "CatuskotiPosition", "LogicalForm", "Proposition",
    "Tradition", "Thinker", "Text", "TextPassage",
    "ConceptRelation", "CrossTraditionMapping",
]
