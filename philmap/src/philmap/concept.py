"""Core data models for philmap."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AlignmentType(Enum):
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    ARGUMENTATIVE = "argumentative"
    HYBRID = "hybrid"


@dataclass
class Tradition:
    """A philosophical tradition or school."""
    name: str
    language: str
    period: tuple[int, int] | None = None
    parent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConceptDescription:
    """A single-language description facet of a concept."""
    language: str
    term: str
    definition: str
    usage_contexts: list[str] = field(default_factory=list)
    source_texts: list[str] = field(default_factory=list)


@dataclass
class Concept:
    """A philosophical concept with multilingual descriptions."""
    id: str
    tradition: Tradition
    descriptions: list[ConceptDescription]
    related_concepts: list[str] = field(default_factory=list)
    argument_roles: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_term(self) -> str:
        for d in self.descriptions:
            if d.language == self.tradition.language:
                return d.term
        return self.descriptions[0].term


@dataclass
class AlignmentEvidence:
    """Justification for why two concepts were aligned."""
    method: AlignmentType
    score: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConceptMapping:
    """A directional mapping between two concepts."""
    source: Concept
    target: Concept
    overall_score: float
    alignment_type: AlignmentType
    evidence: list[AlignmentEvidence] = field(default_factory=list)
    notes: str = ""

    def explain(self) -> str:
        lines = [
            f"Mapping: {self.source.primary_term} -> {self.target.primary_term}",
            f"  Score: {self.overall_score:.3f} ({self.alignment_type.value})",
        ]
        for ev in self.evidence:
            lines.append(f"  [{ev.method.value}] {ev.score:.3f}: {ev.details}")
        if self.notes:
            lines.append(f"  Note: {self.notes}")
        return "\n".join(lines)


@dataclass
class ConceptDiff:
    """Result of comparing two concepts."""
    concept_a: Concept
    concept_b: Concept
    shared_aspects: list[str]
    unique_to_a: list[str]
    unique_to_b: list[str]
    similarity_by_facet: dict[str, float]
    overall_similarity: float
    narrative: str
