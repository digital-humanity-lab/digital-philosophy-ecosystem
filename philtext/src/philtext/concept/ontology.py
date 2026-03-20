"""Philosophical concept ontology."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class ConceptNode:
    """A node in the philosophical concept ontology."""
    id: str
    labels: dict[str, str]
    alt_labels: dict[str, list[str]] = field(default_factory=dict)
    definition: str = ""
    broader: list[str] = field(default_factory=list)
    narrower: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    school_associations: list[str] = field(default_factory=list)

    def label(self, lang: str = "en") -> str:
        return self.labels.get(lang, self.labels.get("en", self.id))


@dataclass
class ConceptMention:
    """A concept mention found in text."""
    concept: ConceptNode
    surface_form: str
    span: tuple[int, int]
    confidence: float
    context_sentence: str
    usage_sense: str = ""


class PhilOntology:
    """In-memory philosophical concept ontology."""

    def __init__(self) -> None:
        self._concepts: dict[str, ConceptNode] = {}

    def add(self, node: ConceptNode) -> None:
        self._concepts[node.id] = node

    def get(self, concept_id: str) -> ConceptNode | None:
        return self._concepts.get(concept_id)

    def all_concepts(self) -> Iterator[ConceptNode]:
        yield from self._concepts.values()

    def search_label(self, text: str, lang: str | None = None) -> list[ConceptNode]:
        text_lower = text.lower()
        results = []
        for node in self._concepts.values():
            for l, label in node.labels.items():
                if lang and l != lang:
                    continue
                if text_lower in label.lower():
                    results.append(node)
                    break
            else:
                for l, alts in node.alt_labels.items():
                    if lang and l != lang:
                        continue
                    if any(text_lower in a.lower() for a in alts):
                        results.append(node)
                        break
        return results

    @classmethod
    def load_default(cls) -> PhilOntology:
        """Return an empty ontology (seed data can be added later)."""
        return cls()

    def __len__(self) -> int:
        return len(self._concepts)
