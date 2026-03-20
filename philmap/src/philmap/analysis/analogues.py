"""Find analogues of a concept in another tradition."""

from __future__ import annotations

from philmap.alignment.base import AlignmentMethod
from philmap.concept import Concept, ConceptMapping, Tradition


def find_analogues(
    concept: Concept,
    target_tradition: Tradition | None,
    *,
    alignment: AlignmentMethod,
    concept_registry: dict[str, Concept],
    top_k: int = 5,
) -> list[ConceptMapping]:
    """Find closest analogues of a concept, optionally in a target tradition."""
    if target_tradition is None:
        candidates = [c for c in concept_registry.values() if c.id != concept.id]
    else:
        candidates = [
            c for c in concept_registry.values()
            if c.tradition.name == target_tradition.name and c.id != concept.id
        ]
    return alignment.align_one_to_many(concept, candidates, top_k=top_k)
