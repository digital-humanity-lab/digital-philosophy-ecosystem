"""Find all mappable concept pairs between two traditions."""

from __future__ import annotations

from philmap.alignment.base import AlignmentMethod
from philmap.concept import Concept, ConceptMapping, Tradition


def tradition_bridge(
    tradition_a: Tradition,
    tradition_b: Tradition,
    *,
    alignment: AlignmentMethod,
    concept_registry: dict[str, Concept],
    threshold: float = 0.5,
    top_k_per_concept: int = 3,
) -> list[ConceptMapping]:
    """Find all mappable concept pairs between two traditions."""
    concepts_a = [
        c for c in concept_registry.values()
        if c.tradition.name == tradition_a.name
    ]
    concepts_b = [
        c for c in concept_registry.values()
        if c.tradition.name == tradition_b.name
    ]

    all_mappings: list[ConceptMapping] = []
    for ca in concepts_a:
        top = alignment.align_one_to_many(ca, concepts_b, top_k=top_k_per_concept)
        all_mappings.extend(m for m in top if m.overall_score >= threshold)

    best: dict[tuple[str, str], ConceptMapping] = {}
    for m in all_mappings:
        key = (m.source.id, m.target.id)
        if key not in best or m.overall_score > best[key].overall_score:
            best[key] = m

    return sorted(best.values(), key=lambda m: m.overall_score, reverse=True)
