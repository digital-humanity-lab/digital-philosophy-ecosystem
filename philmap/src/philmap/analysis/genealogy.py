"""Trace a concept's evolution across time and traditions."""

from __future__ import annotations

from dataclasses import dataclass, field

from philmap.alignment.base import AlignmentMethod
from philmap.concept import Concept


@dataclass
class GenealogyNode:
    concept: Concept
    period: tuple[int, int] | None
    influences: list[str] = field(default_factory=list)
    influenced_by: list[str] = field(default_factory=list)


def concept_genealogy(
    concept: Concept,
    *,
    alignment: AlignmentMethod,
    concept_registry: dict[str, Concept],
    temporal_order: bool = True,
    threshold: float = 0.4,
) -> list[GenealogyNode]:
    """Trace a concept's evolution across time and traditions."""
    all_others = [c for c in concept_registry.values() if c.id != concept.id]
    mappings = alignment.align_one_to_many(concept, all_others, top_k=len(all_others))
    related = [m for m in mappings if m.overall_score >= threshold]

    nodes = [
        GenealogyNode(
            concept=m.target,
            period=m.target.tradition.period,
            influences=[],
            influenced_by=[],
        )
        for m in related
    ]

    anchor = GenealogyNode(
        concept=concept,
        period=concept.tradition.period,
        influences=[],
        influenced_by=[],
    )
    nodes.insert(0, anchor)

    if temporal_order:
        nodes.sort(key=lambda n: (n.period[0] if n.period else 9999))

    for i, node in enumerate(nodes):
        for j in range(i):
            earlier = nodes[j]
            if (earlier.period and node.period
                    and earlier.period[1] <= node.period[0]):
                node.influences.append(earlier.concept.id)
                earlier.influenced_by.append(node.concept.id)

    return nodes
