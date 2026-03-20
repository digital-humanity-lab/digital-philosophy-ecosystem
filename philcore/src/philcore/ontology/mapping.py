"""Cross-tradition concept mapping with commensurability tracking."""

from __future__ import annotations

from dataclasses import dataclass, field

from philcore.models.relation import CrossTraditionMapping
from philcore.types import MappingConfidence, RelationType

_CONFIDENCE_RANK: dict[MappingConfidence, int] = {
    MappingConfidence.EXACT: 4,
    MappingConfidence.CLOSE: 3,
    MappingConfidence.PARTIAL: 2,
    MappingConfidence.SPECULATIVE: 1,
}


@dataclass
class MappingQuery:
    """Parameters for finding cross-tradition mappings."""
    concept_id: str
    target_tradition_id: str | None = None
    min_confidence: MappingConfidence = MappingConfidence.SPECULATIVE
    relation_types: list[RelationType] | None = None

    def accepts(self, mapping: CrossTraditionMapping) -> bool:
        rel = mapping.relation
        if (rel.source_concept_id != self.concept_id
                and rel.target_concept_id != self.concept_id):
            return False
        if (self.target_tradition_id
                and mapping.target_tradition_id != self.target_tradition_id):
            return False
        if (_CONFIDENCE_RANK[rel.confidence]
                < _CONFIDENCE_RANK[self.min_confidence]):
            return False
        if self.relation_types and rel.relation_type not in self.relation_types:
            return False
        return True


class MappingRegistry:
    """Stores and queries CrossTraditionMappings."""

    def __init__(self) -> None:
        self._mappings: dict[str, CrossTraditionMapping] = {}
        self._by_concept: dict[str, list[str]] = {}

    def register(self, mapping: CrossTraditionMapping) -> None:
        self._mappings[mapping.id] = mapping
        src = mapping.relation.source_concept_id
        tgt = mapping.relation.target_concept_id
        self._by_concept.setdefault(src, []).append(mapping.id)
        self._by_concept.setdefault(tgt, []).append(mapping.id)

    def query(self, q: MappingQuery) -> list[CrossTraditionMapping]:
        candidate_ids = self._by_concept.get(q.concept_id, [])
        return [
            self._mappings[mid]
            for mid in candidate_ids
            if q.accepts(self._mappings[mid])
        ]

    def all_mappings(self) -> list[CrossTraditionMapping]:
        return list(self._mappings.values())

    def __len__(self) -> int:
        return len(self._mappings)
