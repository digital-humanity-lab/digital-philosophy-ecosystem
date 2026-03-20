"""Hybrid alignment combining semantic, structural, and argumentative methods."""

from __future__ import annotations

from philmap.alignment.base import AlignmentMethod
from philmap.alignment.semantic import SemanticAlignment
from philmap.alignment.structural import StructuralAlignment
from philmap.alignment.argumentative import ArgumentativeAlignment
from philmap.concept import AlignmentType, Concept, ConceptMapping


class HybridAlignment(AlignmentMethod):
    def __init__(
        self,
        semantic: SemanticAlignment,
        structural: StructuralAlignment,
        argumentative: ArgumentativeAlignment,
        weights: dict[str, float] | None = None,
    ):
        self.semantic = semantic
        self.structural = structural
        self.argumentative = argumentative
        self.weights = weights or {
            "semantic": 0.4, "structural": 0.35, "argumentative": 0.25,
        }

    def align(self, source: Concept, target: Concept) -> ConceptMapping:
        m_sem = self.semantic.align(source, target)
        m_str = self.structural.align(source, target)
        m_arg = self.argumentative.align(source, target)

        score = (
            self.weights["semantic"] * m_sem.overall_score
            + self.weights["structural"] * m_str.overall_score
            + self.weights["argumentative"] * m_arg.overall_score
        )

        return ConceptMapping(
            source=source, target=target,
            overall_score=score,
            alignment_type=AlignmentType.HYBRID,
            evidence=m_sem.evidence + m_str.evidence + m_arg.evidence,
        )

    def align_one_to_many(
        self, source: Concept, candidates: list[Concept], top_k: int = 5
    ) -> list[ConceptMapping]:
        mappings = [self.align(source, c) for c in candidates]
        mappings.sort(key=lambda m: m.overall_score, reverse=True)
        return mappings[:top_k]
