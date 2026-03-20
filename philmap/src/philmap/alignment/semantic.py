"""Semantic alignment based on embedding cosine similarity."""

from __future__ import annotations

from philmap.alignment.base import AlignmentMethod
from philmap.concept import AlignmentEvidence, AlignmentType, Concept, ConceptMapping
from philmap.embedding.embedder import ConceptEmbedder


class SemanticAlignment(AlignmentMethod):
    def __init__(self, embedder: ConceptEmbedder, facet: str = "composite"):
        self.embedder = embedder
        self.facet = facet

    def align(self, source: Concept, target: Concept) -> ConceptMapping:
        emb_s = self.embedder.embed(source)
        emb_t = self.embedder.embed(target)
        score = self.embedder.similarity(emb_s, emb_t, facet=self.facet)

        facet_scores = {}
        for f in ("definition", "usage", "relational"):
            facet_scores[f] = self.embedder.similarity(emb_s, emb_t, facet=f)

        return ConceptMapping(
            source=source, target=target,
            overall_score=score,
            alignment_type=AlignmentType.SEMANTIC,
            evidence=[AlignmentEvidence(
                method=AlignmentType.SEMANTIC,
                score=score,
                details={"facet_scores": facet_scores},
            )],
        )

    def align_one_to_many(
        self, source: Concept, candidates: list[Concept], top_k: int = 5
    ) -> list[ConceptMapping]:
        mappings = [self.align(source, c) for c in candidates]
        mappings.sort(key=lambda m: m.overall_score, reverse=True)
        return mappings[:top_k]
