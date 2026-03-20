"""Argumentative alignment based on shared argument roles."""

from __future__ import annotations

from dataclasses import dataclass, field

from philmap.alignment.base import AlignmentMethod
from philmap.concept import (
    AlignmentEvidence, AlignmentType, Concept, ConceptMapping, Tradition,
)
from philmap.embedding.embedder import ConceptEmbedder


@dataclass
class ArgumentRole:
    """A concept's role in an argument schema."""
    role: str
    argument_type: str
    position: str


@dataclass
class ArgumentSchema:
    """A formalized argument structure."""
    name: str
    tradition: Tradition
    roles: dict[str, Concept] = field(default_factory=dict)
    structure: str = ""


class ArgumentativeAlignment(AlignmentMethod):
    def __init__(
        self,
        embedder: ConceptEmbedder,
        schemas: list[ArgumentSchema] | None = None,
    ):
        self.embedder = embedder
        self.schemas = schemas or []
        self._role_index: dict[str, list[tuple[ArgumentSchema, str]]] = {}
        self._build_index()

    def _build_index(self) -> None:
        for schema in self.schemas:
            for role_name, concept in schema.roles.items():
                self._role_index.setdefault(concept.id, []).append(
                    (schema, role_name)
                )

    def align(self, source: Concept, target: Concept) -> ConceptMapping:
        source_roles = self._role_index.get(source.id, [])
        target_roles = self._role_index.get(target.id, [])

        if not source_roles or not target_roles:
            return self._fallback_align(source, target)

        s_roles_only = {r for _, r in source_roles}
        t_roles_only = {r for _, r in target_roles}
        overlap = s_roles_only & t_roles_only
        union = s_roles_only | t_roles_only
        role_score = len(overlap) / len(union) if union else 0.0

        s_types = {s.structure for s, _ in source_roles}
        t_types = {s.structure for s, _ in target_roles}
        type_overlap = s_types & t_types
        type_union = s_types | t_types
        type_score = len(type_overlap) / len(type_union) if type_union else 0.0

        score = 0.6 * role_score + 0.4 * type_score

        return ConceptMapping(
            source=source, target=target,
            overall_score=score,
            alignment_type=AlignmentType.ARGUMENTATIVE,
            evidence=[AlignmentEvidence(
                method=AlignmentType.ARGUMENTATIVE,
                score=score,
                details={"shared_roles": list(overlap), "role_jaccard": role_score,
                         "structure_jaccard": type_score},
            )],
        )

    def _fallback_align(self, source: Concept, target: Concept) -> ConceptMapping:
        emb_s = self.embedder.embed(source)
        emb_t = self.embedder.embed(target)
        score = self.embedder.similarity(emb_s, emb_t, facet="usage") * 0.7
        return ConceptMapping(
            source=source, target=target,
            overall_score=score,
            alignment_type=AlignmentType.ARGUMENTATIVE,
            evidence=[AlignmentEvidence(
                method=AlignmentType.ARGUMENTATIVE,
                score=score,
                details={"note": "fallback to usage embedding similarity"},
            )],
        )

    def align_one_to_many(
        self, source: Concept, candidates: list[Concept], top_k: int = 5
    ) -> list[ConceptMapping]:
        mappings = [self.align(source, c) for c in candidates]
        mappings.sort(key=lambda m: m.overall_score, reverse=True)
        return mappings[:top_k]
