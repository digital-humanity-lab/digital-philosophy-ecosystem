"""Concept embedding engine using multilingual sentence transformers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from philmap.concept import Concept


@dataclass
class EmbeddingConfig:
    model_name: str = "intfloat/multilingual-e5-large-instruct"
    facet_weights: dict[str, float] = field(default_factory=lambda: {
        "definition": 0.5, "usage": 0.3, "relational": 0.2,
    })
    instruction_prefix: str = (
        "Instruct: Represent the philosophical concept for retrieval\nQuery: "
    )
    max_seq_length: int = 512
    normalize: bool = True


class FacetedEmbedding:
    """Multi-facet embedding of a philosophical concept."""

    def __init__(
        self,
        concept_id: str,
        definition_vec: np.ndarray,
        usage_vec: np.ndarray,
        relational_vec: np.ndarray,
        config: EmbeddingConfig,
    ):
        self.concept_id = concept_id
        self.definition = definition_vec
        self.usage = usage_vec
        self.relational = relational_vec
        self._config = config

    def composite(self) -> np.ndarray:
        w = self._config.facet_weights
        vec = (
            w["definition"] * self.definition
            + w["usage"] * self.usage
            + w["relational"] * self.relational
        )
        if self._config.normalize:
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
        return vec

    def facet(self, name: str) -> np.ndarray:
        if name == "composite":
            return self.composite()
        return getattr(self, name)


class ConceptEmbedder:
    """Embeds philosophical concepts using multilingual transformers."""

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "ConceptEmbedder requires sentence-transformers. "
                "Install with: pip install philmap"
            )
        self._model = SentenceTransformer(
            self.config.model_name,
            trust_remote_code=True,
        )
        self._model.max_seq_length = self.config.max_seq_length
        self._cache: dict[str, FacetedEmbedding] = {}

    def embed(self, concept: Concept) -> FacetedEmbedding:
        if concept.id in self._cache:
            return self._cache[concept.id]

        def_texts = []
        usage_texts = []
        for desc in concept.descriptions:
            def_texts.append(f"[{desc.language}] {desc.term}: {desc.definition}")
            for ctx in desc.usage_contexts:
                usage_texts.append(f"[{desc.language}] {ctx}")

        relational_text = (
            f"{concept.primary_term} belongs to {concept.tradition.name}. "
            f"Related concepts: {', '.join(concept.related_concepts)}."
        )

        prefix = self.config.instruction_prefix

        definition_vec = self._model.encode(
            [prefix + t for t in def_texts],
            normalize_embeddings=self.config.normalize,
        ).mean(axis=0)

        if usage_texts:
            usage_vec = self._model.encode(
                [prefix + t for t in usage_texts],
                normalize_embeddings=self.config.normalize,
            ).mean(axis=0)
        else:
            usage_vec = np.zeros_like(definition_vec)

        relational_vec = self._model.encode(
            [prefix + relational_text],
            normalize_embeddings=self.config.normalize,
        ).squeeze()

        emb = FacetedEmbedding(
            concept_id=concept.id,
            definition_vec=definition_vec,
            usage_vec=usage_vec,
            relational_vec=relational_vec,
            config=self.config,
        )
        self._cache[concept.id] = emb
        return emb

    def embed_many(self, concepts: list[Concept]) -> dict[str, FacetedEmbedding]:
        return {c.id: self.embed(c) for c in concepts}

    def similarity(
        self, a: FacetedEmbedding, b: FacetedEmbedding, facet: str = "composite"
    ) -> float:
        va = a.facet(facet)
        vb = b.facet(facet)
        denom = np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9
        return float(np.dot(va, vb) / denom)
