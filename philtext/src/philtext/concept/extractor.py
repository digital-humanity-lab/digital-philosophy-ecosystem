"""Concept extraction and ontology linking."""

from __future__ import annotations

import re

import numpy as np

from philtext.concept.ontology import ConceptMention, ConceptNode, PhilOntology


class ConceptExtractor:
    """Identify philosophical concepts in text, linking to an ontology.

    Two-stage: candidate generation (dictionary + n-gram) then
    disambiguation via embedding similarity.
    """

    def __init__(
        self,
        ontology: PhilOntology | None = None,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        similarity_threshold: float = 0.55,
    ):
        self.ontology = ontology or PhilOntology.load_default()
        self.threshold = similarity_threshold
        self._encoder = None
        self._embedding_model = embedding_model
        self._label_to_concepts: dict[str, list[ConceptNode]] = {}
        self._build_label_index()

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self._embedding_model)
            except ImportError:
                raise ImportError(
                    "ConceptExtractor requires sentence-transformers. "
                    "Install with: pip install philtext[embed]"
                )
        return self._encoder

    def _build_label_index(self) -> None:
        self._label_to_concepts.clear()
        for node in self.ontology.all_concepts():
            for _, label in node.labels.items():
                self._label_to_concepts.setdefault(label.lower(), []).append(node)
            for _, alts in node.alt_labels.items():
                for alt in alts:
                    self._label_to_concepts.setdefault(alt.lower(), []).append(node)

    def extract(self, text: str, language: str = "en") -> list[ConceptMention]:
        candidates = self._generate_candidates(text)
        mentions = []
        for surface, span, context, nodes in candidates:
            best_node, score = self._disambiguate(context, nodes)
            if score >= self.threshold:
                mentions.append(ConceptMention(
                    concept=best_node, surface_form=surface,
                    span=span, confidence=score,
                    context_sentence=context,
                ))
        return mentions

    def _generate_candidates(
        self, text: str
    ) -> list[tuple[str, tuple[int, int], str, list[ConceptNode]]]:
        results = []
        text_lower = text.lower()
        for label, nodes in self._label_to_concepts.items():
            start = text_lower.find(label)
            if start >= 0:
                end = start + len(label)
                surface = text[start:end]
                ctx = self._get_context_sentence(text, start)
                results.append((surface, (start, end), ctx, nodes))
        return results

    def _disambiguate(
        self, context: str, candidates: list[ConceptNode]
    ) -> tuple[ConceptNode, float]:
        if len(candidates) == 1:
            return candidates[0], 0.8

        encoder = self._get_encoder()
        ctx_emb = encoder.encode(context)
        best_score = -1.0
        best_node = candidates[0]
        for node in candidates:
            def_text = node.definition or node.label()
            def_emb = encoder.encode(def_text)
            score = float(
                np.dot(ctx_emb, def_emb)
                / (np.linalg.norm(ctx_emb) * np.linalg.norm(def_emb) + 1e-9)
            )
            if score > best_score:
                best_score = score
                best_node = node
        return best_node, best_score

    @staticmethod
    def _get_context_sentence(text: str, char_pos: int) -> str:
        start = text.rfind(".", 0, char_pos)
        end = text.find(".", char_pos)
        return text[max(0, start + 1):end + 1 if end > 0 else len(text)].strip()
