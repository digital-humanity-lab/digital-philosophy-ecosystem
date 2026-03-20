"""Parallel text alignment across languages."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass
class AlignedSegment:
    source_text: str
    target_text: str
    source_language: str
    target_language: str
    alignment_score: float
    source_ref: str = ""
    target_ref: str = ""


class TextAligner:
    """Align parallel philosophical texts across languages."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        min_score: float = 0.40,
    ):
        self._model_name = embedding_model
        self.min_score = min_score
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self._model_name)
            except ImportError:
                raise ImportError(
                    "TextAligner requires sentence-transformers. "
                    "Install with: pip install philtext[embed]"
                )
        return self._encoder

    def align(
        self, source: str, target: str,
        source_lang: str, target_lang: str,
    ) -> list[AlignedSegment]:
        encoder = self._get_encoder()
        src_sents = self._split_sentences(source, source_lang)
        tgt_sents = self._split_sentences(target, target_lang)

        if not src_sents or not tgt_sents:
            return []

        src_embs = encoder.encode(src_sents)
        tgt_embs = encoder.encode(tgt_sents)

        sim = np.dot(src_embs, tgt_embs.T) / (
            np.linalg.norm(src_embs, axis=1, keepdims=True)
            * np.linalg.norm(tgt_embs, axis=1, keepdims=True).T + 1e-9
        )

        alignments = self._dp_align(sim)
        results = []
        for i, j, score in alignments:
            if score >= self.min_score:
                results.append(AlignedSegment(
                    source_text=src_sents[i], target_text=tgt_sents[j],
                    source_language=source_lang, target_language=target_lang,
                    alignment_score=float(score),
                ))
        return results

    @staticmethod
    def _split_sentences(text: str, lang: str) -> list[str]:
        if lang in ("ja", "zh"):
            return [s.strip() for s in re.split(r'[。！？\n]', text) if s.strip()]
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    @staticmethod
    def _dp_align(sim: np.ndarray) -> list[tuple[int, int, float]]:
        m, _ = sim.shape
        alignments = []
        last_j = -1
        for i in range(m):
            best_j, best_score = -1, -1.0
            for j in range(last_j + 1, sim.shape[1]):
                if sim[i, j] > best_score:
                    best_score = sim[i, j]
                    best_j = j
            if best_j >= 0:
                alignments.append((i, best_j, best_score))
                last_j = best_j
        return alignments
