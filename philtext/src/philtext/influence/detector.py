"""Intertextual influence detection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class InfluenceLink:
    source_text_id: str
    target_text_id: str
    influence_type: str
    evidence_pairs: list[tuple[str, str]]
    similarity_score: float
    confidence: float
    direction_confidence: float
    metadata: dict = field(default_factory=dict)


class InfluenceDetector:
    """Detect intertextual references and philosophical influences."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        chunk_size: int = 256,
        chunk_overlap: int = 64,
        similarity_threshold: float = 0.60,
    ):
        self._model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.threshold = similarity_threshold
        self._bi_encoder = None

    def _get_encoder(self):
        if self._bi_encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._bi_encoder = SentenceTransformer(self._model_name)
            except ImportError:
                raise ImportError(
                    "InfluenceDetector requires sentence-transformers. "
                    "Install with: pip install philtext[embed]"
                )
        return self._bi_encoder

    def detect(
        self,
        source_texts: dict[str, str],
        target_texts: dict[str, str],
        metadata: dict[str, dict] | None = None,
    ) -> list[InfluenceLink]:
        encoder = self._get_encoder()
        metadata = metadata or {}

        source_chunks: dict[str, list[str]] = {}
        target_chunks: dict[str, list[str]] = {}
        for tid, text in source_texts.items():
            source_chunks[tid] = self._chunk(text)
        for tid, text in target_texts.items():
            target_chunks[tid] = self._chunk(text)

        all_source, source_index = [], []
        for tid, chunks in source_chunks.items():
            for i, chunk in enumerate(chunks):
                all_source.append(chunk)
                source_index.append((tid, i))

        all_target, target_index = [], []
        for tid, chunks in target_chunks.items():
            for i, chunk in enumerate(chunks):
                all_target.append(chunk)
                target_index.append((tid, i))

        if not all_source or not all_target:
            return []

        src_embs = encoder.encode(all_source, convert_to_numpy=True)
        tgt_embs = encoder.encode(all_target, convert_to_numpy=True)

        sim_matrix = np.dot(src_embs, tgt_embs.T) / (
            np.linalg.norm(src_embs, axis=1, keepdims=True)
            * np.linalg.norm(tgt_embs, axis=1, keepdims=True).T + 1e-9
        )

        candidates: dict[tuple[str, str], list[tuple[str, str, float]]] = {}
        rows, cols = np.where(sim_matrix > self.threshold)
        for r, c in zip(rows, cols):
            src_tid = source_index[r][0]
            tgt_tid = target_index[c][0]
            key = (src_tid, tgt_tid)
            candidates.setdefault(key, []).append(
                (all_source[r], all_target[c], float(sim_matrix[r, c]))
            )

        links = []
        for (src_id, tgt_id), pairs in candidates.items():
            pairs.sort(key=lambda x: x[2], reverse=True)
            top_pairs = pairs[:5]
            avg_score = float(np.mean([s for _, _, s in top_pairs]))
            evidence = [(s, t) for s, t, _ in top_pairs]
            direction_conf = self._assess_direction(src_id, tgt_id, metadata)

            links.append(InfluenceLink(
                source_text_id=src_id, target_text_id=tgt_id,
                influence_type=self._classify_type(avg_score),
                evidence_pairs=evidence, similarity_score=avg_score,
                confidence=min(avg_score, 1.0),
                direction_confidence=direction_conf,
            ))

        return sorted(links, key=lambda l: l.confidence, reverse=True)

    def _chunk(self, text: str) -> list[str]:
        words = text.split()
        chunks, step = [], self.chunk_size - self.chunk_overlap
        for i in range(0, len(words), max(step, 1)):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    @staticmethod
    def _assess_direction(src_id: str, tgt_id: str, metadata: dict) -> float:
        src_date = metadata.get(src_id, {}).get("date")
        tgt_date = metadata.get(tgt_id, {}).get("date")
        if src_date and tgt_date and str(src_date) < str(tgt_date):
            return 0.9
        elif src_date and tgt_date and str(src_date) > str(tgt_date):
            return 0.1
        return 0.5

    @staticmethod
    def _classify_type(score: float) -> str:
        if score > 0.90:
            return "direct_citation"
        elif score > 0.75:
            return "paraphrase"
        elif score > 0.60:
            return "conceptual"
        return "structural"
