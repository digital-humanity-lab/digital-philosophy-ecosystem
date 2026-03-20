"""Track semantic evolution of philosophical terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TermUsage:
    term: str
    context: str
    source_id: str
    author: str
    date: Optional[str] = None
    definition_given: str = ""


class TermEvolution:
    """Track how a philosophical term's meaning evolves over time."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self._model_name = embedding_model
        self._encoder = None
        self._usages: list[TermUsage] = []

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self._model_name)
            except ImportError:
                raise ImportError(
                    "TermEvolution requires sentence-transformers. "
                    "Install with: pip install philtext[embed]"
                )
        return self._encoder

    def add_usage(self, usage: TermUsage) -> None:
        self._usages.append(usage)

    def add_usages(self, usages: list[TermUsage]) -> None:
        self._usages.extend(usages)

    def trace(self, term: str) -> dict:
        relevant = [u for u in self._usages if u.term.lower() == term.lower()]
        relevant.sort(key=lambda u: u.date or "")

        if len(relevant) < 2:
            return {"term": term, "usages": len(relevant),
                    "evolution": "insufficient_data"}

        encoder = self._get_encoder()
        contexts = [u.context for u in relevant]
        embeddings = encoder.encode(contexts)

        shifts = []
        for i in range(len(embeddings) - 1):
            sim = float(
                np.dot(embeddings[i], embeddings[i + 1])
                / (np.linalg.norm(embeddings[i])
                   * np.linalg.norm(embeddings[i + 1]) + 1e-9)
            )
            shifts.append({
                "from": {"source": relevant[i].source_id,
                         "date": relevant[i].date,
                         "context": relevant[i].context[:200]},
                "to": {"source": relevant[i + 1].source_id,
                       "date": relevant[i + 1].date,
                       "context": relevant[i + 1].context[:200]},
                "semantic_similarity": sim,
                "shift_detected": sim < 0.70,
            })

        meaning_clusters: dict[int, list[dict]] = {}
        try:
            from sklearn.cluster import AgglomerativeClustering
            if len(embeddings) >= 3:
                clustering = AgglomerativeClustering(
                    n_clusters=None, distance_threshold=0.5,
                    metric="cosine", linkage="average",
                )
                labels = clustering.fit_predict(embeddings)
                for label, usage in zip(labels, relevant):
                    meaning_clusters.setdefault(int(label), []).append(
                        {"source": usage.source_id, "date": usage.date}
                    )
            else:
                meaning_clusters = {0: [{"source": u.source_id, "date": u.date}
                                        for u in relevant]}
        except ImportError:
            meaning_clusters = {0: [{"source": u.source_id, "date": u.date}
                                    for u in relevant]}

        return {
            "term": term,
            "num_usages": len(relevant),
            "date_range": (relevant[0].date, relevant[-1].date),
            "chronological_shifts": shifts,
            "meaning_clusters": meaning_clusters,
            "num_distinct_meanings": len(meaning_clusters),
        }
