"""Cross-source entity resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from philgraph.graph import PhilGraph

from philgraph.schema import Thinker


class EntityResolver:
    """Cross-source entity resolution via external IDs, labels, and dates."""

    def __init__(self, graph: PhilGraph):
        self.graph = graph

    def resolve_all(self, dry_run: bool = False) -> list[tuple[str, str, float]]:
        candidates: list[tuple[str, str, float]] = []
        candidates.extend(self._match_by_external_ids())
        candidates.extend(self._match_by_labels(threshold=0.85))
        candidates.extend(self._match_thinkers_by_dates(year_tolerance=2))
        candidates = self._deduplicate(candidates)

        if not dry_run:
            for uid_a, uid_b, score in candidates:
                if score >= 0.9:
                    self.graph.merge_node(uid_a,
                                          self.graph.get_node(uid_b))

        return candidates

    def _match_by_external_ids(self) -> list[tuple[str, str, float]]:
        index: dict[tuple[str, str], list[str]] = {}
        for uid, node in self.graph.iter_nodes():
            for source, ext_id in node.external_ids.items():
                key = (source, ext_id)
                index.setdefault(key, []).append(uid)
        matches = []
        for _, uids in index.items():
            if len(uids) > 1:
                for i, a in enumerate(uids):
                    for b in uids[i + 1:]:
                        matches.append((a, b, 1.0))
        return matches

    def _match_by_labels(self, threshold: float) -> list[tuple[str, str, float]]:
        try:
            from rapidfuzz import fuzz
        except ImportError:
            return []

        by_type: dict[str, list[tuple[str, list[str]]]] = {}
        for uid, node in self.graph.iter_nodes():
            ntype = type(node).__name__
            all_labels = [node.label] + list(node.labels_i18n.values())
            by_type.setdefault(ntype, []).append((uid, all_labels))

        matches = []
        for _, entries in by_type.items():
            for i, (uid_a, labels_a) in enumerate(entries):
                for uid_b, labels_b in entries[i + 1:]:
                    best = max(
                        fuzz.ratio(la, lb) / 100.0
                        for la in labels_a for lb in labels_b
                    )
                    if best >= threshold:
                        matches.append((uid_a, uid_b, best))
        return matches

    def _match_thinkers_by_dates(
        self, year_tolerance: int
    ) -> list[tuple[str, str, float]]:
        thinkers = [
            (uid, n) for uid, n in self.graph.iter_nodes()
            if isinstance(n, Thinker)
        ]
        matches = []
        for i, (uid_a, ta) in enumerate(thinkers):
            for uid_b, tb in thinkers[i + 1:]:
                if (ta.birth_year and tb.birth_year
                        and abs(ta.birth_year - tb.birth_year) <= year_tolerance
                        and ta.death_year and tb.death_year
                        and abs(ta.death_year - tb.death_year) <= year_tolerance):
                    matches.append((uid_a, uid_b, 0.8))
        return matches

    @staticmethod
    def _deduplicate(
        candidates: list[tuple[str, str, float]]
    ) -> list[tuple[str, str, float]]:
        seen: dict[tuple[str, str], float] = {}
        for a, b, score in candidates:
            key = (min(a, b), max(a, b))
            if key not in seen or score > seen[key]:
                seen[key] = score
        return [(a, b, s) for (a, b), s in sorted(
            seen.items(), key=lambda x: x[1], reverse=True
        )]
