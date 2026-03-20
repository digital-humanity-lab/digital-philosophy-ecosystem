"""DAG-based concept hierarchies backed by NetworkX."""

from __future__ import annotations

from typing import Iterator

import networkx as nx

from philcore.models.concept import Concept
from philcore.models.relation import ConceptRelation
from philcore.types import RelationType


class ConceptHierarchy:
    """A directed acyclic graph of concepts connected by subsumption,
    influence, and other relations."""

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._concepts: dict[str, Concept] = {}

    def add_concept(self, concept: Concept) -> None:
        self._concepts[concept.id] = concept
        self._graph.add_node(concept.id)
        for broader_id in concept.broader_concept_ids:
            self._graph.add_edge(concept.id, broader_id,
                                 relation_type=RelationType.SUBSUMPTION)
        for narrower_id in concept.narrower_concept_ids:
            self._graph.add_edge(narrower_id, concept.id,
                                 relation_type=RelationType.SUBSUMPTION)

    def add_relation(self, rel: ConceptRelation) -> None:
        self._graph.add_edge(
            rel.source_concept_id, rel.target_concept_id,
            relation_type=rel.relation_type, relation=rel,
        )

    def ancestors(self, concept_id: str) -> set[str]:
        return nx.ancestors(self._graph, concept_id)

    def descendants(self, concept_id: str) -> set[str]:
        return nx.descendants(self._graph, concept_id)

    def related(
        self, concept_id: str, relation_type: RelationType | None = None
    ) -> Iterator[tuple[str, ConceptRelation | None]]:
        for _, target, data in self._graph.out_edges(concept_id, data=True):
            if relation_type is None or data.get("relation_type") == relation_type:
                yield target, data.get("relation")

    def shortest_path(self, source_id: str, target_id: str) -> list[str] | None:
        try:
            return nx.shortest_path(self._graph, source_id, target_id)
        except nx.NetworkXNoPath:
            return None

    def concept(self, concept_id: str) -> Concept | None:
        return self._concepts.get(concept_id)

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    def __len__(self) -> int:
        return len(self._concepts)
