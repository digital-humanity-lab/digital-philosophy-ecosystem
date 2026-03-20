"""Structural alignment based on graph position in tradition ontologies."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from philmap.alignment.base import AlignmentMethod
from philmap.concept import (
    AlignmentEvidence, AlignmentType, Concept, ConceptMapping, Tradition,
)


class TraditionOntology:
    """Graph representation of a tradition's concept structure."""

    def __init__(self, tradition: Tradition):
        self.tradition = tradition
        self.graph = nx.DiGraph()

    def add_concept(self, concept: Concept) -> None:
        self.graph.add_node(concept.id, concept=concept)

    def add_relation(self, source_id: str, target_id: str, relation: str) -> None:
        self.graph.add_edge(source_id, target_id, relation=relation)

    def neighborhood(self, concept_id: str, radius: int = 2) -> nx.DiGraph:
        nodes: set[str] = set()
        frontier = {concept_id}
        for _ in range(radius):
            next_frontier: set[str] = set()
            for n in frontier:
                next_frontier.update(self.graph.successors(n))
                next_frontier.update(self.graph.predecessors(n))
            nodes.update(frontier)
            frontier = next_frontier - nodes
        nodes.update(frontier)
        return self.graph.subgraph(nodes).copy()

    def structural_signature(self, concept_id: str) -> dict[str, Any]:
        g = self.graph
        if concept_id not in g:
            return {"in_degree": 0, "out_degree": 0, "betweenness": 0,
                    "pagerank": 0, "depth_from_root": -1, "edge_types": {}}
        bc = nx.betweenness_centrality(g)
        pr = nx.pagerank(g)
        return {
            "in_degree": g.in_degree(concept_id),
            "out_degree": g.out_degree(concept_id),
            "betweenness": bc.get(concept_id, 0),
            "pagerank": pr.get(concept_id, 0),
            "depth_from_root": self._depth(concept_id),
            "edge_types": self._edge_type_distribution(concept_id),
        }

    def _depth(self, concept_id: str) -> int:
        roots = [n for n in self.graph if self.graph.in_degree(n) == 0]
        min_depth = float("inf")
        for r in roots:
            try:
                d = nx.shortest_path_length(self.graph, r, concept_id)
                min_depth = min(min_depth, d)
            except nx.NetworkXNoPath:
                pass
        return int(min_depth) if min_depth < float("inf") else -1

    def _edge_type_distribution(self, concept_id: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, _, data in self.graph.edges(concept_id, data=True):
            r = data.get("relation", "unknown")
            counts[r] = counts.get(r, 0) + 1
        for _, _, data in self.graph.in_edges(concept_id, data=True):
            r = "inv_" + data.get("relation", "unknown")
            counts[r] = counts.get(r, 0) + 1
        return counts


class StructuralAlignment(AlignmentMethod):
    def __init__(
        self,
        source_ontology: TraditionOntology,
        target_ontology: TraditionOntology,
        neighborhood_radius: int = 2,
    ):
        self.source_onto = source_ontology
        self.target_onto = target_ontology
        self.radius = neighborhood_radius

    def align(self, source: Concept, target: Concept) -> ConceptMapping:
        sig_s = self.source_onto.structural_signature(source.id)
        sig_t = self.target_onto.structural_signature(target.id)

        vec_s = self._signature_to_vector(sig_s)
        vec_t = self._signature_to_vector(sig_t)
        denom = np.linalg.norm(vec_s) * np.linalg.norm(vec_t) + 1e-9
        cos_sim = float(np.dot(vec_s, vec_t) / denom)

        sub_s = self.source_onto.neighborhood(source.id, self.radius)
        sub_t = self.target_onto.neighborhood(target.id, self.radius)
        topo_score = self._topology_similarity(sub_s, sub_t)

        score = max(0.0, min(1.0, 0.5 * cos_sim + 0.5 * topo_score))

        return ConceptMapping(
            source=source, target=target,
            overall_score=score,
            alignment_type=AlignmentType.STRUCTURAL,
            evidence=[AlignmentEvidence(
                method=AlignmentType.STRUCTURAL,
                score=score,
                details={
                    "signature_similarity": cos_sim,
                    "topology_similarity": topo_score,
                },
            )],
        )

    def align_one_to_many(
        self, source: Concept, candidates: list[Concept], top_k: int = 5
    ) -> list[ConceptMapping]:
        mappings = [self.align(source, c) for c in candidates]
        mappings.sort(key=lambda m: m.overall_score, reverse=True)
        return mappings[:top_k]

    @staticmethod
    def _signature_to_vector(sig: dict[str, Any]) -> np.ndarray:
        return np.array([
            sig["in_degree"], sig["out_degree"],
            sig["betweenness"], sig["pagerank"],
            max(sig["depth_from_root"], 0),
        ], dtype=float)

    @staticmethod
    def _topology_similarity(g1: nx.DiGraph, g2: nx.DiGraph) -> float:
        def edge_histogram(g: nx.DiGraph) -> dict[str, int]:
            h: dict[str, int] = {}
            for _, _, d in g.edges(data=True):
                r = d.get("relation", "unknown")
                h[r] = h.get(r, 0) + 1
            return h
        h1, h2 = edge_histogram(g1), edge_histogram(g2)
        all_keys = sorted(set(h1) | set(h2))
        if not all_keys:
            return 0.0
        v1 = np.array([h1.get(k, 0) for k in all_keys], dtype=float)
        v2 = np.array([h2.get(k, 0) for k in all_keys], dtype=float)
        denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
        return float(np.dot(v1, v2) / denom)
