"""PhilGraph main class - philosophical knowledge graph."""

from __future__ import annotations

from typing import Iterator

import networkx as nx

from philgraph.backends.base import GraphBackend
from philgraph.backends.networkx_backend import NetworkXBackend
from philgraph.schema import (
    Concept, Edge, EdgeType, EDGE_CONSTRAINTS, NodeBase, Thinker,
)


class PhilGraph:
    """Main entry point for the philosophical knowledge graph."""

    def __init__(
        self, backend: str | GraphBackend = "networkx", **backend_kwargs
    ):
        if isinstance(backend, str):
            self.backend = self._init_backend(backend, **backend_kwargs)
        else:
            self.backend = backend
        self._external_id_index: dict[tuple[str, str], str] = {}
        self._io = None

    @staticmethod
    def _init_backend(name: str, **kwargs) -> GraphBackend:
        if name == "networkx":
            return NetworkXBackend(**kwargs)
        elif name == "rdflib":
            from philgraph.backends.rdflib_backend import RDFLibBackend
            return RDFLibBackend(**kwargs)
        raise ValueError(f"Unknown backend: {name}")

    @property
    def io(self):
        if self._io is None:
            from philgraph.io.graph_io import GraphIO
            self._io = GraphIO(self)
        return self._io

    # ── Node operations ──────────────────────────────────────────────

    def add_node(self, node: NodeBase) -> None:
        self.backend.add_node(node)
        for source, ext_id in node.external_ids.items():
            self._external_id_index[(source, ext_id)] = node.uid

    def get_node(self, uid: str) -> NodeBase | None:
        return self.backend.get_node(uid)

    def merge_node(self, existing_uid: str, new_node: NodeBase) -> None:
        existing = self.backend.get_node(existing_uid)
        if existing is None:
            self.add_node(new_node)
            return
        existing.labels_i18n.update(new_node.labels_i18n)
        existing.external_ids.update(new_node.external_ids)
        existing.provenance = list(set(existing.provenance + new_node.provenance))
        for source, ext_id in new_node.external_ids.items():
            self._external_id_index[(source, ext_id)] = existing_uid
        self.backend.add_node(existing)

    def iter_nodes(self, node_type: str | None = None) -> Iterator[tuple[str, NodeBase]]:
        return self.backend.iter_nodes(node_type)

    # ── Edge operations ──────────────────────────────────────────────

    def add_edge(self, edge: Edge) -> None:
        src = self.get_node(edge.source_uid)
        tgt = self.get_node(edge.target_uid)
        if src and tgt:
            src_type = type(src).__name__
            tgt_type = type(tgt).__name__
            valid = EDGE_CONSTRAINTS.get(edge.edge_type, [])
            if valid and (src_type, tgt_type) not in valid:
                raise ValueError(
                    f"Invalid edge: {edge.edge_type.value} cannot connect "
                    f"{src_type} -> {tgt_type}"
                )
        self.backend.add_edge(edge)

    def iter_edges(self) -> list[Edge]:
        return self.backend.get_edges()

    # ── Entity resolution ────────────────────────────────────────────

    def resolve_external_id(self, ext_id: str, source: str) -> str | None:
        return self._external_id_index.get((source, ext_id))

    def resolve_entities(self, dry_run: bool = False) -> list[tuple[str, str, float]]:
        from philgraph.ingest.resolver import EntityResolver
        return EntityResolver(self).resolve_all(dry_run=dry_run)

    # ── Ingestion ────────────────────────────────────────────────────

    def ingest(self, source: str, **kwargs) -> dict:
        from philgraph.ingest.manual import ManualIngester
        from philgraph.ingest.wikidata import WikidataIngester
        ingesters = {
            "manual": ManualIngester,
            "wikidata": WikidataIngester,
        }
        if source not in ingesters:
            raise ValueError(f"Unknown source: {source}")
        return ingesters[source](self).ingest(**kwargs)

    # ── Query methods ────────────────────────────────────────────────

    def find_path(
        self, source: str, target: str, max_depth: int = 6,
        edge_types: list[EdgeType] | None = None,
    ) -> list[list[str]]:
        if isinstance(self.backend, NetworkXBackend):
            g = self.backend.nx_graph
            if edge_types:
                valid_keys = {et.value for et in edge_types}
                edges = [(u, v, k) for u, v, k in g.edges(keys=True)
                         if k in valid_keys]
                subg = g.edge_subgraph(edges)
            else:
                subg = g
            try:
                paths = list(nx.all_simple_paths(
                    subg, source, target, cutoff=max_depth
                ))
                return sorted(paths, key=len)
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                return []
        return []

    def influence_network(
        self, thinker_uid: str, depth: int = 2, direction: str = "both"
    ) -> PhilGraph:
        visited: set[str] = set()
        queue = [(thinker_uid, 0)]
        while queue:
            uid, d = queue.pop(0)
            if uid in visited or d > depth:
                continue
            visited.add(uid)
            nbrs = self.backend.neighbors(
                uid, direction=direction,
                edge_types=[EdgeType.INFLUENCES],
            )
            for nbr in nbrs:
                if nbr not in visited:
                    queue.append((nbr, d + 1))

        sub_backend = self.backend.subgraph(visited)
        return PhilGraph(backend=sub_backend)

    def tradition_overlap(self, trad_a_uid: str, trad_b_uid: str) -> dict:
        def _members(trad_uid: str, ntype: type) -> set[str]:
            edges = self.backend.get_edges(
                target_uid=trad_uid,
                edge_type=EdgeType.BELONGS_TO_TRADITION,
            )
            uids = {e.source_uid for e in edges}
            return {uid for uid in uids
                    if isinstance(self.get_node(uid), ntype)}

        concepts_a = _members(trad_a_uid, Concept)
        concepts_b = _members(trad_b_uid, Concept)
        thinkers_a = _members(trad_a_uid, Thinker)
        thinkers_b = _members(trad_b_uid, Thinker)

        analogous_pairs = []
        for c_a in concepts_a:
            edges = self.backend.get_edges(
                source_uid=c_a, edge_type=EdgeType.ANALOGOUS_TO,
            )
            for e in edges:
                if e.target_uid in concepts_b:
                    analogous_pairs.append((c_a, e.target_uid))

        def jaccard(a: set, b: set) -> float:
            if not a and not b:
                return 0.0
            return len(a & b) / len(a | b)

        return {
            "shared_concepts": list(concepts_a & concepts_b),
            "shared_thinkers": list(thinkers_a & thinkers_b),
            "analogous_pairs": analogous_pairs,
            "jaccard_concepts": jaccard(concepts_a, concepts_b),
            "jaccard_thinkers": jaccard(thinkers_a, thinkers_b),
        }

    def concept_cluster(
        self, concept_uid: str, depth: int = 2,
        edge_types: list[EdgeType] | None = None,
    ) -> dict:
        if edge_types is None:
            edge_types = [
                EdgeType.EXTENDS, EdgeType.SUBSUMES, EdgeType.ANALOGOUS_TO,
                EdgeType.OPPOSES, EdgeType.TRANSLATES_TO,
            ]
        visited: set[str] = set()
        queue = [(concept_uid, 0)]
        while queue:
            uid, d = queue.pop(0)
            if uid in visited or d > depth:
                continue
            visited.add(uid)
            for nbr in self.backend.neighbors(uid, edge_types=edge_types):
                if nbr not in visited:
                    queue.append((nbr, d + 1))

        communities: dict[int, list[str]] = {0: list(visited)}
        centrality: dict[str, float] = {uid: 0.0 for uid in visited}

        if isinstance(self.backend, NetworkXBackend) and len(visited) > 2:
            sub = self.backend.nx_graph.subgraph(visited).to_undirected()
            try:
                from networkx.algorithms.community import greedy_modularity_communities
                raw = greedy_modularity_communities(sub)
                communities = {i: list(c) for i, c in enumerate(raw)}
            except Exception:
                pass
            try:
                centrality = nx.betweenness_centrality(sub)
            except Exception:
                pass

        return {
            "nodes": list(visited),
            "communities": communities,
            "centrality": centrality,
        }

    def temporal_evolution(
        self, concept_uid: str, start_year: int, end_year: int,
        bin_size: int = 50,
    ) -> list[dict]:
        bins = []
        for y in range(start_year, end_year, bin_size):
            y_end = min(y + bin_size, end_year)
            thinkers, texts, related = [], [], set()

            for uid, node in self.iter_nodes("Thinker"):
                if (node.birth_year and node.death_year
                        and node.birth_year <= y_end and node.death_year >= y):
                    edges = self.backend.get_edges(source_uid=uid, target_uid=concept_uid)
                    edges += self.backend.get_edges(source_uid=concept_uid, target_uid=uid)
                    if edges:
                        thinkers.append(uid)

            for uid, node in self.iter_nodes("Text"):
                if node.year and y <= node.year <= y_end:
                    edges = self.backend.get_edges(source_uid=uid, target_uid=concept_uid)
                    if edges:
                        texts.append(uid)

            all_edges = self.backend.get_edges(source_uid=concept_uid)
            all_edges += self.backend.get_edges(target_uid=concept_uid)
            for e in all_edges:
                ts = e.properties.temporal_start
                te = e.properties.temporal_end
                if (ts is None or ts <= y_end) and (te is None or te >= y):
                    other = (e.target_uid if e.source_uid == concept_uid
                             else e.source_uid)
                    if isinstance(self.get_node(other), Concept):
                        related.add(other)

            bins.append({
                "year_start": y, "year_end": y_end,
                "thinkers": thinkers,
                "related_concepts": list(related),
                "texts": texts,
            })
        return bins

    def summary(self) -> dict:
        type_counts: dict[str, int] = {}
        for _, node in self.iter_nodes():
            ntype = type(node).__name__
            type_counts[ntype] = type_counts.get(ntype, 0) + 1

        edge_counts: dict[str, int] = {}
        for edge in self.iter_edges():
            etype = edge.edge_type.value
            edge_counts[etype] = edge_counts.get(etype, 0) + 1

        return {
            "total_nodes": self.backend.node_count(),
            "total_edges": self.backend.edge_count(),
            "nodes_by_type": type_counts,
            "edges_by_type": edge_counts,
        }
