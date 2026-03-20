"""NetworkX-based in-memory graph backend."""

from __future__ import annotations

from typing import Iterator

import networkx as nx

from philgraph.backends.base import GraphBackend
from philgraph.schema import Edge, EdgeType, NodeBase


class NetworkXBackend(GraphBackend):
    def __init__(self) -> None:
        self._g = nx.MultiDiGraph()

    def add_node(self, node: NodeBase) -> None:
        self._g.add_node(node.uid, data=node, node_type=type(node).__name__)

    def get_node(self, uid: str) -> NodeBase | None:
        if uid in self._g:
            return self._g.nodes[uid]["data"]
        return None

    def remove_node(self, uid: str) -> None:
        if uid in self._g:
            self._g.remove_node(uid)

    def iter_nodes(self, node_type: str | None = None) -> Iterator[tuple[str, NodeBase]]:
        for uid, attrs in self._g.nodes(data=True):
            if node_type is None or attrs.get("node_type") == node_type:
                yield uid, attrs["data"]

    def add_edge(self, edge: Edge) -> None:
        self._g.add_edge(
            edge.source_uid, edge.target_uid,
            key=edge.edge_type.value, data=edge,
        )

    def get_edges(
        self, source_uid: str | None = None,
        target_uid: str | None = None,
        edge_type: EdgeType | None = None,
    ) -> list[Edge]:
        results = []
        for u, v, key, data in self._g.edges(keys=True, data=True):
            if source_uid and u != source_uid:
                continue
            if target_uid and v != target_uid:
                continue
            if edge_type and key != edge_type.value:
                continue
            if "data" in data:
                results.append(data["data"])
        return results

    def remove_edge(self, source_uid: str, target_uid: str, edge_type: EdgeType) -> None:
        if self._g.has_edge(source_uid, target_uid, key=edge_type.value):
            self._g.remove_edge(source_uid, target_uid, key=edge_type.value)

    def neighbors(
        self, uid: str, direction: str = "both",
        edge_types: list[EdgeType] | None = None,
    ) -> list[str]:
        nbrs: set[str] = set()
        if direction in ("out", "both"):
            for _, v, key in self._g.out_edges(uid, keys=True):
                if edge_types is None or EdgeType(key) in edge_types:
                    nbrs.add(v)
        if direction in ("in", "both"):
            for u, _, key in self._g.in_edges(uid, keys=True):
                if edge_types is None or EdgeType(key) in edge_types:
                    nbrs.add(u)
        return list(nbrs)

    def node_count(self) -> int:
        return self._g.number_of_nodes()

    def edge_count(self) -> int:
        return self._g.number_of_edges()

    def subgraph(self, uids: set[str]) -> NetworkXBackend:
        backend = NetworkXBackend()
        backend._g = self._g.subgraph(uids).copy()
        return backend

    def clear(self) -> None:
        self._g.clear()

    @property
    def nx_graph(self) -> nx.MultiDiGraph:
        return self._g
