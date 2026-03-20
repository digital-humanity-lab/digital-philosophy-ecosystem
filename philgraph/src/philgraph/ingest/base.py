"""Base ingester."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from philgraph.graph import PhilGraph
    from philgraph.schema import NodeBase


class BaseIngester(ABC):
    def __init__(self, graph: PhilGraph, config: dict | None = None):
        self.graph = graph
        self.config = config or {}
        self.stats = {"nodes_added": 0, "edges_added": 0, "nodes_merged": 0}

    @abstractmethod
    def ingest(self, **kwargs) -> dict: ...

    def _add_or_merge(self, node: NodeBase) -> str:
        existing_uid = None
        for source, ext_id in node.external_ids.items():
            found = self.graph.resolve_external_id(ext_id, source)
            if found:
                existing_uid = found
                break
        if existing_uid:
            self.graph.merge_node(existing_uid, node)
            self.stats["nodes_merged"] += 1
            return existing_uid
        else:
            self.graph.add_node(node)
            self.stats["nodes_added"] += 1
            return node.uid
