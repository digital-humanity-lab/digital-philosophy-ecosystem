"""Manual ingester from YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml

from philgraph.ingest.base import BaseIngester
from philgraph.schema import (
    Edge, EdgeProperties, EdgeType, ConsensusLevel, NODE_TYPE_MAP,
)


class ManualIngester(BaseIngester):
    """Ingest from curated YAML files."""

    def ingest(
        self, yaml_paths: list[str | Path] | None = None,
        yaml_dir: str | Path | None = None, **kwargs,
    ) -> dict:
        paths: list[Path] = []
        if yaml_paths:
            paths.extend(Path(p) for p in yaml_paths)
        if yaml_dir:
            d = Path(yaml_dir)
            paths.extend(d.glob("*.yml"))
            paths.extend(d.glob("*.yaml"))

        for path in paths:
            with open(path) as f:
                data = yaml.safe_load(f)
            if data:
                self._process_yaml(data)
        return self.stats

    def _process_yaml(self, data: dict) -> None:
        for node_data in data.get("nodes", []):
            node_type = node_data.pop("type")
            cls = NODE_TYPE_MAP.get(node_type)
            if cls is None:
                continue
            node = cls(**node_data)
            if "manual" not in node.provenance:
                node.provenance.append("manual")
            self._add_or_merge(node)

        for edge_data in data.get("edges", []):
            props_data = edge_data.get("properties", {})
            if "consensus" in props_data:
                props_data["consensus"] = ConsensusLevel(props_data["consensus"])
            props = EdgeProperties(**props_data)
            edge = Edge(
                source_uid=edge_data["source"],
                target_uid=edge_data["target"],
                edge_type=EdgeType(edge_data["type"]),
                properties=props,
            )
            self.graph.add_edge(edge)
            self.stats["edges_added"] += 1
