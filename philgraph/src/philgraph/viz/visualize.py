"""Visualization for PhilGraph."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from philgraph.graph import PhilGraph

TRADITION_COLORS = {
    "analytic": "#3498db", "continental": "#e74c3c",
    "confucianism": "#f39c12", "daoism": "#27ae60",
    "buddhism": "#8e44ad", "indian": "#e67e22",
    "islamic": "#1abc9c", "african": "#d35400",
    "japanese": "#c0392b", "pragmatism": "#2980b9",
    "default": "#95a5a6",
}

NODE_TYPE_SHAPES = {
    "Concept": "dot", "Thinker": "diamond", "Text": "square",
    "Tradition": "triangle", "Argument": "star", "Era": "box",
    "Institution": "hexagon", "Language": "ellipse",
}


class PhilGraphViz:
    def __init__(self, graph: PhilGraph):
        self.graph = graph

    def to_pyvis(
        self, subgraph: PhilGraph | None = None,
        height: str = "800px", width: str = "100%",
    ) -> Any:
        try:
            from pyvis.network import Network
        except ImportError:
            raise ImportError(
                "Visualization requires pyvis. Install with: pip install philgraph[viz]"
            )
        g = subgraph or self.graph
        net = Network(height=height, width=width, directed=True)
        net.barnes_hut(gravity=-5000, spring_length=200)

        for uid, node in g.iter_nodes():
            ntype = type(node).__name__
            color = self._node_color(node)
            shape = NODE_TYPE_SHAPES.get(ntype, "dot")
            net.add_node(uid, label=node.label, color=color,
                         shape=shape, size=15)

        for edge in g.iter_edges():
            width = edge.properties.confidence * 3
            net.add_edge(edge.source_uid, edge.target_uid,
                         title=edge.edge_type.value, width=width)
        return net

    def to_d3_json(
        self, subgraph: PhilGraph | None = None, path: str | None = None
    ) -> dict:
        g = subgraph or self.graph
        nodes = [
            {"id": uid, "label": node.label,
             "type": type(node).__name__,
             "color": self._node_color(node)}
            for uid, node in g.iter_nodes()
        ]
        links = [
            {"source": e.source_uid, "target": e.target_uid,
             "type": e.edge_type.value,
             "confidence": e.properties.confidence}
            for e in g.iter_edges()
        ]
        data = {"nodes": nodes, "links": links}
        if path:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        return data

    def temporal_timeline(
        self, concept_uid: str, start_year: int, end_year: int,
        output_path: str | None = None,
    ) -> Any:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Timeline requires matplotlib. Install with: pip install philgraph[viz]"
            )
        evolution = self.graph.temporal_evolution(
            concept_uid, start_year, end_year
        )
        concept = self.graph.get_node(concept_uid)
        label = concept.label if concept else concept_uid

        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        fig.suptitle(f"Temporal Evolution: {label}", fontsize=14)

        years = [(b["year_start"] + b["year_end"]) / 2 for b in evolution]
        bin_w = (end_year - start_year) / max(len(evolution), 1) * 0.8

        axes[0].bar(years, [len(b["thinkers"]) for b in evolution],
                    width=bin_w, alpha=0.7, color="#3498db")
        axes[0].set_ylabel("Active Thinkers")

        axes[1].bar(years, [len(b["related_concepts"]) for b in evolution],
                    width=bin_w, alpha=0.7, color="#e74c3c")
        axes[1].set_ylabel("Related Concepts")

        axes[2].bar(years, [len(b["texts"]) for b in evolution],
                    width=bin_w, alpha=0.7, color="#27ae60")
        axes[2].set_ylabel("Texts")
        axes[2].set_xlabel("Year")

        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    def _node_color(self, node: Any) -> str:
        if hasattr(node, "tradition_uids") and node.tradition_uids:
            trad = self.graph.get_node(node.tradition_uids[0])
            if trad:
                key = trad.label.lower().replace(" ", "_")
                return TRADITION_COLORS.get(key, TRADITION_COLORS["default"])
        return TRADITION_COLORS["default"]
