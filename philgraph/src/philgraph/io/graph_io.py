"""Graph import/export in multiple formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from philgraph.graph import PhilGraph


class GraphIO:
    def __init__(self, graph: PhilGraph):
        self.graph = graph

    def export_graphml(self, path: str | Path) -> None:
        import networkx as nx
        from philgraph.backends.networkx_backend import NetworkXBackend
        if isinstance(self.graph.backend, NetworkXBackend):
            nx.write_graphml(self.graph.backend.nx_graph, str(path))
        else:
            g = nx.MultiDiGraph()
            for uid, node in self.graph.iter_nodes():
                g.add_node(uid, label=node.label, type=type(node).__name__)
            for edge in self.graph.iter_edges():
                g.add_edge(edge.source_uid, edge.target_uid,
                           key=edge.edge_type.value)
            nx.write_graphml(g, str(path))

    def export_jsonld(self, path: str | Path) -> None:
        context = {
            "@context": {
                "phil": "http://philgraph.org/ontology/",
                "phild": "http://philgraph.org/data/",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "label": "rdfs:label",
            }
        }
        nodes = []
        for uid, node in self.graph.iter_nodes():
            nodes.append({
                "@id": f"phild:{uid}",
                "@type": f"phil:{type(node).__name__}",
                "label": node.label,
            })
        doc = {**context, "@graph": nodes}
        with open(path, "w") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)

    def export_cypher(self, path: str | Path) -> None:
        lines = []
        for uid, node in self.graph.iter_nodes():
            ntype = type(node).__name__
            label_esc = node.label.replace('"', '\\"')
            lines.append(
                f'CREATE (:{ntype} {{uid: "{uid}", label: "{label_esc}"}})'
            )
        for edge in self.graph.iter_edges():
            rel = edge.edge_type.value.upper()
            lines.append(
                f'MATCH (a {{uid: "{edge.source_uid}"}}), '
                f'(b {{uid: "{edge.target_uid}"}}) '
                f'CREATE (a)-[:{rel} {{confidence: {edge.properties.confidence}}}]->(b)'
            )
        with open(path, "w") as f:
            f.write(";\n".join(lines) + ";\n")

    def export_rdf(self, path: str | Path, format: str = "turtle") -> None:
        from philgraph.backends.rdflib_backend import RDFLibBackend
        if isinstance(self.graph.backend, RDFLibBackend):
            self.graph.backend.export_rdf(str(path), format=format)
        else:
            rdf = RDFLibBackend()
            for uid, node in self.graph.iter_nodes():
                rdf.add_node(node)
            for edge in self.graph.iter_edges():
                rdf.add_edge(edge)
            rdf.export_rdf(str(path), format=format)
