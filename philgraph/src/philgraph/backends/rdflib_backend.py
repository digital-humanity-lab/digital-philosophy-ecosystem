"""RDFLib-based semantic web backend (optional)."""

from __future__ import annotations

from typing import Iterator

try:
    from rdflib import Graph as RDFGraph, Namespace, Literal, URIRef, RDF, RDFS, OWL
    HAS_RDFLIB = True
except ImportError:
    HAS_RDFLIB = False

from philgraph.backends.base import GraphBackend
from philgraph.schema import Edge, EdgeType, NodeBase

PHIL = Namespace("http://philgraph.org/ontology/") if HAS_RDFLIB else None
PHILD = Namespace("http://philgraph.org/data/") if HAS_RDFLIB else None


class RDFLibBackend(GraphBackend):
    """Semantic web backend using RDFLib."""

    def __init__(self) -> None:
        if not HAS_RDFLIB:
            raise ImportError(
                "RDFLibBackend requires rdflib. Install with: pip install philgraph[rdf]"
            )
        self._g = RDFGraph()
        self._g.bind("phil", PHIL)
        self._g.bind("phild", PHILD)
        self._nodes: dict[str, NodeBase] = {}

    def _uri(self, uid: str) -> URIRef:
        return PHILD[uid.replace(":", "/")]

    def add_node(self, node: NodeBase) -> None:
        uri = self._uri(node.uid)
        node_type = type(node).__name__
        self._g.add((uri, RDF.type, PHIL[node_type]))
        self._g.add((uri, RDFS.label, Literal(node.label)))
        for lang, lbl in node.labels_i18n.items():
            self._g.add((uri, RDFS.label, Literal(lbl, lang=lang)))
        self._nodes[node.uid] = node

    def get_node(self, uid: str) -> NodeBase | None:
        return self._nodes.get(uid)

    def remove_node(self, uid: str) -> None:
        uri = self._uri(uid)
        self._g.remove((uri, None, None))
        self._g.remove((None, None, uri))
        self._nodes.pop(uid, None)

    def iter_nodes(self, node_type: str | None = None) -> Iterator[tuple[str, NodeBase]]:
        for uid, node in self._nodes.items():
            if node_type is None or type(node).__name__ == node_type:
                yield uid, node

    def add_edge(self, edge: Edge) -> None:
        src = self._uri(edge.source_uid)
        tgt = self._uri(edge.target_uid)
        pred = PHIL[edge.edge_type.value]
        self._g.add((src, pred, tgt))

    def get_edges(self, source_uid=None, target_uid=None, edge_type=None):
        results = []
        for s, p, o in self._g:
            if not isinstance(o, URIRef):
                continue
            s_uid = str(s).replace(str(PHILD), "").replace("/", ":")
            o_uid = str(o).replace(str(PHILD), "").replace("/", ":")
            if source_uid and s_uid != source_uid:
                continue
            if target_uid and o_uid != target_uid:
                continue
            p_local = str(p).replace(str(PHIL), "")
            try:
                et = EdgeType(p_local)
            except ValueError:
                continue
            if edge_type and et != edge_type:
                continue
            from philgraph.schema import EdgeProperties
            results.append(Edge(source_uid=s_uid, target_uid=o_uid,
                                edge_type=et, properties=EdgeProperties()))
        return results

    def remove_edge(self, source_uid, target_uid, edge_type):
        src = self._uri(source_uid)
        tgt = self._uri(target_uid)
        pred = PHIL[edge_type.value]
        self._g.remove((src, pred, tgt))

    def neighbors(self, uid, direction="both", edge_types=None):
        nbrs = set()
        uri = self._uri(uid)
        if direction in ("out", "both"):
            for _, p, o in self._g.triples((uri, None, None)):
                if isinstance(o, URIRef):
                    o_uid = str(o).replace(str(PHILD), "").replace("/", ":")
                    if o_uid in self._nodes:
                        nbrs.add(o_uid)
        if direction in ("in", "both"):
            for s, p, _ in self._g.triples((None, None, uri)):
                if isinstance(s, URIRef):
                    s_uid = str(s).replace(str(PHILD), "").replace("/", ":")
                    if s_uid in self._nodes:
                        nbrs.add(s_uid)
        return list(nbrs)

    def node_count(self):
        return len(self._nodes)

    def edge_count(self):
        return sum(1 for _, p, o in self._g if isinstance(o, URIRef)
                   and str(p).startswith(str(PHIL)))

    def subgraph(self, uids):
        backend = RDFLibBackend()
        for uid in uids:
            node = self._nodes.get(uid)
            if node:
                backend.add_node(node)
        for edge in self.get_edges():
            if edge.source_uid in uids and edge.target_uid in uids:
                backend.add_edge(edge)
        return backend

    def clear(self):
        self._g = RDFGraph()
        self._nodes.clear()

    def sparql(self, query: str) -> list[dict]:
        results = self._g.query(query)
        return [dict(zip(results.vars, row)) for row in results]

    def export_rdf(self, path: str, format: str = "turtle"):
        self._g.serialize(destination=path, format=format)
