"""OWL/RDF serialization for philcore models."""

from __future__ import annotations

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

from philcore.models.concept import Concept
from philcore.models.relation import ConceptRelation
from philcore.models.thinker import Thinker
from philcore.models.tradition import Tradition
from philcore.models.text import Text
from philcore.namespaces import CIDOC, PHILCORE, PHILDATA, SKOS, WD


def _uri(entity_id: str) -> URIRef:
    local = entity_id.removeprefix("philcore:")
    return PHILDATA[local]


class RDFExporter:
    """Export philcore models to an RDF graph."""

    def __init__(self) -> None:
        self.graph = Graph()
        self.graph.bind("philcore", PHILCORE)
        self.graph.bind("phildata", PHILDATA)
        self.graph.bind("skos", SKOS)
        self.graph.bind("cidoc", CIDOC)
        self.graph.bind("wd", WD)

    def add_concept(self, c: Concept) -> URIRef:
        uri = _uri(c.id)
        self.graph.add((uri, RDF.type, PHILCORE["Concept"]))
        self.graph.add((uri, RDF.type, SKOS["Concept"]))
        self.graph.add((uri, RDF.type, CIDOC["E89_Propositional_Object"]))

        for lbl in c.labels:
            lit = Literal(lbl.text, lang=lbl.lang)
            prop = SKOS["prefLabel"] if lbl.is_primary else SKOS["altLabel"]
            self.graph.add((uri, prop, lit))

        if c.definition:
            self.graph.add((uri, SKOS["definition"], Literal(c.definition)))

        for broader_id in c.broader_concept_ids:
            self.graph.add((uri, SKOS["broader"], _uri(broader_id)))

        for tid in c.tradition_ids:
            self.graph.add((uri, PHILCORE["hasTradition"], _uri(tid)))

        if c.formal.logic_family:
            self.graph.add((uri, PHILCORE["logicFamily"],
                           Literal(c.formal.logic_family.value)))

        if wd_id := c.external_ids.get("wikidata"):
            self.graph.add((uri, OWL.sameAs, WD[wd_id]))

        return uri

    def add_tradition(self, t: Tradition) -> URIRef:
        uri = _uri(t.id)
        self.graph.add((uri, RDF.type, PHILCORE["Tradition"]))
        self.graph.add((uri, RDF.type, CIDOC["E4_Period"]))
        for lbl in t.labels:
            self.graph.add((uri, RDFS.label, Literal(lbl.text, lang=lbl.lang)))
        return uri

    def add_thinker(self, th: Thinker) -> URIRef:
        uri = _uri(th.id)
        self.graph.add((uri, RDF.type, PHILCORE["Thinker"]))
        self.graph.add((uri, RDF.type, CIDOC["E21_Person"]))
        for lbl in th.labels:
            self.graph.add((uri, RDFS.label, Literal(lbl.text, lang=lbl.lang)))
        if th.born:
            self.graph.add((uri, PHILCORE["birthDate"],
                           Literal(th.born.isoformat(), datatype=XSD.date)))
        if wd_id := th.external_ids.get("wikidata"):
            self.graph.add((uri, OWL.sameAs, WD[wd_id]))
        return uri

    def add_text(self, t: Text) -> URIRef:
        uri = _uri(t.id)
        self.graph.add((uri, RDF.type, PHILCORE["Text"]))
        self.graph.add((uri, RDF.type, CIDOC["E73_Information_Object"]))
        for lbl in t.labels:
            self.graph.add((uri, RDFS.label, Literal(lbl.text, lang=lbl.lang)))
        for aid in t.author_ids:
            self.graph.add((uri, PHILCORE["authoredBy"], _uri(aid)))
        return uri

    def add_relation(self, rel: ConceptRelation) -> None:
        src = _uri(rel.source_concept_id)
        tgt = _uri(rel.target_concept_id)
        pred = PHILCORE[rel.relation_type.value]
        self.graph.add((src, pred, tgt))

        stmt = BNode()
        self.graph.add((stmt, RDF.type, PHILCORE["ConceptRelation"]))
        self.graph.add((stmt, PHILCORE["source"], src))
        self.graph.add((stmt, PHILCORE["target"], tgt))
        self.graph.add((stmt, PHILCORE["relationType"],
                       Literal(rel.relation_type.value)))
        self.graph.add((stmt, PHILCORE["confidence"],
                       Literal(rel.confidence.value)))
        self.graph.add((stmt, PHILCORE["weight"],
                       Literal(rel.weight, datatype=XSD.float)))

    def serialize(self, format: str = "turtle") -> str:
        return self.graph.serialize(format=format)
