"""JSON-LD serialization with context management."""

from __future__ import annotations

import json
from typing import Any

from philcore.models.concept import Concept

JSONLD_CONTEXT = {
    "@context": {
        "philcore": "https://philcore.org/ontology/",
        "skos": "http://www.w3.org/2004/02/skos/core#",
        "cidoc": "http://www.cidoc-crm.org/cidoc-crm/",
        "schema": "https://schema.org/",
        "wd": "http://www.wikidata.org/entity/",
        "label": {"@id": "skos:prefLabel", "@container": "@language"},
        "altLabel": {"@id": "skos:altLabel", "@container": "@language"},
        "definition": "skos:definition",
        "broader": {"@id": "skos:broader", "@type": "@id"},
        "tradition": {"@id": "philcore:hasTradition", "@type": "@id"},
        "thinker": {"@id": "philcore:hasThinker", "@type": "@id"},
        "logicFamily": "philcore:logicFamily",
        "confidence": "philcore:mappingConfidence",
        "sameAs": {"@id": "owl:sameAs", "@type": "@id"},
    }
}


def concept_to_jsonld(c: Concept) -> dict[str, Any]:
    """Convert a Concept to a JSON-LD document."""
    label_map: dict[str, str] = {}
    alt_labels: dict[str, list[str]] = {}
    for lbl in c.labels:
        if lbl.is_primary:
            label_map[lbl.lang] = lbl.text
        else:
            alt_labels.setdefault(lbl.lang, []).append(lbl.text)

    doc: dict[str, Any] = {
        **JSONLD_CONTEXT,
        "@id": f"https://philcore.org/data/{c.id.removeprefix('philcore:')}",
        "@type": ["philcore:Concept", "skos:Concept",
                  "cidoc:E89_Propositional_Object"],
        "label": label_map,
    }
    if alt_labels:
        doc["altLabel"] = alt_labels
    if c.definition:
        doc["definition"] = c.definition
    if c.broader_concept_ids:
        doc["broader"] = [
            f"https://philcore.org/data/{bid.removeprefix('philcore:')}"
            for bid in c.broader_concept_ids
        ]
    if c.formal.logic_family:
        doc["logicFamily"] = c.formal.logic_family.value
    if wd := c.external_ids.get("wikidata"):
        doc["sameAs"] = f"http://www.wikidata.org/entity/{wd}"
    return doc


def to_jsonld_string(doc: dict[str, Any], indent: int = 2) -> str:
    return json.dumps(doc, ensure_ascii=False, indent=indent)
