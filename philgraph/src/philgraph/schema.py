"""Graph schema: node types, edge types, and constraints."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


@dataclass
class NodeBase:
    uid: str
    label: str
    labels_i18n: dict[str, str] = field(default_factory=dict)
    external_ids: dict[str, str] = field(default_factory=dict)
    provenance: list[str] = field(default_factory=list)


@dataclass
class Concept(NodeBase):
    definition: Optional[str] = None
    tradition_uids: list[str] = field(default_factory=list)
    first_attested_year: Optional[int] = None
    keywords: list[str] = field(default_factory=list)


@dataclass
class Thinker(NodeBase):
    birth_year: Optional[int] = None
    death_year: Optional[int] = None
    nationality: Optional[str] = None
    languages: list[str] = field(default_factory=list)
    tradition_uids: list[str] = field(default_factory=list)
    institution_uids: list[str] = field(default_factory=list)


@dataclass
class Text(NodeBase):
    author_uids: list[str] = field(default_factory=list)
    year: Optional[int] = None
    language: Optional[str] = None
    doi: Optional[str] = None
    genre: Optional[str] = None


@dataclass
class Tradition(NodeBase):
    region: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    parent_tradition_uid: Optional[str] = None


@dataclass
class Argument(NodeBase):
    form: Optional[str] = None
    premises: list[str] = field(default_factory=list)
    conclusion: Optional[str] = None
    source_text_uid: Optional[str] = None


@dataclass
class Era(NodeBase):
    start_year: int = 0
    end_year: int = 0
    region: Optional[str] = None


@dataclass
class Institution(NodeBase):
    city: Optional[str] = None
    country: Optional[str] = None
    founded_year: Optional[int] = None
    institution_type: Optional[str] = None


@dataclass
class Language(NodeBase):
    iso_639_1: str = ""
    script: Optional[str] = None
    philosophical_tradition_uids: list[str] = field(default_factory=list)


class EdgeType(str, Enum):
    INFLUENCES = "influences"
    OPPOSES = "opposes"
    EXTENDS = "extends"
    REINTERPRETS = "reinterprets"
    TRANSLATES_TO = "translates_to"
    ANALOGOUS_TO = "analogous_to"
    SUBSUMES = "subsumes"
    USES_IN_ARGUMENT = "uses_in_argument"
    AUTHORED_BY = "authored_by"
    BELONGS_TO_TRADITION = "belongs_to_tradition"
    CONTEMPORARY_WITH = "contemporary_with"
    AFFILIATED_WITH = "affiliated_with"
    WRITTEN_IN = "written_in"
    PART_OF_ERA = "part_of_era"
    CITES = "cites"


class ConsensusLevel(str, Enum):
    ESTABLISHED = "established"
    DEBATED = "debated"
    SPECULATIVE = "speculative"
    NOVEL = "novel"


@dataclass
class EdgeProperties:
    confidence: float = 1.0
    evidence_sources: list[str] = field(default_factory=list)
    consensus: ConsensusLevel = ConsensusLevel.ESTABLISHED
    temporal_start: Optional[int] = None
    temporal_end: Optional[int] = None
    notes: Optional[str] = None
    provenance: Optional[str] = None


@dataclass
class Edge:
    source_uid: str
    target_uid: str
    edge_type: EdgeType
    properties: EdgeProperties = field(default_factory=EdgeProperties)


EDGE_CONSTRAINTS: dict[EdgeType, list[tuple[str, str]]] = {
    EdgeType.INFLUENCES: [("Thinker", "Thinker"), ("Concept", "Concept"),
                          ("Tradition", "Tradition"), ("Text", "Thinker")],
    EdgeType.OPPOSES: [("Concept", "Concept"), ("Thinker", "Thinker"),
                       ("Argument", "Argument")],
    EdgeType.EXTENDS: [("Concept", "Concept"), ("Argument", "Argument")],
    EdgeType.REINTERPRETS: [("Thinker", "Concept")],
    EdgeType.TRANSLATES_TO: [("Concept", "Concept")],
    EdgeType.ANALOGOUS_TO: [("Concept", "Concept")],
    EdgeType.SUBSUMES: [("Concept", "Concept")],
    EdgeType.USES_IN_ARGUMENT: [("Argument", "Concept")],
    EdgeType.AUTHORED_BY: [("Text", "Thinker")],
    EdgeType.BELONGS_TO_TRADITION: [("Thinker", "Tradition"), ("Concept", "Tradition")],
    EdgeType.CONTEMPORARY_WITH: [("Thinker", "Thinker")],
    EdgeType.AFFILIATED_WITH: [("Thinker", "Institution")],
    EdgeType.WRITTEN_IN: [("Text", "Language")],
    EdgeType.PART_OF_ERA: [("Thinker", "Era"), ("Tradition", "Era")],
    EdgeType.CITES: [("Text", "Text")],
}

NODE_TYPE_MAP: dict[str, type] = {
    "Concept": Concept, "Thinker": Thinker, "Text": Text,
    "Tradition": Tradition, "Argument": Argument, "Era": Era,
    "Institution": Institution, "Language": Language,
}
