"""Philosophical concept model with multilingual support."""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from philcore.types import Era, LangCode, LogicFamily


class ConceptLabel(BaseModel):
    """A name/label for a concept in a specific language."""
    text: str
    lang: LangCode
    script: str | None = None
    transliteration: str | None = None
    is_primary: bool = False


class TemporalContext(BaseModel):
    """When a concept was active/formulated."""
    era: Era | None = None
    date_earliest: date | None = None
    date_latest: date | None = None
    period_label: str | None = None


class FormalProperty(BaseModel):
    """Formal/logical structure of a concept, if applicable."""
    logic_family: LogicFamily = LogicFamily.CLASSICAL
    formal_definition: str | None = None
    arity: int | None = None
    is_reflexive: bool | None = None
    is_symmetric: bool | None = None
    is_transitive: bool | None = None
    custom_properties: dict[str, Any] = Field(default_factory=dict)


class Concept(BaseModel):
    """A philosophical concept.

    Examples: 和辻の「間柄」(aidagara), Heidegger's Dasein, Ubuntu, Confucian 仁 (ren).
    """
    id: str = Field(default_factory=lambda: f"philcore:concept/{uuid.uuid4().hex[:12]}")
    labels: list[ConceptLabel] = Field(min_length=1)
    definition: str | None = None
    description: str | None = None
    tradition_ids: list[str] = Field(default_factory=list)
    thinker_ids: list[str] = Field(default_factory=list)
    source_text_ids: list[str] = Field(default_factory=list)
    temporal: TemporalContext = Field(default_factory=TemporalContext)
    formal: FormalProperty = Field(default_factory=FormalProperty)
    broader_concept_ids: list[str] = Field(default_factory=list)
    narrower_concept_ids: list[str] = Field(default_factory=list)
    notes: dict[str, str] = Field(default_factory=dict)
    external_ids: dict[str, str] = Field(default_factory=dict)

    @property
    def primary_label(self) -> ConceptLabel:
        for lbl in self.labels:
            if lbl.is_primary:
                return lbl
        return self.labels[0]

    def label_in(self, lang: str) -> str | None:
        for lbl in self.labels:
            if lbl.lang == lang:
                return lbl.text
        return None
