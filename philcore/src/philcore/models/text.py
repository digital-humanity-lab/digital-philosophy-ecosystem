"""Philosophical text / work model."""

from __future__ import annotations

import uuid
from datetime import date

from pydantic import BaseModel, Field

from philcore.models.concept import ConceptLabel


class TextPassage(BaseModel):
    """A specific passage within a text."""
    location: str
    content: str | None = None
    lang: str | None = None
    concept_ids: list[str] = Field(default_factory=list)


class Text(BaseModel):
    """A philosophical text or work.

    Examples: 善の研究, Sein und Zeit, Zhongyong, Republic.
    """
    id: str = Field(default_factory=lambda: f"philcore:text/{uuid.uuid4().hex[:12]}")
    labels: list[ConceptLabel] = Field(min_length=1)
    author_ids: list[str] = Field(default_factory=list)
    date_composed: date | None = None
    date_composed_approx: str | None = None
    tradition_ids: list[str] = Field(default_factory=list)
    lang: str | None = None
    passages: list[TextPassage] = Field(default_factory=list)
    external_ids: dict[str, str] = Field(default_factory=dict)
