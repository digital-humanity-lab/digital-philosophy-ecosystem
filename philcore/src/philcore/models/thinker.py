"""Philosopher / thinker model."""

from __future__ import annotations

import uuid
from datetime import date

from pydantic import BaseModel, Field

from philcore.models.concept import ConceptLabel


class Thinker(BaseModel):
    """A philosopher or thinker.

    Examples: 西田幾多郎, Nagarjuna, Kwame Gyekye, Simone de Beauvoir.
    """
    id: str = Field(default_factory=lambda: f"philcore:thinker/{uuid.uuid4().hex[:12]}")
    labels: list[ConceptLabel] = Field(min_length=1)
    born: date | None = None
    died: date | None = None
    floruit: str | None = None
    tradition_ids: list[str] = Field(default_factory=list)
    key_concept_ids: list[str] = Field(default_factory=list)
    key_text_ids: list[str] = Field(default_factory=list)
    influenced_by_ids: list[str] = Field(default_factory=list)
    influenced_ids: list[str] = Field(default_factory=list)
    external_ids: dict[str, str] = Field(default_factory=dict)
