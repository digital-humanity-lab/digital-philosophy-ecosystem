"""Philosophical tradition / school."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from philcore.models.concept import ConceptLabel, TemporalContext


class Tradition(BaseModel):
    """A philosophical tradition or school.

    Examples: 京都学派, Nyaya, Continental phenomenology, Ubuntu philosophy.
    """
    id: str = Field(default_factory=lambda: f"philcore:tradition/{uuid.uuid4().hex[:12]}")
    labels: list[ConceptLabel] = Field(min_length=1)
    description: str | None = None
    temporal: TemporalContext = Field(default_factory=TemporalContext)
    region: str | None = None
    parent_tradition_id: str | None = None
    influenced_by_ids: list[str] = Field(default_factory=list)
    key_concept_ids: list[str] = Field(default_factory=list)
    external_ids: dict[str, str] = Field(default_factory=dict)
