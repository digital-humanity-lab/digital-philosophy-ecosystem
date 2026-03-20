"""Relations between philosophical concepts, including cross-tradition mappings."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from philcore.types import MappingConfidence, RelationType


class ConceptRelation(BaseModel):
    """A directed relation between two concepts."""
    id: str = Field(default_factory=lambda: f"philcore:relation/{uuid.uuid4().hex[:12]}")
    source_concept_id: str
    target_concept_id: str
    relation_type: RelationType
    confidence: MappingConfidence = MappingConfidence.CLOSE
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    justification: str | None = None
    argument_ids: list[str] = Field(default_factory=list)
    source_text_ids: list[str] = Field(default_factory=list)
    asserted_by: str | None = None
    valid_within_tradition_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class CrossTraditionMapping(BaseModel):
    """A higher-level mapping between concepts from distinct traditions.

    Bundles a ConceptRelation with richer metadata about the
    commensurability problem.
    """
    id: str = Field(default_factory=lambda: f"philcore:mapping/{uuid.uuid4().hex[:12]}")
    relation: ConceptRelation
    source_tradition_id: str
    target_tradition_id: str
    preserved_features: list[str] = Field(default_factory=list)
    lost_features: list[str] = Field(default_factory=list)
    gained_features: list[str] = Field(default_factory=list)
    scholarly_references: list[str] = Field(default_factory=list)
