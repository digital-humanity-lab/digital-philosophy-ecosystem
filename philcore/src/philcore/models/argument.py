"""Philosophical argument model with support for non-classical logics."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from philcore.types import LogicFamily


class Proposition(BaseModel):
    """A single proposition (premise or conclusion)."""
    id: str = Field(default_factory=lambda: f"prop/{uuid.uuid4().hex[:8]}")
    text: str
    formal_expression: str | None = None
    source_text_id: str | None = None
    source_location: str | None = None


class LogicalForm(BaseModel):
    """The logical skeleton of an argument."""
    logic_family: LogicFamily = LogicFamily.CLASSICAL
    logical_schema: str | None = None
    formal_representation: str | None = None
    is_valid: bool | None = None
    is_sound: bool | None = None
    notes: str | None = None


class CatuskotiPosition(BaseModel):
    """A position within the Buddhist tetralemma (catuskoti).

    The four koti:
      1. P            (affirmation)
      2. not-P        (negation)
      3. P and not-P  (both)
      4. neither P nor not-P (neither)
    """
    koti: int = Field(ge=1, le=4)
    proposition: Proposition
    accepted: bool | None = None
    commentary: str | None = None


class Argument(BaseModel):
    """A philosophical argument: premises, conclusion, logical form."""
    id: str = Field(default_factory=lambda: f"philcore:argument/{uuid.uuid4().hex[:12]}")
    name: str | None = None
    premises: list[Proposition] = Field(min_length=1)
    conclusion: Proposition
    logical_form: LogicalForm = Field(default_factory=LogicalForm)
    antithesis: Proposition | None = None
    synthesis: Proposition | None = None
    catuskoti_positions: list[CatuskotiPosition] | None = None
    thinker_ids: list[str] = Field(default_factory=list)
    source_text_ids: list[str] = Field(default_factory=list)
    tradition_ids: list[str] = Field(default_factory=list)
    objections: list[Argument] = Field(default_factory=list)
    replies: list[Argument] = Field(default_factory=list)
