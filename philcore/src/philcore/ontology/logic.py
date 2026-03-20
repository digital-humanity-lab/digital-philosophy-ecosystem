"""Support for non-classical logical frameworks in philosophical reasoning."""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field


class Koti(int, enum.Enum):
    """The four corners of the catuskoti."""
    AFFIRMATION = 1
    NEGATION = 2
    BOTH = 3
    NEITHER = 4


class CatuskotiEvaluation(BaseModel):
    """Evaluate a proposition under the catuskoti framework.

    In Nagarjuna's Mulamadhyamakakarika, all four koti may be
    rejected (prasajya-pratisedha) to point to sunyata.
    """
    proposition: str
    koti_values: dict[Koti, bool | None] = Field(
        default_factory=lambda: {k: None for k in Koti}
    )
    all_rejected: bool = False
    commentary: str | None = None

    def accepted_koti(self) -> list[Koti]:
        return [k for k, v in self.koti_values.items() if v is True]


class BashoLevel(str, enum.Enum):
    """The three basho in Nishida Kitaro's logic of place."""
    BEING = "u_no_basho"
    RELATIVE_NOTHINGNESS = "soutai_mu"
    ABSOLUTE_NOTHINGNESS = "zettai_mu"


class BashoEnvelopment(BaseModel):
    """A concept situated within Nishida's basho hierarchy."""
    concept_id: str
    basho_level: BashoLevel
    enveloped_by: BashoLevel | None = None
    self_aware: bool = False
    notes: str | None = None


class ParaconsistentValuation(BaseModel):
    """A valuation in a paraconsistent (contradiction-tolerant) logic.

    Uses Belnap's four-valued semantics: {T, F, Both, Neither}.
    """
    proposition: str
    value: str = Field(pattern=r"^(T|F|Both|Neither)$")
    entails: list[str] = Field(default_factory=list)
    notes: str | None = None


class DialecticalMoment(BaseModel):
    """A moment in a Hegelian dialectical progression."""
    thesis: str
    antithesis: str
    synthesis: str | None = None
    aufhebung_notes: str | None = None
    concept_ids: list[str] = Field(default_factory=list)


LogicEvaluation = (
    CatuskotiEvaluation
    | ParaconsistentValuation
    | DialecticalMoment
    | BashoEnvelopment
)
