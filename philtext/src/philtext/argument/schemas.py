"""Argument data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class InferenceType(str, Enum):
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    TRANSCENDENTAL = "transcendental"
    DIALECTICAL = "dialectical"
    REDUCTIO = "reductio_ad_absurdum"


@dataclass(frozen=True)
class Proposition:
    text: str
    span: tuple[int, int] = (0, 0)
    confidence: float = 1.0
    language: str = "en"


@dataclass(frozen=True)
class Premise(Proposition):
    is_implicit: bool = False
    reconstruction_note: str = ""


@dataclass(frozen=True)
class Conclusion(Proposition):
    is_intermediate: bool = False


@dataclass
class Argument:
    premises: list[Premise]
    conclusion: Conclusion
    inference_type: InferenceType
    source_text: str
    confidence: float
    sub_arguments: list[Argument] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_standard_form(self) -> str:
        lines = []
        for i, p in enumerate(self.premises, 1):
            implicit = " [implicit]" if p.is_implicit else ""
            lines.append(f"P{i}. {p.text}{implicit}")
        lines.append("---")
        lines.append(f"C.  {self.conclusion.text}  [{self.inference_type.value}]")
        return "\n".join(lines)
