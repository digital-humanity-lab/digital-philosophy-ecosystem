"""Interpretation tracking - making interpretive pluralism explicit."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class Interpretation:
    """A scholarly interpretation of a text passage or concept."""
    id: str
    interpreter: str
    target_text: str
    target_ref: str
    reading: str
    evidence: list[str] = field(default_factory=list)
    school_of_interpretation: str = ""
    published_in: str = ""
    date: Optional[date] = None
    tags: list[str] = field(default_factory=list)

    def conflicts_with(self, other: Interpretation) -> bool:
        return (self.target_ref == other.target_ref
                and self.reading != other.reading)


class InterpretationTracker:
    """Track and compare scholarly interpretations.

    Design principle: 哲学は文献が重要であるがその解釈は属人的である.
    This class makes interpretive differences explicit.
    """

    def __init__(self) -> None:
        self._by_ref: dict[str, list[Interpretation]] = defaultdict(list)
        self._by_interpreter: dict[str, list[Interpretation]] = defaultdict(list)
        self._all: list[Interpretation] = []

    def add(self, interpretation: Interpretation) -> None:
        self._all.append(interpretation)
        self._by_ref[interpretation.target_ref].append(interpretation)
        self._by_interpreter[interpretation.interpreter].append(interpretation)

    def add_batch(self, interpretations: list[Interpretation]) -> None:
        for interp in interpretations:
            self.add(interp)

    def find_conflicts(
        self, target_ref: str
    ) -> list[tuple[Interpretation, Interpretation]]:
        interps = self._by_ref.get(target_ref, [])
        conflicts = []
        for i, a in enumerate(interps):
            for b in interps[i + 1:]:
                if a.conflicts_with(b):
                    conflicts.append((a, b))
        return conflicts

    def get_by_school(self, school: str) -> list[Interpretation]:
        return [i for i in self._all if i.school_of_interpretation == school]

    def interpreters_for(self, target_ref: str) -> list[str]:
        return list({i.interpreter for i in self._by_ref.get(target_ref, [])})

    def summarize_debate(self, target_ref: str) -> dict:
        interps = self._by_ref.get(target_ref, [])
        if not interps:
            return {"target_ref": target_ref, "status": "no_interpretations_found"}

        by_school: dict[str, list[Interpretation]] = defaultdict(list)
        for interp in interps:
            by_school[interp.school_of_interpretation or "unspecified"].append(interp)

        conflicts = self.find_conflicts(target_ref)
        return {
            "target_ref": target_ref,
            "target_text": interps[0].target_text,
            "num_interpretations": len(interps),
            "interpreters": [i.interpreter for i in interps],
            "schools_represented": list(by_school.keys()),
            "readings_by_school": {
                school: [i.reading for i in group]
                for school, group in by_school.items()
            },
            "num_conflicts": len(conflicts),
            "conflict_pairs": [
                {"a": a.interpreter, "b": b.interpreter,
                 "a_reading": a.reading, "b_reading": b.reading}
                for a, b in conflicts
            ],
        }
