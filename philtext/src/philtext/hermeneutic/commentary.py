"""Commentary linking."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Commentary:
    id: str
    commentator: str
    source_ref: str
    source_text: str
    commentary_text: str
    commentary_type: str = ""
    language: str = "en"
    date: str = ""
    metadata: dict = field(default_factory=dict)


class CommentaryLinker:
    """Link commentaries to source texts with fine-grained references."""

    def __init__(self) -> None:
        self._commentaries: list[Commentary] = []
        self._by_ref: dict[str, list[Commentary]] = {}

    def add(self, commentary: Commentary) -> None:
        self._commentaries.append(commentary)
        self._by_ref.setdefault(commentary.source_ref, []).append(commentary)

    def get_commentaries(self, source_ref: str) -> list[Commentary]:
        return self._by_ref.get(source_ref, [])

    def find_by_commentator(self, name: str) -> list[Commentary]:
        return [c for c in self._commentaries if c.commentator == name]

    def coverage_report(self, all_refs: list[str]) -> dict:
        covered = [r for r in all_refs if r in self._by_ref]
        uncovered = [r for r in all_refs if r not in self._by_ref]
        return {
            "total_refs": len(all_refs),
            "covered": len(covered),
            "uncovered": len(uncovered),
            "coverage_pct": len(covered) / max(len(all_refs), 1) * 100,
            "uncovered_refs": uncovered,
        }
