"""Multilingual philosophical corpus builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


@dataclass
class PhilDocument:
    id: str
    title: str
    author: str
    text: str
    language: str
    source: str
    date: Optional[str] = None
    school: Optional[str] = None
    url: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class CorpusBuilder:
    """Build a multilingual philosophical corpus from multiple sources."""

    def __init__(self, cache_dir: str | Path = "~/.cache/philtext/corpus"):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._documents: list[PhilDocument] = []

    def add_local(self, path: str | Path, language: str, **metadata) -> CorpusBuilder:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        self._documents.append(PhilDocument(
            id=f"local:{p.stem}", title=metadata.get("title", p.stem),
            author=metadata.get("author", "Unknown"),
            text=text, language=language, source="local", metadata=metadata,
        ))
        return self

    def add_document(self, doc: PhilDocument) -> CorpusBuilder:
        self._documents.append(doc)
        return self

    def filter(
        self, language: str | None = None,
        school: str | None = None, source: str | None = None,
    ) -> list[PhilDocument]:
        docs = self._documents
        if language:
            docs = [d for d in docs if d.language == language]
        if school:
            docs = [d for d in docs if d.school == school]
        if source:
            docs = [d for d in docs if d.source == source]
        return docs

    @property
    def documents(self) -> list[PhilDocument]:
        return list(self._documents)

    def __len__(self) -> int:
        return len(self._documents)

    def __iter__(self) -> Iterator[PhilDocument]:
        return iter(self._documents)
