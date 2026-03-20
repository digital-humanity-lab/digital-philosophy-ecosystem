"""In-memory registry for philosophical entities."""

from __future__ import annotations

from philcore.models.concept import Concept
from philcore.models.thinker import Thinker
from philcore.models.tradition import Tradition
from philcore.models.text import Text


class PhilRegistry:
    """Central in-memory registry for concepts, thinkers, traditions, and texts."""

    def __init__(self) -> None:
        self._concepts: dict[str, Concept] = {}
        self._thinkers: dict[str, Thinker] = {}
        self._traditions: dict[str, Tradition] = {}
        self._texts: dict[str, Text] = {}

    def add_concept(self, concept: Concept) -> None:
        self._concepts[concept.id] = concept

    def add_thinker(self, thinker: Thinker) -> None:
        self._thinkers[thinker.id] = thinker

    def add_tradition(self, tradition: Tradition) -> None:
        self._traditions[tradition.id] = tradition

    def add_text(self, text: Text) -> None:
        self._texts[text.id] = text

    def get_concept(self, concept_id: str) -> Concept | None:
        return self._concepts.get(concept_id)

    def get_thinker(self, thinker_id: str) -> Thinker | None:
        return self._thinkers.get(thinker_id)

    def get_tradition(self, tradition_id: str) -> Tradition | None:
        return self._traditions.get(tradition_id)

    def get_text(self, text_id: str) -> Text | None:
        return self._texts.get(text_id)

    @property
    def concepts(self) -> dict[str, Concept]:
        return dict(self._concepts)

    @property
    def thinkers(self) -> dict[str, Thinker]:
        return dict(self._thinkers)

    @property
    def traditions(self) -> dict[str, Tradition]:
        return dict(self._traditions)

    @property
    def texts(self) -> dict[str, Text]:
        return dict(self._texts)

    def __len__(self) -> int:
        return (len(self._concepts) + len(self._thinkers)
                + len(self._traditions) + len(self._texts))
