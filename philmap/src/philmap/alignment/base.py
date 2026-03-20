"""Base class for alignment methods."""

from __future__ import annotations

from abc import ABC, abstractmethod

from philmap.concept import Concept, ConceptMapping


class AlignmentMethod(ABC):
    @abstractmethod
    def align(self, source: Concept, target: Concept) -> ConceptMapping: ...

    @abstractmethod
    def align_one_to_many(
        self, source: Concept, candidates: list[Concept], top_k: int = 5
    ) -> list[ConceptMapping]: ...
