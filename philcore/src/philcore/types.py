"""Shared enumerations and type aliases for philcore."""

from __future__ import annotations

import enum
from typing import Annotated

from pydantic import StringConstraints

LangCode = Annotated[str, StringConstraints(pattern=r"^[a-z]{2,3}(-[A-Za-z0-9]+)*$")]


class Era(str, enum.Enum):
    ANCIENT = "ancient"
    CLASSICAL = "classical"
    EARLY_MODERN = "early_modern"
    MODERN = "modern"
    CONTEMPORARY = "contemporary"


class RelationType(str, enum.Enum):
    EQUIVALENCE = "equivalence"
    OPPOSITION = "opposition"
    SUBSUMPTION = "subsumption"
    ANALOGY = "analogy"
    INFLUENCE = "influence"
    TRANSLATION_PAIR = "translation_pair"
    DIALECTICAL = "dialectical"
    COMPLEMENT = "complement"
    CRITIQUE = "critique"


class MappingConfidence(str, enum.Enum):
    EXACT = "exact"
    CLOSE = "close"
    PARTIAL = "partial"
    SPECULATIVE = "speculative"


class LogicFamily(str, enum.Enum):
    CLASSICAL = "classical"
    INTUITIONISTIC = "intuitionistic"
    PARACONSISTENT = "paraconsistent"
    CATUSKOTI = "catuskoti"
    BASHO = "basho"
    MODAL = "modal"
    DEONTIC = "deontic"
    FUZZY = "fuzzy"
    DIALECTICAL = "dialectical"
