"""Argument extraction and analysis."""

from philtext.argument.schemas import Argument, Conclusion, InferenceType, Premise, Proposition
from philtext.argument.extractor import ArgumentExtractor

__all__ = [
    "Argument", "Conclusion", "InferenceType", "Premise", "Proposition",
    "ArgumentExtractor",
]
