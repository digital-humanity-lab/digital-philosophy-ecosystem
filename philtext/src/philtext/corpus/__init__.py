"""Multilingual philosophical corpus tools."""
from philtext.corpus.builder import CorpusBuilder, PhilDocument
from philtext.corpus.aligner import TextAligner, AlignedSegment
from philtext.corpus.tokenizers import get_tokenizer
__all__ = ["CorpusBuilder", "PhilDocument", "TextAligner", "AlignedSegment", "get_tokenizer"]
