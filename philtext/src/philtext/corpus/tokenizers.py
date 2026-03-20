"""Language-specific tokenizer wrappers."""

from __future__ import annotations

from typing import Protocol


class Tokenizer(Protocol):
    def tokenize(self, text: str) -> list[str]: ...


class SimpleTokenizer:
    """Whitespace-based fallback tokenizer."""
    def tokenize(self, text: str) -> list[str]:
        return text.split()


class SudachiTokenizer:
    def __init__(self, mode: str = "C"):
        try:
            from sudachipy import Dictionary, Tokenizer as SudachiTok
            self._dict = Dictionary()
            self._mode_map = {
                "A": SudachiTok.SplitMode.A,
                "B": SudachiTok.SplitMode.B,
                "C": SudachiTok.SplitMode.C,
            }
            self._mode = self._mode_map.get(mode, SudachiTok.SplitMode.C)
        except ImportError:
            raise ImportError(
                "Japanese tokenization requires sudachipy. "
                "Install with: pip install philtext[ja]"
            )

    def tokenize(self, text: str) -> list[str]:
        tok = self._dict.create()
        return [m.surface() for m in tok.tokenize(text, self._mode)]


class JiebaTokenizer:
    def __init__(self):
        try:
            import jieba  # noqa: F401
        except ImportError:
            raise ImportError(
                "Chinese tokenization requires jieba. "
                "Install with: pip install philtext[zh]"
            )

    def tokenize(self, text: str) -> list[str]:
        import jieba
        return list(jieba.cut(text))


class SpaCyTokenizer:
    def __init__(self, model: str):
        try:
            import spacy
            self.nlp = spacy.load(model, disable=["ner", "parser"])
        except ImportError:
            raise ImportError(
                "spaCy tokenization requires spacy. "
                "Install with: pip install philtext[nlp]"
            )

    def tokenize(self, text: str) -> list[str]:
        return [tok.text for tok in self.nlp(text)]


class CLTKTokenizer:
    def __init__(self, language: str):
        self.language = language
        try:
            from cltk.tokenizers.word import WordTokenizer  # noqa: F401
        except ImportError:
            raise ImportError(
                f"Classical language tokenization requires cltk. "
                f"Install with: pip install philtext[classical]"
            )

    def tokenize(self, text: str) -> list[str]:
        from cltk.tokenizers.word import WordTokenizer
        tok = WordTokenizer(self.language)
        return tok.tokenize(text)


def get_tokenizer(language: str) -> Tokenizer:
    """Factory: return the appropriate tokenizer for a language."""
    _MAP: dict[str, callable] = {
        "en": lambda: SpaCyTokenizer("en_core_web_sm"),
        "de": lambda: SpaCyTokenizer("de_core_news_sm"),
        "fr": lambda: SpaCyTokenizer("fr_core_news_sm"),
        "ja": lambda: SudachiTokenizer("C"),
        "zh": lambda: JiebaTokenizer(),
        "la": lambda: CLTKTokenizer("lat"),
        "grc": lambda: CLTKTokenizer("grc"),
        "sa": lambda: CLTKTokenizer("san"),
    }
    factory = _MAP.get(language)
    if factory is None:
        return SimpleTokenizer()
    try:
        return factory()
    except ImportError:
        return SimpleTokenizer()
