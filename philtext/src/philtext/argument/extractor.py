"""Argument extraction from philosophical texts."""

from __future__ import annotations

import re
from typing import Literal

from philtext.argument.schemas import Argument, Conclusion, InferenceType, Premise
from philtext.argument.rules import ARGUMENT_INDICATORS


class ArgumentExtractor:
    """Extract argument structures from philosophical text.

    Supports rule-based, LLM-based, and hybrid extraction.
    """

    def __init__(
        self,
        method: Literal["rule", "llm", "hybrid"] = "rule",
        llm_model: str = "claude-sonnet-4-20250514",
        language: str = "en",
    ):
        self.method = method
        self.language = language
        self.llm_model = llm_model
        self._rule_indicators = ARGUMENT_INDICATORS.get(language, {})

    def extract(self, text: str) -> list[Argument]:
        if self.method == "rule":
            return self._rule_extract(text)
        elif self.method == "llm":
            return self._llm_extract(text)
        else:
            candidates = self._identify_argument_passages(text)
            if not candidates:
                return self._rule_extract(text)
            arguments: list[Argument] = []
            for passage in candidates:
                arguments.extend(self._rule_extract(passage))
            return arguments

    def _identify_argument_passages(self, text: str) -> list[str]:
        segments = []
        paragraphs = text.split("\n\n")
        for para in paragraphs:
            score = 0
            for category, patterns in self._rule_indicators.items():
                for pat in patterns:
                    if re.search(pat, para, re.IGNORECASE):
                        score += 1
                        break
            if score >= 2:
                segments.append(para)
        return segments

    def _rule_extract(self, text: str) -> list[Argument]:
        premise_pats = self._rule_indicators.get("premise", [])
        conclusion_pats = self._rule_indicators.get("conclusion", [])

        # First try clause-level splitting within sentences
        clauses = self._split_clauses(text)
        premise_clauses: list[str] = []
        conclusion_clause: str | None = None

        for clause in clauses:
            is_conclusion = any(
                re.search(p, clause, re.IGNORECASE) for p in conclusion_pats
            )
            is_premise = any(
                re.search(p, clause, re.IGNORECASE) for p in premise_pats
            )
            if is_conclusion:
                # Strip the indicator word from the conclusion text
                cleaned = clause.strip()
                for p in conclusion_pats:
                    cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE).strip()
                cleaned = cleaned.strip(" ,.")
                conclusion_clause = cleaned if cleaned else clause.strip()
            elif is_premise:
                cleaned = clause.strip()
                for p in premise_pats:
                    cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE).strip()
                cleaned = cleaned.strip(" ,.")
                if cleaned:
                    premise_clauses.append(cleaned)

        if premise_clauses and conclusion_clause:
            premises = [
                Premise(text=s.strip(), language=self.language)
                for s in premise_clauses
            ]
            conclusion = Conclusion(
                text=conclusion_clause.strip(), language=self.language
            )
            return [Argument(
                premises=premises, conclusion=conclusion,
                inference_type=InferenceType.DEDUCTIVE,
                source_text=text, confidence=0.5,
            )]
        return []

    def _split_clauses(self, text: str) -> list[str]:
        """Split text into clauses (by commas, semicolons, 'and', and sentence boundaries)."""
        if self.language in ("ja", "zh"):
            parts = re.split(r'[、，。；]', text)
        else:
            parts = re.split(r'[,;]\s*|\band\b\s+', text)
            # Also split by sentence boundaries
            expanded = []
            for part in parts:
                expanded.extend(
                    s.strip() for s in re.split(r'(?<=[.!?])\s+', part) if s.strip()
                )
            parts = expanded
        return [p.strip() for p in parts if p.strip()]

    def _llm_extract(self, text: str) -> list[Argument]:
        try:
            from litellm import completion
        except ImportError:
            raise ImportError(
                "LLM extraction requires litellm. "
                "Install with: pip install philtext[llm]"
            )
        import json
        prompt = (
            f"Extract all argument structures from this philosophical text "
            f"(language: {self.language}). Return JSON array with premises, "
            f"conclusion, inference_type, confidence.\n\n{text}"
        )
        response = completion(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        try:
            raw = json.loads(response.choices[0].message.content)
            if not isinstance(raw, list):
                raw = [raw]
        except (json.JSONDecodeError, AttributeError):
            return []

        arguments = []
        for item in raw:
            premises = [
                Premise(
                    text=p.get("text", p) if isinstance(p, dict) else str(p),
                    is_implicit=p.get("is_implicit", False) if isinstance(p, dict) else False,
                    language=self.language,
                )
                for p in item.get("premises", [])
            ]
            if not premises:
                continue
            conclusion = Conclusion(
                text=item.get("conclusion", {}).get("text", ""),
                language=self.language,
            )
            arguments.append(Argument(
                premises=premises, conclusion=conclusion,
                inference_type=InferenceType(
                    item.get("inference_type", "deductive")
                ),
                source_text=text,
                confidence=item.get("confidence", 0.7),
            ))
        return arguments

    def _split_sentences(self, text: str) -> list[str]:
        if self.language in ("ja", "zh"):
            return [s.strip() for s in re.split(r'[。！？]', text) if s.strip()]
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
