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

        # Split into sentences first, then find indicator-bearing sentences
        sentences = self._split_sentences(text)
        premise_sents: list[str] = []
        conclusion_sent: str | None = None

        for sent in sentences:
            has_conclusion = any(
                re.search(p, sent, re.IGNORECASE) for p in conclusion_pats
            )
            has_premise = any(
                re.search(p, sent, re.IGNORECASE) for p in premise_pats
            )

            if has_conclusion and has_premise:
                # Same sentence has both: split on conclusion indicator
                for p in conclusion_pats:
                    m = re.search(p, sent, re.IGNORECASE)
                    if m:
                        before = sent[:m.start()].strip().strip(",;.")
                        after = sent[m.end():].strip().strip(",;.")
                        if len(before) > 15:
                            premise_sents.append(before)
                        if len(after) > 15:
                            conclusion_sent = after
                        break
            elif has_conclusion:
                # Strip indicator, keep the substantive content
                cleaned = sent
                for p in conclusion_pats:
                    cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE)
                cleaned = cleaned.strip(" ,.;:")
                if len(cleaned) > 15:
                    conclusion_sent = cleaned
            elif has_premise:
                cleaned = sent
                for p in premise_pats:
                    cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE)
                cleaned = cleaned.strip(" ,.;:")
                if len(cleaned) > 15:
                    premise_sents.append(cleaned)

        if premise_sents and conclusion_sent:
            premises = [
                Premise(text=s.strip(), language=self.language)
                for s in premise_sents
            ]
            conclusion = Conclusion(
                text=conclusion_sent.strip(), language=self.language
            )
            return [Argument(
                premises=premises, conclusion=conclusion,
                inference_type=InferenceType.DEDUCTIVE,
                source_text=text, confidence=0.5,
            )]
        return []

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
