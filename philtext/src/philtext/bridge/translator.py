"""Translate philosophical concepts for non-philosophical audiences."""

from __future__ import annotations

from dataclasses import dataclass, field

from jinja2 import Template


@dataclass
class TranslatedConcept:
    original_term: str
    original_definition: str
    target_audience: str
    translated_term: str
    explanation: str
    analogy: str = ""
    caveats: list[str] = field(default_factory=list)


_AUDIENCE_TEMPLATES = {
    "engineer": Template(
        "**{{ original_term }}** (philosophy) -> "
        "**{{ translated_term }}** (engineering)\n\n"
        "In engineering terms: {{ explanation }}\n"
        "{% if analogy %}\nThink of it like: {{ analogy }}{% endif %}\n"
        "{% if caveats %}\nCaveats:\n"
        "{% for c in caveats %}- {{ c }}\n{% endfor %}{% endif %}"
    ),
    "policymaker": Template(
        "**{{ original_term }}** -> "
        "**{{ translated_term }}** (policy context)\n\n"
        "Policy relevance: {{ explanation }}\n"
        "{% if analogy %}\nAnalogy: {{ analogy }}{% endif %}\n"
        "{% if caveats %}\nImportant qualifications:\n"
        "{% for c in caveats %}- {{ c }}\n{% endfor %}{% endif %}"
    ),
}


class ConceptTranslator:
    """Translate philosophical concepts for target audiences."""

    def __init__(self, use_llm: bool = False, llm_model: str = "claude-sonnet-4-20250514"):
        self.use_llm = use_llm
        self.llm_model = llm_model

    def translate(
        self, term: str, definition: str, target_audience: str,
    ) -> TranslatedConcept:
        if self.use_llm:
            return self._llm_translate(term, definition, target_audience)
        return TranslatedConcept(
            original_term=term, original_definition=definition,
            target_audience=target_audience,
            translated_term=term, explanation=definition,
            caveats=["Automated translation not available; showing original."],
        )

    def _llm_translate(
        self, term: str, definition: str, audience: str
    ) -> TranslatedConcept:
        try:
            from litellm import completion
            import json
            prompt = (
                f"Translate this philosophical concept for a {audience}:\n"
                f"Term: {term}\nDefinition: {definition}\n"
                f'Return JSON: {{"translated_term": "...", '
                f'"explanation": "...", "analogy": "...", "caveats": ["..."]}}'
            )
            response = completion(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            data = json.loads(response.choices[0].message.content)
            return TranslatedConcept(
                original_term=term, original_definition=definition,
                target_audience=audience,
                translated_term=data.get("translated_term", term),
                explanation=data.get("explanation", definition),
                analogy=data.get("analogy", ""),
                caveats=data.get("caveats", []),
            )
        except Exception:
            return self.translate(term, definition, audience)

    def render(self, concept: TranslatedConcept) -> str:
        template = _AUDIENCE_TEMPLATES.get(concept.target_audience)
        if template:
            return template.render(
                original_term=concept.original_term,
                translated_term=concept.translated_term,
                explanation=concept.explanation,
                analogy=concept.analogy,
                caveats=concept.caveats,
            )
        return f"{concept.original_term} -> {concept.translated_term}\n{concept.explanation}"
