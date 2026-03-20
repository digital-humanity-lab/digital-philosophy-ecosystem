"""Map philosophical concepts to practical domains."""

from __future__ import annotations

from dataclasses import dataclass, field

_DOMAIN_MAPPINGS: dict[tuple[str, str], dict] = {
    ("ethics", "policy"): {
        "description": "Ethical frameworks informing policy design",
        "concept_map": {
            "categorical imperative": "universalizability test for regulations",
            "utilitarianism": "cost-benefit analysis, welfare maximization",
            "virtue ethics": "character-based standards for public officials",
            "justice as fairness": "Rawlsian veil of ignorance in policy evaluation",
            "care ethics": "vulnerability-responsive policy design",
        },
    },
    ("epistemology", "ai"): {
        "description": "Epistemological concepts applied to AI systems",
        "concept_map": {
            "justified true belief": "model confidence calibration",
            "epistemic humility": "uncertainty quantification, abstention",
            "coherentism": "ensemble methods, consistency checking",
            "foundationalism": "axiomatic system design, ground truth",
            "social epistemology": "federated learning, collective intelligence",
        },
    },
    ("aesthetics", "design"): {
        "description": "Aesthetic theory applied to design practice",
        "concept_map": {
            "sublime": "awe-inducing UX, scale in data visualization",
            "wabi-sabi": "imperfection-tolerant design, graceful degradation",
            "mono no aware": "designing for ephemerality, time-aware UX",
        },
    },
    ("philosophy of mind", "ai"): {
        "description": "Philosophy of mind concepts in AI",
        "concept_map": {
            "intentionality": "goal-directed agent design",
            "qualia": "sensor grounding, embodied perception",
            "functionalism": "computational theory of mind as AI architecture",
            "extended mind": "human-AI cognitive coupling",
        },
    },
}


@dataclass
class PracticalMapping:
    concept: str
    philosophical_domain: str
    practical_domain: str
    mapping_description: str
    concrete_examples: list[str] = field(default_factory=list)
    confidence: float = 0.0
    sources: list[str] = field(default_factory=list)


class PracticalMapper:
    """Map philosophical concepts to practical domains."""

    def __init__(self, use_llm: bool = False, llm_model: str = "claude-sonnet-4-20250514"):
        self._mappings = dict(_DOMAIN_MAPPINGS)
        self.use_llm = use_llm
        self.llm_model = llm_model

    def map(
        self, concept: str, philosophical_domain: str, practical_domain: str,
    ) -> PracticalMapping:
        key = (philosophical_domain.lower(), practical_domain.lower())
        domain_info = self._mappings.get(key)
        if domain_info:
            concept_lower = concept.lower()
            if concept_lower in domain_info["concept_map"]:
                return PracticalMapping(
                    concept=concept,
                    philosophical_domain=philosophical_domain,
                    practical_domain=practical_domain,
                    mapping_description=domain_info["concept_map"][concept_lower],
                    confidence=0.85,
                )

        if self.use_llm:
            return self._llm_map(concept, philosophical_domain, practical_domain)

        return PracticalMapping(
            concept=concept,
            philosophical_domain=philosophical_domain,
            practical_domain=practical_domain,
            mapping_description="No mapping found in knowledge base.",
            confidence=0.0,
        )

    def _llm_map(
        self, concept: str, phil_domain: str, prac_domain: str
    ) -> PracticalMapping:
        try:
            from litellm import completion
            import json
            prompt = (
                f'Map "{concept}" from {phil_domain} to {prac_domain}. '
                f'Return JSON: {{"mapping_description": "...", '
                f'"concrete_examples": ["..."], "confidence": 0.0-1.0}}'
            )
            response = completion(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            data = json.loads(response.choices[0].message.content)
            return PracticalMapping(
                concept=concept, philosophical_domain=phil_domain,
                practical_domain=prac_domain,
                mapping_description=data.get("mapping_description", ""),
                concrete_examples=data.get("concrete_examples", []),
                confidence=data.get("confidence", 0.5),
            )
        except Exception:
            return PracticalMapping(
                concept=concept, philosophical_domain=phil_domain,
                practical_domain=prac_domain,
                mapping_description="LLM mapping failed.",
                confidence=0.0,
            )

    def available_domains(self) -> list[tuple[str, str]]:
        return list(self._mappings.keys())
