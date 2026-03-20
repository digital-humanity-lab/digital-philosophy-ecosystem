"""Philosophical school/tradition classifier."""

from __future__ import annotations

from dataclasses import dataclass, field

SCHOOL_TAXONOMY: dict[str, list[str]] = {
    "Western": [
        "Presocratic", "Platonic", "Aristotelian", "Stoic", "Epicurean",
        "Neoplatonic", "Scholastic", "Rationalist", "Empiricist",
        "Kantian", "German Idealism", "Phenomenology", "Existentialism",
        "Analytic", "Pragmatism", "Critical Theory", "Poststructuralism",
        "Process Philosophy",
    ],
    "East Asian": [
        "Confucian", "Daoist", "Legalist", "Mohist", "Chan/Zen Buddhist",
        "Neo-Confucian", "Kyoto School", "New Confucianism",
    ],
    "South Asian": [
        "Nyaya", "Vaisheshika", "Samkhya", "Yoga", "Mimamsa", "Vedanta",
        "Buddhist (Madhyamaka)", "Buddhist (Yogacara)", "Jain",
    ],
    "Islamic": [
        "Kalam", "Falsafa", "Sufi Philosophy", "Illuminationist",
    ],
}


@dataclass
class SchoolPrediction:
    school: str
    tradition: str
    confidence: float
    top_k: list[tuple[str, float]] = field(default_factory=list)


class SchoolClassifier:
    """Classify philosophical text by school/tradition.

    Uses zero-shot NLI classification as the default approach.
    """

    def __init__(
        self,
        nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        device: str | None = None,
    ):
        self._nli_model = nli_model
        self._device = device
        self._schools = self._flatten_taxonomy()
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
                self._pipeline = pipeline(
                    "zero-shot-classification",
                    model=self._nli_model,
                    device=self._device,
                )
            except ImportError:
                raise ImportError(
                    "SchoolClassifier requires transformers. "
                    "Install with: pip install philtext[nlp]"
                )
        return self._pipeline

    def classify(self, text: str, top_k: int = 5) -> SchoolPrediction:
        pipe = self._get_pipeline()
        result = pipe(
            text[:1024],
            candidate_labels=self._schools,
            multi_label=False,
        )
        top_school = result["labels"][0]
        tradition = self._find_tradition(top_school)
        return SchoolPrediction(
            school=top_school,
            tradition=tradition,
            confidence=result["scores"][0],
            top_k=list(zip(result["labels"][:top_k], result["scores"][:top_k])),
        )

    @staticmethod
    def _flatten_taxonomy() -> list[str]:
        return [s for schools in SCHOOL_TAXONOMY.values() for s in schools]

    @staticmethod
    def _find_tradition(school: str) -> str:
        for tradition, schools in SCHOOL_TAXONOMY.items():
            if school in schools:
                return tradition
        return "Unknown"
