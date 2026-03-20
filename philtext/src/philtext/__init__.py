"""philtext - Philosophical text analysis toolkit."""

from philtext.argument.schemas import Argument, Conclusion, InferenceType, Premise, Proposition
from philtext.argument.extractor import ArgumentExtractor
from philtext.concept.ontology import ConceptMention, ConceptNode, PhilOntology
from philtext.concept.extractor import ConceptExtractor
from philtext.classify.school import SchoolClassifier, SchoolPrediction, SCHOOL_TAXONOMY
from philtext.influence.detector import InfluenceDetector, InfluenceLink
from philtext.corpus.builder import CorpusBuilder, PhilDocument
from philtext.corpus.aligner import TextAligner, AlignedSegment
from philtext.hermeneutic.interpretation import Interpretation, InterpretationTracker
from philtext.hermeneutic.commentary import Commentary, CommentaryLinker
from philtext.hermeneutic.evolution import TermEvolution, TermUsage
from philtext.bridge.mapper import PracticalMapper, PracticalMapping
from philtext.bridge.translator import ConceptTranslator, TranslatedConcept

__version__ = "0.1.0"

__all__ = [
    "Argument", "Conclusion", "InferenceType", "Premise", "Proposition",
    "ArgumentExtractor",
    "ConceptMention", "ConceptNode", "PhilOntology", "ConceptExtractor",
    "SchoolClassifier", "SchoolPrediction", "SCHOOL_TAXONOMY",
    "InfluenceDetector", "InfluenceLink",
    "CorpusBuilder", "PhilDocument", "TextAligner", "AlignedSegment",
    "Interpretation", "InterpretationTracker",
    "Commentary", "CommentaryLinker",
    "TermEvolution", "TermUsage",
    "PracticalMapper", "PracticalMapping",
    "ConceptTranslator", "TranslatedConcept",
]
