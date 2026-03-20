"""philgraph - Philosophical knowledge graph builder and query engine."""

from philgraph.graph import PhilGraph
from philgraph.schema import (
    Concept, Thinker, Text, Tradition, Argument, Era, Institution, Language,
    Edge, EdgeType, EdgeProperties, ConsensusLevel, NodeBase,
)
from philgraph.viz.visualize import PhilGraphViz

__version__ = "0.1.0"

__all__ = [
    "PhilGraph",
    "Concept", "Thinker", "Text", "Tradition", "Argument",
    "Era", "Institution", "Language", "NodeBase",
    "Edge", "EdgeType", "EdgeProperties", "ConsensusLevel",
    "PhilGraphViz",
]
