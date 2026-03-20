# Digital Philosophy Ecosystem - Architecture Specification

## Vision

哲学分野にデジタル基盤を提供し、これまで分断されていた領域統合をデジタル技術で実現する。
国ごとに文化風習が異なるゆえに分断されてきた哲学的概念を、計算的手法で対応・比較可能にする。

## Current State of the Field (2026年3月調査)

### Existing Resources
| Resource | Coverage | Limitation |
|----------|----------|------------|
| PhilPapers | Largest philosophy bibliography, JSON API | Bibliographic only, no concept structure |
| InPhO | Semi-automated ontology, OWL/REST API | Western-centric |
| Chinese Text Project | 30,000+ classical Chinese texts, Python API | Chinese philosophy only |
| Sanskrit Library | Digitized texts, morphological tools | Indian philosophy only |
| Wikidata | Philosopher/influence SPARQL data | Shallow concept structure |

### Critical Gaps
1. **No philosophy-specific Python packages** (unlike BioPython, AstroPy)
2. **No cross-cultural concept mapping tools**
3. **No computational models of non-Western philosophical concepts**
4. **No philosophy knowledge graph**
5. **No multilingual philosophical corpus**
6. **Philosophy under-represented in digital humanities**

---

## Package Ecosystem

```
philcore   (Foundation Layer - Ontology & Data Models)
  |
  +-- philmap    (Cross-Cultural Concept Alignment)
  +-- philtext   (Philosophical Text Analysis)
  +-- philgraph  (Knowledge Graph Builder & Query Engine)
```

---

## 1. philcore - Foundation Ontology & Data Models

**Purpose**: Provide the shared data model layer for all downstream packages.

### Core Data Models (Pydantic-based)

| Model | Description | Key Features |
|-------|-------------|--------------|
| `Concept` | Philosophical concept (e.g., 間柄, Dasein, Ubuntu) | Multilingual labels, tradition association, formal properties, temporal context |
| `Argument` | Philosophical argument | Premises, conclusions, logical form, catuskoti support |
| `Tradition` | Philosophical tradition/school | Hierarchical, temporal, geographic |
| `Thinker` | Philosopher | Biographical data, conceptual contributions, influence relations |
| `Text` | Philosophical text/work | Passages, authorship, canonical references |
| `ConceptRelation` | Relations between concepts | Equivalence, opposition, subsumption, analogy, with confidence |
| `CrossTraditionMapping` | Higher-level cross-tradition mapping | Preserved/lost/gained features, scholarly references |

### Non-Classical Logic Support (First-Class)
- **Catuskoti** (Buddhist tetralemma): Four-valued evaluation with Nagarjuna's total rejection
- **Basho Logic** (Nishida): Three-level place hierarchy (有の場所 → 相対無 → 絶対無)
- **Paraconsistent Logic**: Belnap's four-valued semantics {T, F, Both, Neither}
- **Dialectical Logic**: Hegelian thesis-antithesis-synthesis with Aufhebung

### Serialization
- OWL/RDF export via rdflib (SKOS + CIDOC-CRM alignment)
- JSON-LD with philosophy-specific context
- Wikidata `owl:sameAs` bridging

### Dependencies
```
pydantic>=2.6, rdflib>=7.0, networkx>=3.2, pyld>=2.0
```

### Package Structure
```
philcore/
├── models/
│   ├── concept.py      # Concept, ConceptLabel, FormalProperty
│   ├── argument.py     # Argument, Premise, Conclusion, CatuskotiPosition
│   ├── tradition.py    # Tradition
│   ├── thinker.py      # Thinker
│   ├── text.py         # Text, TextPassage
│   └── relation.py     # ConceptRelation, CrossTraditionMapping
├── ontology/
│   ├── hierarchy.py    # DAG-based concept hierarchies (NetworkX)
│   ├── mapping.py      # Cross-tradition mapping registry with queries
│   └── logic.py        # Non-classical logic support
├── serialization/
│   ├── rdf.py          # OWL/RDF export
│   ├── jsonld.py       # JSON-LD export
│   ├── cidoc_crm.py    # CIDOC-CRM alignment
│   └── wikidata.py     # Wikidata property mapping
├── registry.py         # In-memory concept/thinker registry
├── namespaces.py       # RDF namespace declarations
└── types.py            # Shared enums (Era, RelationType, MappingConfidence, LogicFamily)
```

### Design Decisions
- **Pydantic over dataclasses**: Validation, JSON serialization, schema generation
- **ID scheme**: `philcore:{entity_type}/{slug_or_uuid}` → URI mapping
- **CIDOC-CRM alignment**: Concept→E89, Thinker→E21, Tradition→E4
- **SKOS alignment**: broader/narrower for concept hierarchies

---

## 2. philmap - Cross-Cultural Concept Alignment

**Purpose**: Enable computational comparison of philosophical concepts across traditions and languages.

### Key Innovation: Faceted Embeddings

Single embedding vectors are too coarse for philosophical concepts. philmap produces a **faceted embedding** - separate vectors for:
1. **Definition**: What the concept means
2. **Usage**: How it's used in arguments and texts
3. **Relational**: Its position in the tradition's ontology

These are combined via configurable weighted pooling.

### Alignment Methods

| Method | Approach | Captures |
|--------|----------|----------|
| `SemanticAlignment` | Multilingual embedding cosine similarity | Meaning overlap |
| `StructuralAlignment` | Graph-based (degree, centrality, motifs) | Positional analogy |
| `ArgumentativeAlignment` | Shared argument roles | Functional analogy |
| `HybridAlignment` | Weighted combination of all three | Multi-dimensional similarity |

### Analysis Tools
- `find_analogues(concept, target_tradition)` - Find closest analogues in another tradition
- `concept_diff(concept_a, concept_b)` - Shared vs. divergent aspects
- `tradition_bridge(trad_a, trad_b)` - All mappable concept pairs
- `concept_genealogy(concept)` - Temporal/cross-tradition evolution

### Concrete Examples
```python
# Map Watsuji's aidagara to Western concepts
mappings = semantic.align_one_to_many(aidagara, [buber_i_thou, levinas_face])
# Result: 間柄 → I-Thou (0.782), 間柄 → le visage (0.683)

# Compare Nishida's basho with Heidegger's Lichtung
diff = pm.concept_diff(nishida_basho, heidegger_lichtung, embedder=embedder)
# Overall similarity: 0.721, strongest on 'definition' (0.758)

# Find analogues of Confucian ren across all traditions
analogues = pm.find_analogues(ren, target_tradition=None, ...)
# ubuntu (0.812), karuṇā (0.774), 間柄 (0.691), I-Thou (0.654)
```

### Dependencies
```
sentence-transformers>=3.0, torch>=2.0, networkx>=3.0, numpy>=1.24, scipy>=1.10, pydantic>=2.0
```

### Default Model
`intfloat/multilingual-e5-large-instruct` (100+ languages, instruction-tuned)

---

## 3. philtext - Philosophical Text Analysis

**Purpose**: Apply NLP to philosophical texts with domain-specific capabilities.

### Core Principle
> 哲学は文献が重要であるがその解釈は属人的である

Rather than hiding interpretive subjectivity, philtext makes it explicit, trackable, and comparable.

### Components

#### 3.1 Argument Extraction
- **Rule-based**: Multilingual discourse markers (en/ja/de/zh/la/grc)
- **LLM-based**: Structured extraction via Claude/GPT
- **Hybrid**: Rules identify candidates, LLM reconstructs structure
- Supports: deductive, inductive, abductive, transcendental, dialectical, reductio

#### 3.2 Concept Extraction & Linking
- Two-stage: candidate generation (dictionary + n-gram) → disambiguation (embedding similarity)
- Links to philcore concept ontology

#### 3.3 School Classification
- Fine-tuned transformer or zero-shot NLI fallback
- 40+ schools across Western, East Asian, South Asian, Islamic traditions

#### 3.4 Influence Detection
- Dense retrieval → cross-encoder reranking → temporal direction assessment
- Types: direct_citation, paraphrase, conceptual, structural

#### 3.5 Multilingual Corpus Tools
- **CorpusBuilder**: PhilPapers, CTP, NDL, Sanskrit Library, Gutenberg
- **TextAligner**: Parallel text alignment via monotonic DP on embeddings
- **Tokenizers**: SudachiPy (ja), jieba (zh), CLTK (la/grc/sa), spaCy (en/de/fr)

#### 3.6 Hermeneutic Analysis
- **InterpretationTracker**: Track/compare scholarly interpretations, find conflicts
- **CommentaryLinker**: Link commentaries to source texts with fine-grained refs
- **TermEvolution**: Track semantic drift across works/time via embedding clustering

#### 3.7 Applied Philosophy Bridge
- **PracticalMapper**: Map philosophical concepts to practical domains (ethics→policy, epistemology→AI, aesthetics→design)
- **ConceptTranslator**: Translate for target audiences (engineer, policymaker) with explicit caveats

### Dependencies
```
spacy>=3.7, sentence-transformers>=3.0, transformers>=4.40, httpx>=0.27, jinja2>=3.1
Optional: sudachipy (ja), jieba (zh), cltk (classical), litellm (LLM backends)
```

---

## 4. philgraph - Knowledge Graph Builder & Query Engine

**Purpose**: Create and manage a unified knowledge graph spanning philosophical traditions worldwide.

### Graph Schema

**Node Types**: Concept, Thinker, Text, Tradition, Argument, Era, Institution, Language

**Edge Types** (15):
influences, opposes, extends, reinterprets, translates_to, analogous_to,
subsumes, uses_in_argument, authored_by, belongs_to_tradition,
contemporary_with, affiliated_with, written_in, part_of_era, cites

**Edge Properties**: confidence (0-1), evidence_sources, consensus_level (established/debated/speculative/novel), temporal_validity

### Data Ingestion Pipeline

| Ingester | Source | Extracts |
|----------|--------|----------|
| `PhilPapersIngester` | PhilPapers JSON API | Texts, Thinkers, categories |
| `InPhOIngester` | InPhO REST API + OWL | Concepts, Thinkers, relations |
| `WikidataIngester` | Wikidata SPARQL | Philosophers, concepts, traditions, works |
| `CTPIngester` | Chinese Text Project API | Classical Chinese texts |
| `ManualIngester` | Curated YAML files | Japanese philosophy, etc. (no API) |

**Entity Resolution**: External ID matching + fuzzy label matching + biographical overlap

### Storage Backends

| Backend | Use Case |
|---------|----------|
| `NetworkXBackend` | Prototyping, small graphs (<100k nodes), in-memory |
| `Neo4jBackend` | Production, persistent, Cypher queries |
| `RDFLibBackend` | Semantic web, SPARQL queries, OWL export |

**Unified query interface** abstracts over all backends.

### Query & Analysis API
```python
graph.find_path(concept_a, concept_b)          # Conceptual paths
graph.influence_network(thinker, depth=2)       # Influence subgraph
graph.tradition_overlap(trad_a, trad_b)         # Shared conceptual ground
graph.concept_cluster(concept, depth=2)         # Community detection
graph.temporal_evolution(concept, start, end)   # Temporal tracking
graph.sparql(query)                             # SPARQL (any backend)
```

### Visualization
- **pyvis**: Interactive network visualization (tradition-colored)
- **D3.js**: JSON export for web applications
- **matplotlib**: Temporal evolution timelines

### Import/Export
RDF/OWL (Turtle), JSON-LD, GraphML (Gephi), Cypher (Neo4j)

### Dependencies
```
networkx>=3.0, pyyaml>=6.0, httpx>=0.27
Optional: neo4j>=5.0, rdflib>=7.0, SPARQLWrapper>=2.0, pyvis>=0.3, rapidfuzz>=3.0
```

---

## Integration with Existing Projects

| Existing Project | Integration Point |
|-----------------|-------------------|
| `openalex-philosophy-corpus-builder` (R) | Country-level seeds → philgraph Thinker/Concept nodes |
| `jp-phil-gaplab` (R) | Gap analysis via `philgraph.tradition_overlap()` and `concept_cluster()` |
| `philo-bridge` (R) | philgraph as authoritative entity registry |

## Implementation Priority

1. **philcore** (foundation - all other packages depend on this)
2. **philgraph** (knowledge graph - enables data-driven exploration)
3. **philmap** (concept mapping - the key differentiator)
4. **philtext** (text analysis - builds on all above)

## Technical Stack Summary

| Layer | Technologies |
|-------|-------------|
| Data Models | Pydantic v2, Python 3.11+ |
| NLP | spaCy, sentence-transformers, SudachiPy, jieba, CLTK |
| Embeddings | multilingual-e5-large-instruct, Qwen3-Embedding |
| Graph | NetworkX, Neo4j, RDFLib |
| Ontology | OWL, SKOS, CIDOC-CRM, JSON-LD |
| LLM (optional) | Claude API, LiteLLM |
| Visualization | pyvis, D3.js, matplotlib |
