# Digital Philosophy Ecosystem - Validation Plan

## Overview

This document specifies concrete validation strategies for all four packages in the ecosystem. Each section identifies ground truth sources, benchmark tasks with specific test cases, evaluation metrics, and a human evaluation protocol.

---

## 1. philcore - Foundation Ontology & Data Models

### 1.1 Ground Truth Sources

| Source | What it validates | Access |
|--------|-------------------|--------|
| **InPhO (Indiana Philosophy Ontology)** | Concept hierarchy, subsumption, broader/narrower | OWL export at `https://inpho.cogs.indiana.edu/` |
| **PhilPapers taxonomy** | 5,000+ categories of philosophical topics | JSON API, also downloadable |
| **Wikidata philosophy subgraph** | Thinker biographical data, tradition membership, concept labels | SPARQL endpoint |
| **CIDOC-CRM standard** | Ontological alignment correctness | Published specification |
| **SEP / IEP articles** | Concept definitions, tradition boundaries | Publicly available |

### 1.2 Benchmark Tasks

**Task C1: Concept roundtrip serialization**
- Serialize 100 `Concept` objects (covering all traditions in `SCHOOL_TAXONOMY`) to JSON-LD, then to RDF/Turtle, then back to `Concept`. Assert lossless roundtrip for all fields.
- Metric: 100% field preservation rate.

**Task C2: Ontology alignment with InPhO**
- Export philcore concept hierarchy as OWL. Load InPhO OWL. For overlapping concepts (matched by label+Wikidata QID), check that `broader/narrower` edges agree.
- Metric: Agreement rate on `skos:broader` relations for shared concepts.
- Target: >=85% agreement on established hierarchies.

**Task C3: Non-classical logic consistency**
- For each `LogicFamily` (catuskoti, basho, paraconsistent, dialectical), validate that the formal semantics produce correct truth tables.
- Specific cases:
  - Catuskoti: `(T, F, Both, Neither)` for "all dharmas are empty" should allow `Both` as valid.
  - Basho: Three-level nesting `有の場所 ⊂ 相対無の場所 ⊂ 絶対無の場所` must be a strict total order.
  - Dialectical: `thesis ⊕ antithesis → synthesis` and the synthesis must carry `Aufhebung` metadata.

**Task C4: Multilingual label integrity**
- For 50 concepts with known labels in 3+ languages, verify `label_in(lang)` returns the correct string.
- Test cases include:
  - `仁` (zh) / `ren` (en transliteration) / `じん` (ja)
  - `Dasein` (de) / `現存在` (ja) / `being-there` (en)
  - `śūnyatā` (sa) / `空` (zh) / `空` (ja) / `emptiness` (en)

### 1.3 Evaluation Metrics

| Metric | Target | Method |
|--------|--------|--------|
| Roundtrip fidelity | 100% | Automated unit tests |
| InPhO hierarchy agreement | >=85% | Script comparing OWL exports |
| Logic truth-table correctness | 100% | Manually verified test cases |
| Wikidata QID linking accuracy | >=90% | Sample check against Wikidata |

---

## 2. philmap - Cross-Cultural Concept Alignment

### 2.1 Ground Truth Sources

The following published works in comparative philosophy provide scholarly consensus on concept mappings:

| Source | Mappings provided |
|--------|-------------------|
| **Ames & Rosemont, *The Analects of Confucius* (1998)** | 仁↔benevolence/humaneness, 禮↔ritual propriety, 道↔Way |
| **Heidegger & Japanese philosophy**: Parkes, *Heidegger and Asian Thought* (1987) | 無↔Nichts, 場所↔Lichtung, 間↔Zwischen |
| **Metz, *Ubuntu as a Moral Theory* (2007)** | Ubuntu↔communitarian ethics, Ubuntu↔I-Thou |
| **Rambachan, *A Hindu Theology of Liberation* (2015)** | Brahman↔God, mokṣa↔salvation, ahiṃsā↔non-violence |
| **Priest, *Beyond the Limits of Thought* (2002)** | śūnyatā↔Nichts↔absolute nothingness comparisons |
| **Watsuji Tetsurō, *Fūdo* / *Ethics***: Carter & McCarthy, *The Japanese Arts and Self-Cultivation* | 間柄↔betweenness↔I-Thou |
| **Matilal, *Perception* (1986)** | Nyāya pramāṇa↔epistemic justification |
| **Izutsu, *Sufism and Taoism* (1983)** | fanāʾ↔無為↔mystical annihilation |
| **Nishitani, *Religion and Nothingness* (1982)** | 絶対無↔śūnyatā↔Nichts (Kyoto School synthesis) |
| **Wiredu, *Cultural Universals and Particulars* (1996)** | Akan truth/knowledge↔Western epistemology |

### 2.2 Benchmark Test Cases: Cross-Tradition Concept Pairs

These 15 mappings are well-attested in comparative philosophy literature. Each entry includes the scholarly basis and expected alignment score range.

| # | Concept A | Concept B | Concept C | Scholarly basis | Expected similarity |
|---|-----------|-----------|-----------|-----------------|---------------------|
| 1 | 間柄 (aidagara, Watsuji) | I-Thou (Buber) | Ubuntu (Metz) | Carter 2001; Metz 2007; relational selfhood | High (>0.7) |
| 2 | 絶対無 (zettai mu, Nishida) | Nichts (Heidegger) | śūnyatā (Nagarjuna) | Nishitani 1982; Parkes 1987 | High (>0.7) |
| 3 | 仁 ren (Confucius) | karuṇā (Buddhism) | agape (Christianity) | Yearley 1990; compassion as virtue | Medium-High (0.6-0.8) |
| 4 | 道 dao (Laozi) | logos (Heraclitus) | ṛta (Vedic) | Mou 2001; cosmic order | Medium (0.5-0.7) |
| 5 | 無為 wuwei (Daoism) | Gelassenheit (Heidegger) | fanāʾ (Sufism) | Izutsu 1983; Parkes 1987 | Medium-High (0.6-0.8) |
| 6 | 理 li (Zhu Xi) | eidos/Form (Plato) | — | Needham; Neo-Confucian/Greek comparison | Medium (0.5-0.7) |
| 7 | dharma (Buddhism) | natural law (Aquinas) | ṛta (Vedic) | Olivelle 2009 | Medium (0.5-0.7) |
| 8 | pramāṇa (Nyāya) | epistemic justification (analytic) | — | Matilal 1986; Ganeri 2001 | High (>0.7) |
| 9 | mokṣa (Hinduism) | nirvāṇa (Buddhism) | salvation (Christianity) | Rambachan 2015 | Medium-High (0.6-0.8) |
| 10 | ahiṃsā (Jainism/Gandhi) | non-violence (Tolstoy/King) | — | Direct influence, well-documented | High (>0.7) |
| 11 | 場所 basho (Nishida) | Lichtung (Heidegger) | chōra (Plato) | Davis 2019; topology of being | Medium (0.5-0.7) |
| 12 | 気 qi (Chinese) | pneuma (Stoic) | prāṇa (Vedic) | Needham; vital force comparison | Medium-High (0.6-0.8) |
| 13 | 禮 li/ritual (Confucius) | Sittlichkeit (Hegel) | — | Rosemont 2015; ethical life as practice | Medium (0.5-0.7) |
| 14 | pratītyasamutpāda (Buddhism) | process (Whitehead) | — | Odin 1982; relational becoming | Medium-High (0.6-0.8) |
| 15 | māyā (Advaita Vedanta) | Schein/appearance (Kant) | — | Deutsch 1969; illusion vs. appearance | Medium (0.5-0.7) |

**Negative test cases** (should NOT be aligned):

| Concept A | Concept B | Why dissimilar |
|-----------|-----------|----------------|
| 仁 ren (Confucius) | will to power (Nietzsche) | Opposite ethical valence |
| śūnyatā (Nagarjuna) | substance (Aristotle) | Ontological opposites |
| karma (Hinduism) | predestination (Calvin) | Superficial similarity, deep divergence (agency vs. decree) |
| 無為 wuwei | categorical imperative (Kant) | Effortless non-action vs. duty-based maxim |
| Ubuntu | Hobbesian state of nature | Communal vs. atomistic anthropology |

### 2.3 Benchmark Tasks

**Task M1: Known-pair retrieval (precision@k)**
- Given concept A, retrieve its known analogues from a pool of 200+ concepts across traditions.
- Use the 15 positive pairs above as queries.
- Metric: `precision@5`, `precision@10`, `MAP@10`.
- Target: MAP@10 >= 0.5 (baseline; improve with fine-tuning).

**Task M2: Concept discrimination**
- Given concept A and two candidates (one scholarly-attested analogue, one negative), the system must rank the correct analogue higher.
- Use all 15 positive and 5 negative pairs.
- Metric: Pairwise accuracy (% correct orderings).
- Target: >=80%.

**Task M3: Facet contribution analysis**
- For each of the 15 pairs, report `definition`, `usage`, `relational` facet scores.
- Verify that the facet with the highest score matches the scholarly reason for comparison (e.g., 間柄↔I-Thou should score highest on `relational` or `usage`, not just `definition`).
- Metric: Facet-reason concordance rate.

**Task M4: Tradition bridge completeness**
- Run `tradition_bridge(Kyoto_School, German_Phenomenology)` and verify that at least these known pairs appear in the top 20 results:
  - 絶対無↔Nichts
  - 場所↔Lichtung
  - 自覚↔Selbstbewusstsein
  - 行為的直観↔Anschauung
- Metric: Recall of known pairs in top-k results.

### 2.4 Evaluation Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| MAP@10 | Mean average precision at 10 for known-pair retrieval | >=0.50 |
| Pairwise accuracy | % of (positive, negative) pairs correctly ordered | >=80% |
| Facet concordance | % of pairs where top facet matches scholarly reason | >=60% |
| Bridge recall@20 | % of known pairs retrieved in tradition bridge | >=70% |

---

## 3. philtext - Philosophical Text Analysis

### 3.1 Ground Truth Sources

#### 3.1.1 Argument Extraction

| Source | Description | Size |
|--------|-------------|------|
| **AIFdb (Argument Interchange Format)** | Largest argument corpus; includes Araucaria annotations | ~14,000 argument maps |
| **US-UK-2016 corpus** (Reed et al.) | Annotated arguments from essays | ~1,000 annotated arguments |
| **Walton's argument schemes** | Taxonomy of 60+ argument schemes with examples | Reference classification |
| **Manual annotation set** (to be created) | 100 passages from canonical philosophical texts annotated by 2+ philosophers | Focus on non-Western texts |

Specific texts for annotation:
- Plato, *Meno* 80d-86c (paradox of inquiry; known argument structure)
- Aquinas, *Summa Theologica* I, Q2, A3 (Five Ways; well-analyzed)
- Nagarjuna, *Mūlamadhyamakakārikā* Ch. 1 (examination of causal conditions)
- 朱熹, *近思録* selections (Neo-Confucian reasoning)
- Kant, *Critique of Pure Reason* B-Deduction (transcendental argument)

#### 3.1.2 School Classification

| Source | Description | Labels |
|--------|-------------|--------|
| **PhilPapers categories** | Papers self-classified by area | ~5,000 categories (need mapping to SCHOOL_TAXONOMY) |
| **Project Gutenberg philosophy texts** | Full-text books with known authorship | Author→school mapping |
| **SEP article introductions** | ~1,800 articles; each associated with philosophical areas | Extract first paragraph + label |
| **JSTOR Data for Research** | Philosophy journal articles with metadata | Journal→tradition mapping (e.g., *Philosophy East and West*) |

Concrete labeled dataset construction:
1. Scrape first 500 words of 200 SEP articles covering all 40 schools in `SCHOOL_TAXONOMY`.
2. Map PhilPapers taxonomy to `SCHOOL_TAXONOMY` using a manually curated mapping table.
3. Use abstracts from *Philosophy East and West*, *Asian Philosophy*, *Journal of Indian Philosophy* for non-Western coverage.

#### 3.1.3 Influence Detection

| Source | Established influence | Evidence type |
|--------|----------------------|---------------|
| **Heidegger → Sartre** | *Being and Time* → *Being and Nothingness* | Direct citation, conceptual |
| **Husserl → Heidegger** | *Logical Investigations* → *Being and Time* | Student-teacher, conceptual |
| **Nishida → Tanabe → Nishitani** | Kyoto School lineage | Direct lineage, documented |
| **Schopenhauer → Nietzsche** | *WWR* → *Birth of Tragedy* | Acknowledged influence |
| **Nagarjuna → Chandrakirti** | MMK → Prasannapada | Commentary tradition |
| **Zhu Xi → Wang Yangming** | Critique and reinterpretation | Documented disagreement-influence |
| **Wittgenstein (early → late)** | *Tractatus* → *PI* | Self-influence, evolution |
| **Confucius → Mencius → Xunzi** | Analects → Mencius → Xunzi | Direct lineage |
| **Plato → Aristotle → Alexander of Aphrodisias** | Academy lineage | Documented |
| **Al-Farabi → Ibn Sina → Ibn Rushd** | Islamic Peripatetic tradition | Documented lineage |
| **Gandhi ← Tolstoy ← Thoreau** | Non-violence tradition | Documented direct influence |

Negative cases (no influence despite superficial similarity):
- Democritus and Vaisheshika atomism (independent development)
- Pyrrhonian skepticism and Madhyamaka (no historical contact, debated)

### 3.2 Benchmark Tasks

**Task T1: Argument extraction accuracy**
- Input: 50 pre-annotated philosophical passages (10 per language: en, de, ja, zh, la).
- Output: Extracted `Argument` objects.
- Metrics: Premise recall, conclusion recall, structural F1.
- Evaluation: Exact match is too strict; use ROUGE-L overlap between extracted premise text and gold premise text.
- Target F1: >=0.50 for rule-based, >=0.70 for hybrid.

Specific test case (English, rule-based):
```
Input: "Since all men are mortal, and since Socrates is a man,
        therefore Socrates is mortal."
Expected output:
  Premises: ["all men are mortal", "Socrates is a man"]
  Conclusion: "Socrates is mortal"
  InferenceType: DEDUCTIVE
```

Specific test case (Japanese, rule-based):
```
Input: "すべての人間は死すべきものであるから、そしてソクラテスは人間であるから、
        したがってソクラテスは死すべきものである。"
Expected output:
  Premises: ["すべての人間は死すべきものである", "ソクラテスは人間である"]
  Conclusion: "ソクラテスは死すべきものである"
```

Specific test case (non-trivial, from Aquinas):
```
Input: "The fifth way is taken from the governance of the world.
        We see that things which lack intelligence, such as natural bodies,
        act for an end, and this is evident from their acting always,
        or nearly always, in the same way, so as to obtain the best result.
        Hence it is plain that not fortuitously, but designedly,
        do they achieve their end."
Expected output:
  Premises: ["things which lack intelligence act for an end",
             "this is evident from their acting always in the same way"]
  Conclusion: "not fortuitously, but designedly, do they achieve their end"
  InferenceType: ABDUCTIVE (teleological argument)
```

**Task T2: School classification accuracy**
- Input: 200 text passages (5 per school for the 40 schools in `SCHOOL_TAXONOMY`).
- Output: `SchoolPrediction` with `school` and `tradition`.
- Metrics: Top-1 accuracy, Top-3 accuracy, macro-F1.
- Target: Top-1 accuracy >=50%, Top-3 accuracy >=75% (zero-shot NLI is inherently limited).
- Note: Confusing pairs to track specifically:
  - Phenomenology vs. Existentialism
  - Nyaya vs. Vaisheshika
  - Confucian vs. Neo-Confucian
  - Chan/Zen vs. Buddhist (Madhyamaka)
  - Stoic vs. Epicurean

Specific test cases:
```
Input: "Sein und Zeit demonstrates that the question of the meaning of
        Being has been forgotten by the philosophical tradition. Dasein,
        as the being that understands Being, must be interrogated first."
Expected: school="Phenomenology" or "Existentialism", tradition="Western"

Input: "仁者は人を愛す。克己復礼を仁と為す。"
Expected: school="Confucian", tradition="East Asian"

Input: "All compounded things are impermanent. When one sees this
        with wisdom, one turns away from suffering."
Expected: school in ["Buddhist (Madhyamaka)", "Buddhist (Yogacara)", "Chan/Zen Buddhist"]
          tradition="South Asian" or "East Asian"
```

**Task T3: Influence detection precision/recall**
- Input: 20 text pairs (11 positive from the table above, 9 negatives including both "no influence" and "reverse direction").
- Output: `InfluenceLink` objects.
- Metrics: Precision, recall, direction accuracy.
- Target: Precision >=0.7, recall >=0.6, direction accuracy >=0.8.

Specific test case:
```
Source: Heidegger, Being and Time, Sec. 9 (Dasein analysis)
Target: Sartre, Being and Nothingness, Introduction (being-for-itself)
Expected: InfluenceLink with influence_type="conceptual",
          direction: Heidegger → Sartre (direction_confidence > 0.8)
```

**Task T4: Multilingual argument indicator coverage**
- For each language in `ARGUMENT_INDICATORS` (en, ja, de, zh, la, grc), run the extractor on 10 passages known to contain arguments.
- Metric: % of passages where at least one argument is extracted.
- Target: >=70% per language.

### 3.3 Evaluation Metrics Summary

| Task | Metric | Baseline target | Stretch target |
|------|--------|----------------|----------------|
| Argument extraction (rule) | Structural F1 | 0.50 | 0.65 |
| Argument extraction (hybrid) | Structural F1 | 0.70 | 0.85 |
| School classification | Top-1 accuracy | 50% | 65% |
| School classification | Top-3 accuracy | 75% | 85% |
| Influence detection | Precision | 0.70 | 0.85 |
| Influence detection | Recall | 0.60 | 0.75 |
| Influence direction | Accuracy | 0.80 | 0.90 |

---

## 4. philgraph - Knowledge Graph

### 4.1 Ground Truth Sources

| Source | What it validates | Specific checks |
|--------|-------------------|-----------------|
| **Wikidata SPARQL** | Thinker biographical data, influence edges | Query `P737` (influenced by) for all philosophers |
| **InPhO** | Concept-concept relations | `related-to`, `is-a` edges |
| **PhilPapers** | Citation relations between texts | Co-citation, bibliographic links |
| **SEP bibliographies** | Canonical works per thinker | Cross-check `AUTHORED_BY` edges |
| **MacTutor-style timelines** | Contemporary-with relations | Date-based validation |

### 4.2 Benchmark Tasks

**Task G1: Wikidata influence coverage**
- Extract all `P737` (influenced by) relations from Wikidata for the top 200 philosophers.
- Check what percentage appear as `INFLUENCES` edges in philgraph.
- Metric: Coverage = |philgraph ∩ wikidata| / |wikidata|
- Target: >=70% for philosophers with Wikidata entries.

Expected results for spot-checks:
```
Plato → Aristotle (P737)         → INFLUENCES edge must exist
Husserl → Heidegger (P737)      → INFLUENCES edge must exist
Confucius → Mencius (P737)      → INFLUENCES edge must exist
Al-Kindi → Al-Farabi (P737)     → INFLUENCES edge must exist
```

**Task G2: Graph consistency checks**
- **Temporal consistency**: If A influences B, A's active period should not be entirely after B's.
- **Tradition membership**: If A belongs_to_tradition T and authored text X, then X should have tradition context consistent with T.
- **Edge type constraints**: Validate all edges against `EDGE_CONSTRAINTS` in schema.py.
- Metric: % of edges passing consistency checks.
- Target: >=95%.

**Task G3: Path discovery validation**
- Known intellectual lineages must be discoverable via `find_path`:
  - Socrates → Plato → Aristotle → Alexander of Aphrodisias
  - Confucius → Mencius → Zhu Xi → Wang Yangming
  - Husserl → Heidegger → Sartre → de Beauvoir
  - Nagarjuna → Chandrakirti → Tsongkhapa → Dalai Lama lineage
  - Al-Farabi → Ibn Sina → Ibn Rushd → Aquinas (Islamic-Christian transmission)
- Metric: % of known paths fully connected in graph.
- Target: 100% for established lineages after ingestion.

**Task G4: Entity resolution accuracy**
- Ingest the same 50 philosophers from both Wikidata and InPhO.
- Check that the resolver correctly merges them (same entity) rather than creating duplicates.
- Metric: Precision and recall of entity merges.
- Target: Precision >=0.90, recall >=0.85.

**Task G5: Cross-tradition concept cluster validation**
- Run `concept_cluster("emptiness", depth=2)` and verify it returns concepts from multiple traditions:
  - Must include: śūnyatā, 空 (kū), 無 (mu), Nichts
  - Should include: 絶対無, anicca, void
- Run `concept_cluster("virtue", depth=2)`:
  - Must include: aretē, 仁 ren, 德 de, Ubuntu, dharma
- Metric: Multi-tradition coverage (number of distinct traditions in cluster).
- Target: >=3 traditions per cluster for core philosophical concepts.

### 4.3 Evaluation Metrics Summary

| Metric | Definition | Target |
|--------|-----------|--------|
| Wikidata influence coverage | % of WD P737 edges present in graph | >=70% |
| Temporal consistency | % of influence edges respecting chronology | >=95% |
| Edge constraint compliance | % of edges passing EDGE_CONSTRAINTS | 100% |
| Path connectivity | % of known lineages fully connected | 100% |
| Entity resolution precision | Correct merges / total merges | >=90% |
| Entity resolution recall | Correct merges / expected merges | >=85% |
| Cluster tradition coverage | Avg distinct traditions per concept cluster | >=3 |

---

## 5. Human Evaluation Protocol

### 5.1 Expert Recruitment

**Target evaluators** (minimum 5, ideally 8-10):

| Role | Expertise needed | Where to recruit |
|------|-----------------|------------------|
| 2 comparative philosophy scholars | East-West concept mapping | *Philosophy East and West* editorial board; 比較思想学会 |
| 1 Japanese philosophy specialist | Kyoto School, Watsuji | 日本哲学会; Nanzan Institute |
| 1 Indian philosophy specialist | Nyaya, Madhyamaka, Vedanta | IABS; Journal of Indian Philosophy |
| 1 Chinese philosophy specialist | Confucian, Daoist, Neo-Confucian | Society for Asian and Comparative Philosophy |
| 1 Islamic philosophy specialist | Falsafa, Kalam | BRAIS; Journal of Islamic Philosophy |
| 1 African philosophy specialist | Ubuntu, Akan | Philosophia Africana network |
| 1 analytic/history of Western philosophy | Broad Western coverage | Local philosophy department |

### 5.2 Evaluation Tasks for Experts

#### Task H1: Concept Mapping Judgment (philmap validation)

Experts review 50 system-generated concept mappings and rate each on:
1. **Validity** (1-5): Is this a legitimate philosophical comparison?
2. **Similarity strength** (1-5): How close are these concepts?
3. **Commensurability notes**: Free text on what is preserved/lost.

Format:
```
System output: 間柄 (Watsuji) ↔ I-Thou (Buber), score=0.782
  Q1: Is this a valid comparison? [1=no, 5=clearly yes]
  Q2: How similar are these concepts? [1=very different, 5=near-identical]
  Q3: What is preserved in this mapping? [free text]
  Q4: What is lost? [free text]
  Q5: Scholarly references supporting/opposing this mapping: [citations]
```

Metrics derived:
- Inter-annotator agreement: Krippendorff's alpha >=0.6
- System-expert correlation: Spearman rho between system score and expert similarity rating
- Target correlation: rho >=0.5

#### Task H2: Argument Structure Review (philtext validation)

Experts review 30 system-extracted arguments and judge:
1. Are all premises correctly identified? (precision/recall)
2. Is the conclusion correctly identified? (binary)
3. Is the inference type correct? (categorical)
4. Are any implicit premises missing? (recall gap)

#### Task H3: Knowledge Graph Audit (philgraph validation)

Experts review a 500-node subgraph covering their specialty area:
1. Identify missing edges (false negatives).
2. Identify incorrect edges (false positives).
3. Rate `consensus_level` labels on 50 randomly sampled edges.
4. Flag any anachronisms or tradition misattributions.

#### Task H4: School Classification Review (philtext validation)

Experts are shown 50 passages with system-predicted school labels:
1. Is the top-1 prediction correct? (binary)
2. If not, what is the correct school? (provides ground truth)
3. Is the correct school in the top-3? (binary)

### 5.3 Annotation Guidelines

To ensure consistency across evaluators:

1. **Calibration session**: All evaluators annotate the same 10 items. Discuss disagreements. Revise guidelines.
2. **Double annotation**: Each item annotated by >=2 experts. Disagreements resolved by discussion or third expert.
3. **Cultural sensitivity**: Evaluators should not rate mappings outside their area of expertise. The Chinese philosophy specialist does not rate Nyaya concepts.
4. **Anti-bias protocol**: System scores are hidden from evaluators during Tasks H1 and H4 to prevent anchoring.

### 5.4 Compensation and Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| Recruitment | 2 weeks | Contact scholars, IRB if needed |
| Calibration | 1 session (2h) | Shared annotation of 10 items |
| Main evaluation | 2 weeks | Asynchronous, web-based annotation tool |
| Adjudication | 1 week | Resolve disagreements |
| Analysis | 1 week | Compute metrics, write report |

Compensation: Honorarium per evaluator (standard rate for expert annotation in DH projects).

---

## 6. Automated Test Suite Structure

```
tests/
├── philcore/
│   ├── test_roundtrip_serialization.py     # Task C1
│   ├── test_inpho_alignment.py             # Task C2
│   ├── test_nonclassical_logic.py          # Task C3
│   └── test_multilingual_labels.py         # Task C4
├── philmap/
│   ├── test_known_pair_retrieval.py        # Task M1
│   ├── test_concept_discrimination.py      # Task M2
│   ├── test_facet_analysis.py              # Task M3
│   ├── test_tradition_bridge.py            # Task M4
│   └── conftest.py                         # Fixture: 200 concept objects
├── philtext/
│   ├── test_argument_extraction.py         # Task T1
│   ├── test_school_classification.py       # Task T2
│   ├── test_influence_detection.py         # Task T3
│   ├── test_indicator_coverage.py          # Task T4
│   └── fixtures/
│       ├── annotated_arguments_en.json     # 10 English passages
│       ├── annotated_arguments_ja.json     # 10 Japanese passages
│       ├── annotated_arguments_de.json     # 10 German passages
│       ├── school_labeled_passages.json    # 200 passages, 5 per school
│       └── influence_pairs.json            # 20 text pairs
├── philgraph/
│   ├── test_wikidata_coverage.py           # Task G1
│   ├── test_consistency.py                 # Task G2
│   ├── test_path_discovery.py              # Task G3
│   ├── test_entity_resolution.py           # Task G4
│   └── test_concept_clusters.py            # Task G5
└── integration/
    ├── test_core_to_map.py                 # philcore concept → philmap alignment
    ├── test_text_to_graph.py               # philtext extraction → philgraph ingestion
    └── test_full_pipeline.py               # Text → extraction → mapping → graph
```

---

## 7. Validation Data Collection Priority

Since many ground truth datasets must be constructed, here is the priority order:

| Priority | Dataset | Effort | Blocks |
|----------|---------|--------|--------|
| P0 | 15 cross-tradition concept pairs (Section 2.2) with full descriptions | 1 week manual curation | philmap benchmarks |
| P0 | SCHOOL_TAXONOMY → labeled passages (200 passages) | 2 weeks (SEP scraping + manual) | philtext T2 |
| P1 | 50 annotated argument passages (10 per language) | 3 weeks (requires annotators) | philtext T1 |
| P1 | Wikidata P737 influence graph dump for top 200 philosophers | 1 day (SPARQL query) | philgraph G1 |
| P2 | 20 influence text pairs with full passages | 2 weeks (text sourcing) | philtext T3 |
| P2 | InPhO OWL export + mapping table to philcore | 1 week | philcore C2 |
| P3 | Full 200-concept pool for philmap retrieval benchmark | 4 weeks | philmap M1 at scale |
| P3 | Human evaluation recruitment + annotation | 6 weeks | All H-tasks |

---

## 8. Known Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Zero-shot NLI classifier performs poorly on non-Western schools | School classification F1 drops for South Asian/East Asian | Fine-tune on curated dataset; use few-shot prompting as backup |
| Embedding model has weak coverage of classical languages (Sanskrit, Classical Chinese, Ancient Greek) | Concept alignment fails for historical concepts | Use definition-based embedding (translate definitions to English first, then embed) as fallback |
| Comparative philosophy literature is itself Western-centric | Ground truth biased toward Western-accessible comparisons | Recruit non-Western specialists; include intra-tradition benchmarks (e.g., Confucian vs. Daoist) |
| Inter-annotator agreement low on concept similarity judgments | Human evaluation results unreliable | Use calibration session; limit rating to expert's own tradition |
| Small test sets produce noisy metrics | Confidence intervals too wide | Report 95% CI via bootstrap; plan for larger test sets in v2 |
