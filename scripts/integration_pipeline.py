#!/usr/bin/env python3
"""Full integration pipeline: merge all data sources into unified ecosystem.

Pipeline stages:
1. Build unified philgraph from Wikidata + seed + OpenAlex + NDL
2. Build school classifier training data from SEP + Gutenberg
3. Train school classifier and evaluate
4. Extract concepts from SEP to expand ontology
5. Build multilingual corpus and run cross-lingual analysis
6. Final ecosystem statistics
"""

import sys
import json
import re
import random
import time
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "philcore" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "philgraph" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "philtext" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "philmap" / "src"))

DATA = Path(__file__).parent.parent / "data"
MODEL = Path(__file__).parent.parent / "models"

import numpy as np
from sentence_transformers import SentenceTransformer


# ════════════════════════════════════════════════════════════════
# STAGE 1: Unified Knowledge Graph
# ════════════════════════════════════════════════════════════════

def stage1_unified_graph():
    print("=" * 70)
    print("STAGE 1: 統合ナレッジグラフ構築")
    print("=" * 70)

    from philgraph import PhilGraph, Edge, EdgeType, EdgeProperties, ConsensusLevel
    from philgraph.schema import Thinker, Text, Concept, Tradition

    g = PhilGraph()

    # 1a: Seed data (Kyoto School + East Asian + South Asian)
    for f in ["kyoto_school.yaml", "east_asian_philosophy.yaml", "south_asian_philosophy.yaml"]:
        g.ingest("manual", yaml_paths=[str(DATA / "seed" / f)])
    print(f"  Seed data: {g.summary()['total_nodes']} nodes")

    # 1b: Wikidata philosophers + influences
    with open(DATA / "wikidata" / "philosophers.json") as f:
        philosophers = json.load(f)
    with open(DATA / "wikidata" / "influences.json") as f:
        influences = json.load(f)
    with open(DATA / "wikidata" / "movements.json") as f:
        movements = json.load(f)

    qid_to_uid = {}
    for mv in movements:
        if mv["label_en"]:
            node = Tradition(uid=f"tradition:wd:{mv['qid']}", label=mv["label_en"],
                           labels_i18n={k.split("_")[1]: mv[k] for k in ["label_ja"] if mv.get(k)},
                           external_ids={"wikidata": mv["qid"]}, provenance=["wikidata"])
            g.add_node(node)

    for phil in philosophers:
        if not phil["label_en"]:
            continue
        uid = f"thinker:wd:{phil['qid']}"
        if g.resolve_external_id(phil["qid"], "wikidata"):
            qid_to_uid[phil["qid"]] = g.resolve_external_id(phil["qid"], "wikidata")
            continue
        labels_i18n = {k.split("_")[1]: phil[k] for k in ["label_ja","label_zh","label_de"] if phil.get(k)}
        node = Thinker(uid=uid, label=phil["label_en"], labels_i18n=labels_i18n,
                      birth_year=phil.get("birth_year"), death_year=phil.get("death_year"),
                      external_ids={"wikidata": phil["qid"]}, provenance=["wikidata"])
        g.add_node(node)
        qid_to_uid[phil["qid"]] = uid

    inf_count = 0
    for inf in influences:
        src = qid_to_uid.get(inf["influencer_qid"], f"thinker:wd:{inf['influencer_qid']}")
        tgt = qid_to_uid.get(inf["philosopher_qid"], f"thinker:wd:{inf['philosopher_qid']}")
        if g.get_node(src) and g.get_node(tgt):
            try:
                g.add_edge(Edge(source_uid=src, target_uid=tgt, edge_type=EdgeType.INFLUENCES,
                              properties=EdgeProperties(confidence=0.8, provenance="wikidata")))
                inf_count += 1
            except ValueError:
                pass
    print(f"  Wikidata: +{len(qid_to_uid)} thinkers, +{inf_count} influences")

    # 1c: OpenAlex papers as Text nodes
    with open(DATA / "openalex" / "philosophy_works.json") as f:
        works = json.load(f)
    oa_count = 0
    for w in works[:500]:  # Top 500 by citation
        if not w.get("title"):
            continue
        uid = f"text:oa:{w['id'].split('/')[-1]}"
        node = Text(uid=uid, label=w["title"][:100], year=w.get("year"),
                   external_ids={"openalex": w["id"]}, provenance=["openalex"])
        g.add_node(node)
        oa_count += 1
    print(f"  OpenAlex: +{oa_count} texts")

    # 1d: NDL Japanese records
    with open(DATA / "ndl" / "ndl_records.json") as f:
        ndl_data = json.load(f)
    ndl_count = 0
    for query_name, records in ndl_data.items():
        for rec in records[:20]:
            title = (rec.get("title", [""])[0])[:100]
            if not title:
                continue
            uid = f"text:ndl:{hash(title) % 1000000:06d}"
            creator = rec.get("creator", [""])[0]
            node = Text(uid=uid, label=title, language="ja",
                       external_ids={"ndl": rec.get("identifier", [""])[0]},
                       provenance=["ndl"])
            g.add_node(node)
            ndl_count += 1
    print(f"  NDL: +{ndl_count} Japanese texts")

    # 1e: Internet Archive metadata
    with open(DATA / "internet_archive" / "ia_philosophy.json") as f:
        ia_data = json.load(f)
    ia_count = 0
    for category, items in ia_data.items():
        for item in items[:10]:
            title = item.get("title", "")
            if not title:
                continue
            uid = f"text:ia:{item.get('identifier', str(hash(title) % 1000000))}"
            node = Text(uid=uid, label=title[:100],
                       external_ids={"ia": item.get("identifier", "")},
                       provenance=["internet_archive"])
            g.add_node(node)
            ia_count += 1
    print(f"  Internet Archive: +{ia_count} texts")

    summary = g.summary()
    print(f"\n  統合グラフ: {summary['total_nodes']} nodes, {summary['total_edges']} edges")
    print(f"  ノード種別: {summary['nodes_by_type']}")

    # Validation
    print("\n  【系譜検証】")
    tests = [
        ("thinker:confucius", "thinker:wang-yangming", "孔子→王陽明"),
        ("thinker:nagarjuna", "thinker:nishitani-keiji", "龍樹→西谷"),
        ("thinker:laozi", "thinker:zhuangzi", "老子→荘子"),
    ]
    for src, tgt, label in tests:
        paths = g.find_path(src, tgt, max_depth=6)
        if paths:
            names = [g.get_node(u).label for u in paths[0]]
            print(f"    {label}: {' → '.join(names)}")
        else:
            print(f"    {label}: 経路なし")

    return g


# ════════════════════════════════════════════════════════════════
# STAGE 2: 学派分類器の訓練データ構築
# ════════════════════════════════════════════════════════════════

def stage2_school_training_data():
    print("\n" + "=" * 70)
    print("STAGE 2: 学派分類器の訓練データ構築")
    print("=" * 70)

    # Map SEP titles to schools
    TITLE_TO_SCHOOL = {
        "plato": "Platonic", "aristotle": "Aristotelian",
        "stoic": "Stoic", "epicur": "Epicurean",
        "neoplatonism": "Neoplatonic", "aquinas": "Scholastic",
        "descartes": "Rationalist", "spinoza": "Rationalist",
        "leibniz": "Rationalist", "locke": "Empiricist",
        "hume": "Empiricist", "berkeley": "Empiricist",
        "kant": "Kantian", "hegel": "German Idealism",
        "fichte": "German Idealism", "schelling": "German Idealism",
        "husserl": "Phenomenology", "heidegger": "Phenomenology",
        "merleau-ponty": "Phenomenology",
        "sartre": "Existentialism", "kierkegaard": "Existentialism",
        "camus": "Existentialism", "nietzsche": "Existentialism",
        "wittgenstein": "Analytic", "russell": "Analytic",
        "frege": "Analytic", "carnap": "Analytic", "quine": "Analytic",
        "pragmati": "Pragmatism", "dewey": "Pragmatism", "james": "Pragmatism",
        "marx": "Critical Theory", "frankfurt school": "Critical Theory",
        "habermas": "Critical Theory", "adorno": "Critical Theory",
        "derrida": "Poststructuralism", "foucault": "Poststructuralism",
        "deleuze": "Poststructuralism",
        "confuci": "Confucian", "mencius": "Confucian",
        "daoi": "Daoist", "laozi": "Daoist", "zhuangzi": "Daoist",
        "buddhis": "Buddhist (Madhyamaka)", "nagarjuna": "Buddhist (Madhyamaka)",
        "zen": "Chan/Zen Buddhist", "chan": "Chan/Zen Buddhist",
        "neo-confuc": "Neo-Confucian", "zhu xi": "Neo-Confucian",
        "nyaya": "Nyaya", "vedanta": "Vedanta",
        "samkhya": "Samkhya", "yoga": "Yoga",
        "mimamsa": "Mimamsa", "jain": "Jain",
        "islamic": "Kalam", "al-farabi": "Falsafa", "ibn sina": "Falsafa",
        "process": "Process Philosophy", "whitehead": "Process Philosophy",
    }

    with open(DATA / "huggingface" / "sep_entries.json") as f:
        sep_entries = json.load(f)

    labeled_data = []
    for entry in sep_entries:
        title = entry.get("title", "").lower()
        text = entry.get("text_preview", "")
        if len(text) < 100:
            continue
        for keyword, school in TITLE_TO_SCHOOL.items():
            if keyword in title:
                labeled_data.append({"text": text, "school": school, "title": entry["title"]})
                break

    # Add Gutenberg-based samples
    gutenberg_school = {
        "1497_plato.txt": "Platonic",
        "9662_aristotle.txt": "Aristotelian",
        "5827_kant.txt": "Kantian",
        "4280_kant.txt": "Kantian",
        "4705_hume.txt": "Empiricist",
        "3600_locke.txt": "Empiricist",
        "10615_descartes.txt": "Rationalist",
        "7370_nietzsche.txt": "Existentialism",
        "38427_nietzsche.txt": "Existentialism",
        "30821_mill.txt": "Empiricist",
        "11224_hobbes.txt": "Social Contract",
        "1232_machiavelli.txt": "Political Realism",
    }

    gut_dir = DATA / "gutenberg"
    for fname, school in gutenberg_school.items():
        fpath = gut_dir / fname
        if fpath.exists():
            text = fpath.read_text(errors="ignore")
            # Extract 5 random 500-char passages
            text = text[text.find("***") + 3:]  # Skip Gutenberg header
            for _ in range(5):
                start = random.randint(0, max(0, len(text) - 600))
                passage = text[start:start + 500].strip()
                if len(passage) > 200:
                    labeled_data.append({"text": passage, "school": school, "source": "gutenberg"})

    # Count distribution
    school_counts = Counter(d["school"] for d in labeled_data)
    print(f"\nラベル付きデータ: {len(labeled_data)}パッセージ")
    print(f"学派分布:")
    for school, count in school_counts.most_common(20):
        print(f"  {school:30s} {count:4d}")

    return labeled_data


# ════════════════════════════════════════════════════════════════
# STAGE 3: 学派分類器の訓練と評価
# ════════════════════════════════════════════════════════════════

def stage3_school_classifier(labeled_data):
    print("\n" + "=" * 70)
    print("STAGE 3: 学派分類器の訓練と評価")
    print("=" * 70)

    model = SentenceTransformer(str(MODEL / "philmap-e5-finetuned-v2"))

    # Build school prototypes by averaging embeddings of all passages per school
    school_texts = defaultdict(list)
    for d in labeled_data:
        school_texts[d["school"]].append(d["text"])

    school_names = list(school_texts.keys())
    school_embs = {}
    for school, texts in school_texts.items():
        embs = model.encode([f"query: {t[:256]}" for t in texts[:20]], normalize_embeddings=True)
        school_embs[school] = embs.mean(axis=0)
        school_embs[school] /= np.linalg.norm(school_embs[school])

    prototype_matrix = np.array([school_embs[s] for s in school_names])

    # Evaluate: hold out 20% of data
    random.seed(42)
    random.shuffle(labeled_data)
    split = int(len(labeled_data) * 0.8)
    train_data = labeled_data[:split]
    test_data = labeled_data[split:]

    # Re-build prototypes from train only
    train_school_texts = defaultdict(list)
    for d in train_data:
        train_school_texts[d["school"]].append(d["text"])

    train_protos = {}
    for school in school_names:
        texts = train_school_texts.get(school, [])
        if texts:
            embs = model.encode([f"query: {t[:256]}" for t in texts[:20]], normalize_embeddings=True)
            train_protos[school] = embs.mean(axis=0)
            train_protos[school] /= np.linalg.norm(train_protos[school])

    # Classify test set
    correct_top1 = 0
    correct_top3 = 0
    results_detail = []
    for d in test_data:
        if d["school"] not in train_protos:
            continue
        emb = model.encode(f"query: {d['text'][:256]}", normalize_embeddings=True)
        sims = {s: float(np.dot(emb, proto)) for s, proto in train_protos.items()}
        ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)

        pred_top1 = ranked[0][0]
        pred_top3 = [r[0] for r in ranked[:3]]
        is_top1 = pred_top1 == d["school"]
        is_top3 = d["school"] in pred_top3

        if is_top1: correct_top1 += 1
        if is_top3: correct_top3 += 1
        results_detail.append({
            "expected": d["school"], "predicted": pred_top1,
            "correct_top1": is_top1, "correct_top3": is_top3,
        })

    n = len(results_detail)
    print(f"\nテストセット: {n}パッセージ")
    print(f"Top-1 正解率: {correct_top1}/{n} = {correct_top1/max(n,1):.1%}")
    print(f"Top-3 正解率: {correct_top3}/{n} = {correct_top3/max(n,1):.1%}")

    # Confusion analysis
    print(f"\n【誤分類パターン (Top-5)】")
    confusion = Counter()
    for r in results_detail:
        if not r["correct_top1"]:
            confusion[(r["expected"], r["predicted"])] += 1
    for (exp, pred), count in confusion.most_common(5):
        print(f"  {exp:25s} → {pred:25s} ({count}回)")

    return train_protos, school_names


# ════════════════════════════════════════════════════════════════
# STAGE 4: SEPからの概念オントロジー自動拡張
# ════════════════════════════════════════════════════════════════

def stage4_expand_ontology():
    print("\n" + "=" * 70)
    print("STAGE 4: SEPからの概念オントロジー自動拡張")
    print("=" * 70)

    from philtext.concept.ontology import PhilOntology, ConceptNode

    with open(DATA / "huggingface" / "sep_entries.json") as f:
        sep_entries = json.load(f)

    ontology = PhilOntology()

    # Extract concepts from SEP titles
    PHILOSOPHICAL_TERMS = {
        "epistemology", "metaphysics", "ethics", "logic", "aesthetics",
        "ontology", "phenomenology", "hermeneutics", "dialectics",
        "substance", "causation", "identity", "consciousness", "freedom",
        "justice", "virtue", "duty", "rights", "good", "truth",
        "knowledge", "belief", "perception", "reason", "intuition",
        "being", "existence", "essence", "form", "matter",
        "mind", "body", "soul", "self", "person",
        "god", "religion", "faith", "revelation",
        "language", "meaning", "reference", "proposition",
        "time", "space", "infinity", "necessity", "possibility",
        "action", "intention", "responsibility", "autonomy",
    }

    concept_count = 0
    for entry in sep_entries:
        title = entry.get("title", "")
        text_preview = entry.get("text_preview", "")

        if not title or not text_preview:
            continue

        # Check if title contains philosophical terms
        title_lower = title.lower()
        matched_terms = [t for t in PHILOSOPHICAL_TERMS if t in title_lower]

        if matched_terms or len(text_preview) > 200:
            cid = f"sep:{title_lower.replace(' ', '-')[:50]}"
            node = ConceptNode(
                id=cid,
                labels={"en": title},
                definition=text_preview[:300],
                school_associations=[],
            )
            ontology.add(node)
            concept_count += 1

    print(f"  SEPから抽出した概念: {concept_count}")
    print(f"  オントロジーサイズ: {len(ontology)}")

    # Show sample
    print(f"\n  【サンプル概念 (先頭10)】")
    for i, node in enumerate(ontology.all_concepts()):
        if i >= 10:
            break
        print(f"    {node.label('en'):50s}  {node.definition[:60]}...")

    return ontology


# ════════════════════════════════════════════════════════════════
# STAGE 5: 多言語コーパス構築と横断分析
# ════════════════════════════════════════════════════════════════

def stage5_multilingual_analysis():
    print("\n" + "=" * 70)
    print("STAGE 5: 多言語コーパス構築と横断分析")
    print("=" * 70)

    model = SentenceTransformer(str(MODEL / "philmap-e5-finetuned-v2"))

    # Collect representative passages from each language/tradition
    passages = {}

    # English (Gutenberg)
    gut = DATA / "gutenberg"
    if (gut / "1497_plato.txt").exists():
        text = (gut / "1497_plato.txt").read_text(errors="ignore")
        # Find the allegory of the cave area
        idx = text.lower().find("allegory")
        if idx < 0: idx = len(text) // 3
        passages["Plato (en)"] = text[idx:idx+500]

    if (gut / "5827_kant.txt").exists():
        text = (gut / "5827_kant.txt").read_text(errors="ignore")
        idx = text.lower().find("transcendental")
        if idx < 0: idx = len(text) // 4
        passages["Kant (en)"] = text[idx:idx+500]

    if (gut / "4705_hume.txt").exists():
        text = (gut / "4705_hume.txt").read_text(errors="ignore")
        idx = text.lower().find("cause")
        if idx < 0: idx = len(text) // 3
        passages["Hume (en)"] = text[idx:idx+500]

    # Chinese classical (CTP)
    ctp_conf = DATA / "ctp" / "confucianism.json"
    if ctp_conf.exists():
        with open(ctp_conf) as f:
            ctp = json.load(f)
        for text_data in ctp:
            if text_data.get("title") == "論語":
                chapters = text_data.get("chapters", [])
                if chapters:
                    content = json.dumps(chapters[0].get("content", ""), ensure_ascii=False)
                    passages["論語 (zh)"] = content[:500]
                break

    ctp_dao = DATA / "ctp" / "daoism.json"
    if ctp_dao.exists():
        with open(ctp_dao) as f:
            ctp = json.load(f)
        for text_data in ctp:
            if "道德" in text_data.get("title", ""):
                chapters = text_data.get("chapters", [])
                if chapters:
                    content = json.dumps(chapters[0].get("content", ""), ensure_ascii=False)
                    passages["道徳経 (zh)"] = content[:500]
                break

    # Greek (Perseus)
    perseus_f = DATA / "perseus" / "classical_texts.json"
    if perseus_f.exists():
        with open(perseus_f) as f:
            perseus = json.load(f)
        for name, data in perseus.items():
            sections = data.get("sections", [])
            if sections:
                text = sections[0].get("text", "")
                if text and len(text) > 100:
                    passages[f"{name} (grc)"] = text[:500]
                    break

    # Japanese philosophy concepts (from seed data - representing key texts)
    passages["西田幾多郎 (ja)"] = (
        "絶対無の場所においてこそ自覚が成立する。場所は包むものであり、"
        "包まれるものではない。有の場所、対立的無の場所、そして絶対無の場所。"
        "純粋経験は主客未分の直接的経験であり、それが最も根源的な実在である。"
    )
    passages["和辻哲郎 (ja)"] = (
        "人間の存在は本質的に間柄的である。人間という言葉は「人の間」を意味し、"
        "個人はつねにすでに他者との関係の中にある。倫理学は間柄の学でなければならない。"
    )

    if len(passages) < 3:
        print("  十分なテキストが見つかりません")
        return

    print(f"  収集パッセージ: {len(passages)}テキスト")
    for name in passages:
        print(f"    {name}: {len(passages[name])} chars")

    # Embed all passages
    names = list(passages.keys())
    texts = [f"query: {passages[n][:300]}" for n in names]
    embs = model.encode(texts, normalize_embeddings=True)

    # Similarity matrix
    print(f"\n  【多言語・多伝統 類似度マトリクス】\n")
    short_names = [n.split("(")[0].strip()[:12] for n in names]
    header = f"{'':>14s} | " + " | ".join(f"{s:>12s}" for s in short_names)
    print(header)
    print("-" * len(header))
    for i, name in enumerate(names):
        row = f"{short_names[i]:>14s} | "
        for j in range(len(names)):
            if i == j:
                row += f"{'---':>12s} | "
            else:
                s = float(np.dot(embs[i], embs[j]))
                row += f"{s:>12.3f} | "
        print(row)

    # Find most similar cross-tradition pairs
    print(f"\n  【異伝統間の類似度 Top-10】")
    pairs = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            s = float(np.dot(embs[i], embs[j]))
            pairs.append((names[i], names[j], s))
    pairs.sort(key=lambda x: x[2], reverse=True)
    for a, b, s in pairs[:10]:
        a_short = a.split("(")[0].strip()
        b_short = b.split("(")[0].strip()
        print(f"    {s:.3f}  {a_short:15s} ↔ {b_short}")


# ════════════════════════════════════════════════════════════════
# STAGE 6: 最終統計
# ════════════════════════════════════════════════════════════════

def stage6_final_stats(graph, ontology, train_protos):
    print("\n" + "=" * 70)
    print("STAGE 6: エコシステム最終統計")
    print("=" * 70)

    summary = graph.summary()

    print(f"""
┌─────────────────────────────────────────────────────┐
│          Digital Philosophy Ecosystem 統計           │
├─────────────────────────────────────────────────────┤
│ ナレッジグラフ                                       │
│   総ノード数:      {summary['total_nodes']:>6,}                          │
│   総エッジ数:      {summary['total_edges']:>6,}                          │
│   哲学者:          {summary['nodes_by_type'].get('Thinker',0):>6,}                          │
│   テキスト:        {summary['nodes_by_type'].get('Text',0):>6,}                          │
│   概念:            {summary['nodes_by_type'].get('Concept',0):>6,}                          │
│   伝統/学派:       {summary['nodes_by_type'].get('Tradition',0):>6,}                          │
│   影響関係:        {summary['edges_by_type'].get('influences',0):>6,}                          │
├─────────────────────────────────────────────────────┤
│ 概念オントロジー                                     │
│   SEP由来概念:     {len(ontology):>6,}                          │
│   シード概念:          30                          │
├─────────────────────────────────────────────────────┤
│ 学派分類器                                          │
│   分類可能学派数:  {len(train_protos):>6,}                          │
├─────────────────────────────────────────────────────┤
│ データソース                                         │
│   Wikidata:        哲学者2,000 + 影響3,305         │
│   OpenAlex:        論文2,000                        │
│   CTP:             中国古典9テキスト                 │
│   Perseus:         古典8テキスト                     │
│   Gutenberg:       全文13冊 (10.2MB)                │
│   NDL:             日本語392レコード                  │
│   Internet Archive: 1,000アイテム                    │
│   HuggingFace SEP: 1,000エントリ                    │
│   シードデータ:    概念ペア40組                      │
└─────────────────────────────────────────────────────┘
""")


# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Digital Philosophy Ecosystem - Full Integration Pipeline")
    print("=" * 70)

    random.seed(42)

    graph = stage1_unified_graph()
    labeled_data = stage2_school_training_data()
    train_protos, school_names = stage3_school_classifier(labeled_data)
    ontology = stage4_expand_ontology()
    stage5_multilingual_analysis()
    stage6_final_stats(graph, ontology, train_protos)

    print("パイプライン完了")


if __name__ == "__main__":
    main()
