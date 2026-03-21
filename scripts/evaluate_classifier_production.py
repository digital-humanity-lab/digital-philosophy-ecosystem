#!/usr/bin/env python3
"""Evaluate SchoolClassifier on production data from Gutenberg + SEP + CTP."""

import sys
import json
import random
import re
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "philtext" / "src"))

DATA = Path(__file__).parent.parent / "data"


def extract_gutenberg_passages():
    """Extract labeled passages from Gutenberg full texts."""
    AUTHOR_SCHOOL = {
        "1497_plato.txt": "Platonic",
        "1656_plato.txt": "Platonic",
        "9662_aristotle.txt": "Aristotelian",
        "5827_kant.txt": "Kantian",
        "4280_kant.txt": "Kantian",
        "4705_hume.txt": "Empiricist",
        "3600_locke.txt": "Empiricist",
        "10615_descartes.txt": "Rationalist",
        "7370_nietzsche.txt": "Existentialism",
        "38427_nietzsche.txt": "Existentialism",
        "30821_mill.txt": "Empiricist",
        "11224_hobbes.txt": "Empiricist",
    }
    passages = []
    gut_dir = DATA / "gutenberg"
    for fname, school in AUTHOR_SCHOOL.items():
        fp = gut_dir / fname
        if not fp.exists():
            continue
        text = fp.read_text(errors="ignore")
        # Skip Gutenberg header/footer
        start_marker = "*** START"
        end_marker = "*** END"
        si = text.find(start_marker)
        if si > 0:
            text = text[text.find("\n", si) + 1:]
        ei = text.find(end_marker)
        if ei > 0:
            text = text[:ei]

        # Extract random 300-char passages, skip very short or garbage
        random.seed(hash(fname))
        for _ in range(20):
            pos = random.randint(0, max(0, len(text) - 400))
            # Find sentence boundary
            start = text.rfind(".", pos - 50, pos)
            if start < 0:
                start = pos
            else:
                start += 1
            passage = text[start:start + 350].strip()
            # Clean up
            passage = re.sub(r"\s+", " ", passage)
            if len(passage) > 150 and not passage.startswith("Produced by"):
                passages.append({
                    "text": passage,
                    "school": school,
                    "source": f"gutenberg:{fname}",
                })
    return passages


def extract_sep_passages():
    """Extract labeled passages from SEP entries using title-to-school mapping."""
    TITLE_SCHOOL = {
        "plato": "Platonic",
        "aristotle": "Aristotelian",
        "stoicism": "Stoic",
        "epicurus": "Epicurean",
        "neoplatonism": "Neoplatonic",
        "aquinas": "Scholastic",
        "descartes": "Rationalist",
        "spinoza": "Rationalist",
        "leibniz": "Rationalist",
        "locke": "Empiricist",
        "hume": "Empiricist",
        "berkeley": "Empiricist",
        "kant": "Kantian",
        "hegel": "German Idealism",
        "fichte": "German Idealism",
        "husserl": "Phenomenology",
        "heidegger": "Phenomenology",
        "phenomenology": "Phenomenology",
        "sartre": "Existentialism",
        "kierkegaard": "Existentialism",
        "existentialism": "Existentialism",
        "nietzsche": "Existentialism",
        "wittgenstein": "Analytic",
        "russell": "Analytic",
        "frege": "Analytic",
        "logical-positivism": "Analytic",
        "pragmatism": "Pragmatism",
        "dewey": "Pragmatism",
        "james": "Pragmatism",
        "peirce": "Pragmatism",
        "critical-theory": "Critical Theory",
        "frankfurt-school": "Critical Theory",
        "habermas": "Critical Theory",
        "adorno": "Critical Theory",
        "marx": "Critical Theory",
        "derrida": "Poststructuralism",
        "foucault": "Poststructuralism",
        "deleuze": "Poststructuralism",
        "confucius": "Confucian",
        "mencius": "Confucian",
        "confucian": "Confucian",
        "daoist": "Daoist",
        "laozi": "Daoist",
        "zhuangzi": "Daoist",
        "buddhism": "Buddhist (Madhyamaka)",
        "nagarjuna": "Buddhist (Madhyamaka)",
        "madhyamaka": "Buddhist (Madhyamaka)",
        "zen": "Chan/Zen Buddhist",
        "chan": "Chan/Zen Buddhist",
        "neo-confucian": "Neo-Confucian",
        "zhu-xi": "Neo-Confucian",
        "wang-yangming": "Neo-Confucian",
        "nyaya": "Nyaya",
        "vedanta": "Vedanta",
        "advaita-vedanta": "Vedanta",
        "samkhya": "Samkhya",
        "yoga-philosophy": "Yoga",
        "mimamsa": "Mimamsa",
        "jainism": "Jain",
        "process-philosophy": "Process Philosophy",
        "whitehead": "Process Philosophy",
        "abhidharma": "Buddhist (Madhyamaka)",
    }

    passages = []
    sep_file = DATA / "huggingface" / "sep_entries.json"
    if not sep_file.exists():
        return passages

    with open(sep_file) as f:
        entries = json.load(f)

    seen_titles = set()
    for entry in entries:
        title = entry.get("title", "").lower().strip()
        text = entry.get("text_preview", "")
        if not title or not text or len(text) < 150 or title in seen_titles:
            continue
        seen_titles.add(title)

        school = None
        for keyword, s in TITLE_SCHOOL.items():
            if keyword in title:
                school = s
                break
        if school:
            passages.append({
                "text": text[:400],
                "school": school,
                "source": f"sep:{title}",
            })
    return passages


def extract_ctp_passages():
    """Extract labeled passages from CTP Chinese classical texts."""
    TRADITION_MAP = {
        "confucianism": "Confucian",
        "daoism": "Daoist",
        "legalism": "Legalist",
        "mohism": "Mohist",
    }
    passages = []
    for tradition, school in TRADITION_MAP.items():
        fp = DATA / "ctp" / f"{tradition}.json"
        if not fp.exists():
            continue
        with open(fp) as f:
            texts = json.load(f)
        for text_data in texts:
            chapters = text_data.get("chapters", [])
            for ch in chapters:
                content = ch.get("content", "")
                if isinstance(content, dict):
                    # Extract text from CTP format
                    text_parts = []
                    for key, val in content.items():
                        if isinstance(val, str):
                            text_parts.append(val)
                        elif isinstance(val, list):
                            text_parts.extend(str(v) for v in val)
                    content = " ".join(text_parts)
                elif isinstance(content, list):
                    content = " ".join(str(c) for c in content)
                content = str(content)
                if len(content) > 50:
                    passages.append({
                        "text": content[:400],
                        "school": school,
                        "source": f"ctp:{text_data.get('title', tradition)}",
                    })
    return passages


def main():
    print("=" * 70)
    print("SchoolClassifier 本番データ評価")
    print("=" * 70)

    # Collect all passages
    print("\nデータ収集...")
    gut_passages = extract_gutenberg_passages()
    sep_passages = extract_sep_passages()
    ctp_passages = extract_ctp_passages()

    all_passages = gut_passages + sep_passages + ctp_passages
    random.seed(42)
    random.shuffle(all_passages)

    print(f"  Gutenberg: {len(gut_passages)} passages")
    print(f"  SEP:       {len(sep_passages)} passages")
    print(f"  CTP:       {len(ctp_passages)} passages")
    print(f"  合計:       {len(all_passages)} passages")

    # Distribution
    school_dist = Counter(p["school"] for p in all_passages)
    print(f"\n学派分布:")
    for school, count in school_dist.most_common():
        print(f"  {school:30s} {count:4d}")

    # Load classifier
    from philtext.classify.school import SchoolClassifier

    clf = SchoolClassifier(
        method="prototype",
        embedding_model=str(Path(__file__).parent.parent / "models" / "philmap-e5-finetuned-v2"),
    )
    clf.load_default_examples()
    print(f"\n分類器: {len(clf.registered_schools)}学派, 各3-shot")

    # Filter to only schools we have prototypes for
    testable = [p for p in all_passages if p["school"] in clf.registered_schools]
    print(f"テスト可能パッセージ: {len(testable)} / {len(all_passages)}")

    # Evaluate
    print(f"\n{'=' * 70}")
    print("評価実行中...")
    print(f"{'=' * 70}")

    correct_top1 = 0
    correct_top3 = 0
    per_school_results = defaultdict(lambda: {"total": 0, "top1": 0, "top3": 0})
    confusion = Counter()
    errors = []

    for p in testable:
        pred = clf.classify(p["text"], top_k=3)
        expected = p["school"]

        is_top1 = pred.school == expected
        is_top3 = expected in [s for s, _ in pred.top_k]

        if is_top1:
            correct_top1 += 1
        if is_top3:
            correct_top3 += 1

        per_school_results[expected]["total"] += 1
        if is_top1:
            per_school_results[expected]["top1"] += 1
        if is_top3:
            per_school_results[expected]["top3"] += 1

        if not is_top1:
            confusion[(expected, pred.school)] += 1
            if len(errors) < 30:
                errors.append({
                    "expected": expected, "predicted": pred.school,
                    "confidence": pred.confidence,
                    "source": p["source"],
                    "text": p["text"][:80],
                })

    n = len(testable)
    top1_acc = correct_top1 / n if n else 0
    top3_acc = correct_top3 / n if n else 0

    print(f"\n{'=' * 70}")
    print(f"総合結果: {n}パッセージ")
    print(f"{'=' * 70}")
    print(f"  Top-1 正解率: {correct_top1}/{n} = {top1_acc:.1%}")
    print(f"  Top-3 正解率: {correct_top3}/{n} = {top3_acc:.1%}")

    # Per-school breakdown
    print(f"\n{'学派':30s} {'Total':>5s} {'Top-1':>8s} {'Top-3':>8s}")
    print("-" * 60)
    for school in sorted(per_school_results.keys()):
        r = per_school_results[school]
        t1 = r["top1"] / r["total"] if r["total"] else 0
        t3 = r["top3"] / r["total"] if r["total"] else 0
        print(f"  {school:28s} {r['total']:5d} {t1:8.1%} {t3:8.1%}")

    # Top confusion pairs
    print(f"\n誤分類パターン Top-10:")
    for (exp, pred), count in confusion.most_common(10):
        print(f"  {exp:25s} → {pred:25s} ({count}回)")

    # Sample errors
    print(f"\n誤分類サンプル:")
    for e in errors[:10]:
        print(f"  [{e['source']:30s}] 期待:{e['expected']:20s} → {e['predicted']:20s}")
        print(f"    {e['text']}...")
        print()

    # Per-source accuracy
    print(f"ソース別精度:")
    source_results = defaultdict(lambda: {"total": 0, "top1": 0, "top3": 0})
    for p in testable:
        src = p["source"].split(":")[0]
        pred = clf.classify(p["text"], top_k=3)
        source_results[src]["total"] += 1
        if pred.school == p["school"]:
            source_results[src]["top1"] += 1
        if p["school"] in [s for s, _ in pred.top_k]:
            source_results[src]["top3"] += 1
    for src, r in sorted(source_results.items()):
        t1 = r["top1"] / r["total"] if r["total"] else 0
        t3 = r["top3"] / r["total"] if r["total"] else 0
        print(f"  {src:15s} {r['total']:5d}件  Top-1: {t1:.1%}  Top-3: {t3:.1%}")


if __name__ == "__main__":
    main()
