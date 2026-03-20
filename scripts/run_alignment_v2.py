#!/usr/bin/env python3
"""V2: Improved alignment with contrastive prompting and usage-context enrichment."""

import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "philmap" / "src"))

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent.parent / "data"


def load_pairs():
    with open(DATA_DIR / "seed" / "cross_tradition_pairs.yaml") as f:
        return yaml.safe_load(f)


def build_contrastive_text(cdata: dict) -> str:
    """Build a richer representation that emphasizes distinguishing features."""
    parts = []
    # Term in original language
    for key in ["term_ja", "term_zh", "term_sa", "term_de", "term_grc", "term_ar"]:
        if key in cdata:
            parts.append(cdata[key])
    # English term
    if "term_en" in cdata:
        parts.append(cdata["term_en"])
    # Definition (prefer original language, then English)
    for key in ["definition_ja", "definition_zh", "definition_sa",
                "definition_de", "definition_en"]:
        if key in cdata:
            parts.append(cdata[key])
            break
    # Usage contexts add crucial distinguishing information
    for ctx in cdata.get("usage_contexts", []):
        parts.append(ctx)
    # Tradition name as context
    if "tradition" in cdata:
        parts.append(f"Tradition: {cdata['tradition']}")
    return " | ".join(parts)


def main():
    pairs_data = load_pairs()

    print("Loading model...")
    model = SentenceTransformer("intfloat/multilingual-e5-base")

    # Strategy 1: Use instruction-tuned query format
    prefix = "query: "

    # Build all concept texts
    all_concepts = {}
    for pair in pairs_data["positive_pairs"]:
        for key in ["concept_a", "concept_b", "concept_c"]:
            if key in pair:
                cdata = pair[key]
                text = build_contrastive_text(cdata)
                all_concepts[cdata["id"]] = {
                    "text": text,
                    "term": cdata.get("term_en", cdata.get("term_ja", cdata.get("term_zh", cdata["id"]))),
                    "tradition": cdata.get("tradition", ""),
                }

    for pair in pairs_data["negative_pairs"]:
        for key in ["concept_a", "concept_b"]:
            cdata = pair[key]
            cid = cdata["id"]
            if cid not in all_concepts:
                # Negative pairs have minimal data; enrich from positive pairs if possible
                text = f"{cdata.get('term', '')} | {cdata.get('definition_en', cdata.get('term', ''))} | Tradition: {cdata.get('tradition', '')}"
                all_concepts[cid] = {
                    "text": text,
                    "term": cdata.get("term", cid),
                    "tradition": cdata.get("tradition", ""),
                }

    # Encode all concepts
    print(f"Encoding {len(all_concepts)} concepts...")
    ids = list(all_concepts.keys())
    texts = [prefix + all_concepts[cid]["text"] for cid in ids]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    emb_map = {cid: emb for cid, emb in zip(ids, embeddings)}

    def sim(a_id, b_id):
        return float(np.dot(emb_map[a_id], emb_map[b_id]))

    # ── Results ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("POSITIVE PAIRS")
    print("=" * 70)

    pos_scores = []
    for pair in pairs_data["positive_pairs"]:
        a_id = pair["concept_a"]["id"]
        b_id = pair["concept_b"]["id"]
        score = sim(a_id, b_id)
        pos_scores.append(score)
        a_term = all_concepts[a_id]["term"]
        b_term = all_concepts[b_id]["term"]
        print(f"  {a_term:25s} <-> {b_term:30s}  {score:.3f}  (exp: {pair['expected_similarity']})")

        if "concept_c" in pair:
            c_id = pair["concept_c"]["id"]
            c_term = all_concepts[c_id]["term"]
            print(f"    {a_term:23s} <-> {c_term:30s}  {sim(a_id, c_id):.3f}")
            print(f"    {b_term:23s} <-> {c_term:30s}  {sim(b_id, c_id):.3f}")

    print("\n" + "=" * 70)
    print("NEGATIVE PAIRS")
    print("=" * 70)

    neg_scores = []
    for pair in pairs_data["negative_pairs"]:
        a_id = pair["concept_a"]["id"]
        b_id = pair["concept_b"]["id"]
        score = sim(a_id, b_id)
        neg_scores.append(score)
        a_term = all_concepts[a_id]["term"]
        b_term = all_concepts[b_id]["term"]
        print(f"  {a_term:25s} <-> {b_term:30s}  {score:.3f}  (exp: {pair['expected_similarity']})")

    # ── Full similarity matrix for interesting subset ────────────
    print("\n" + "=" * 70)
    print("SIMILARITY MATRIX (selected concepts)")
    print("=" * 70)

    # Select key concepts for the matrix
    matrix_ids = [
        "watsuji.aidagara", "buber.i_thou", "ubuntu.ubuntu",
        "nishida.zettaimu", "nagarjuna.sunyata", "heidegger.nichts",
        "confucius.ren", "buddhism.karuna", "christianity.agape",
        "nishida.basho", "heidegger.lichtung",
    ]
    matrix_ids = [mid for mid in matrix_ids if mid in emb_map]

    if matrix_ids:
        terms = [all_concepts[cid]["term"][:12] for cid in matrix_ids]
        # Header
        header = f"{'':>14s} | " + " | ".join(f"{t:>12s}" for t in terms)
        print(header)
        print("-" * len(header))
        for i, a_id in enumerate(matrix_ids):
            row = f"{terms[i]:>14s} | "
            for j, b_id in enumerate(matrix_ids):
                s = sim(a_id, b_id)
                if i == j:
                    row += f"{'---':>12s} | "
                else:
                    row += f"{s:>12.3f} | "
            print(row)

    # ── find_analogues simulation ───────────────────────────────
    print("\n" + "=" * 70)
    print("FIND ANALOGUES: What is most similar to 間柄 (aidagara)?")
    print("=" * 70)

    if "watsuji.aidagara" in emb_map:
        query_id = "watsuji.aidagara"
        scores_all = []
        for cid in ids:
            if cid == query_id:
                continue
            s = sim(query_id, cid)
            scores_all.append((cid, all_concepts[cid]["term"],
                              all_concepts[cid]["tradition"], s))
        scores_all.sort(key=lambda x: x[3], reverse=True)
        for cid, term, trad, s in scores_all[:10]:
            print(f"  {s:.3f}  {term:30s}  ({trad})")

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_pos = sum(pos_scores) / len(pos_scores)
    avg_neg = sum(neg_scores) / len(neg_scores)
    separation = avg_pos - avg_neg

    print(f"Positive avg: {avg_pos:.3f}  (min={min(pos_scores):.3f}, max={max(pos_scores):.3f})")
    print(f"Negative avg: {avg_neg:.3f}  (min={min(neg_scores):.3f}, max={max(neg_scores):.3f})")
    print(f"Separation:   {separation:.3f}")

    # Pairwise discrimination
    correct = sum(1 for p in pos_scores for n in neg_scores if p > n)
    total = len(pos_scores) * len(neg_scores)
    print(f"Pairwise discrimination: {correct}/{total} = {correct/total:.1%}")

    # Ranking analysis
    all_scores = [(s, "pos") for s in pos_scores] + [(s, "neg") for s in neg_scores]
    all_scores.sort(key=lambda x: x[0], reverse=True)
    print(f"\nRanked scores (top=highest similarity):")
    for i, (s, label) in enumerate(all_scores):
        marker = "+" if label == "pos" else "-"
        print(f"  {i+1:2d}. [{marker}] {s:.3f}")


if __name__ == "__main__":
    main()
