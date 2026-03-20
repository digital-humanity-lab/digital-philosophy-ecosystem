#!/usr/bin/env python3
"""Fine-tune embedding model on 15 ground-truth concept pairs + 5 negatives.

Uses Contrastive Loss (positive pairs should be close, negatives far apart)
with sentence-transformers training API.
"""

import sys
import time
import yaml
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "philmap" / "src"))

import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_OUT = Path(__file__).parent.parent / "models" / "philmap-e5-finetuned"


def build_text(cdata: dict) -> str:
    """Build rich text representation of a concept."""
    parts = []
    for key in ["term_ja", "term_zh", "term_sa", "term_de", "term_grc", "term_ar"]:
        if key in cdata:
            parts.append(cdata[key])
    if "term_en" in cdata:
        parts.append(cdata["term_en"])
    for key in ["definition_ja", "definition_zh", "definition_sa",
                "definition_de", "definition_en"]:
        if key in cdata:
            parts.append(cdata[key])
    for ctx in cdata.get("usage_contexts", []):
        parts.append(ctx)
    if "tradition" in cdata:
        parts.append(f"Tradition: {cdata['tradition']}")
    return " | ".join(parts)


def build_neg_text(cdata: dict) -> str:
    """Build text for negative pair concepts (minimal data)."""
    term = cdata.get("term", cdata.get("term_en", cdata.get("id", "")))
    defn = cdata.get("definition_en", term)
    trad = cdata.get("tradition", "")
    return f"{term} | {defn} | Tradition: {trad}"


def main():
    print("=" * 70)
    print("Fine-tuning multilingual-e5-base on philosophy concept pairs")
    print("=" * 70)

    with open(DATA_DIR / "seed" / "cross_tradition_pairs.yaml") as f:
        pairs_data = yaml.safe_load(f)

    # ── Build training examples ─────────────────────────────────
    train_examples = []

    # Positive pairs (label=1.0): all A-B, A-C, B-C combinations
    for pair in pairs_data["positive_pairs"]:
        concepts = []
        for key in ["concept_a", "concept_b", "concept_c"]:
            if key in pair:
                concepts.append(build_text(pair[key]))

        # All pairwise combinations within this group
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                train_examples.append(InputExample(
                    texts=[concepts[i], concepts[j]], label=1.0
                ))

    print(f"Positive training pairs: {sum(1 for e in train_examples if e.label == 1.0)}")

    # Negative pairs (label=0.0)
    for pair in pairs_data["negative_pairs"]:
        text_a = build_neg_text(pair["concept_a"])
        text_b = build_neg_text(pair["concept_b"])
        train_examples.append(InputExample(
            texts=[text_a, text_b], label=0.0
        ))

    # Generate additional hard negatives by pairing concepts from
    # different positive groups that are NOT known analogues
    all_concept_texts = {}
    for pair in pairs_data["positive_pairs"]:
        for key in ["concept_a", "concept_b", "concept_c"]:
            if key in pair:
                cid = pair[key]["id"]
                all_concept_texts[cid] = build_text(pair[key])

    # Known positive pairs (by ID)
    known_pos = set()
    for pair in pairs_data["positive_pairs"]:
        ids_in_pair = []
        for key in ["concept_a", "concept_b", "concept_c"]:
            if key in pair:
                ids_in_pair.append(pair[key]["id"])
        for i in range(len(ids_in_pair)):
            for j in range(i + 1, len(ids_in_pair)):
                known_pos.add((ids_in_pair[i], ids_in_pair[j]))
                known_pos.add((ids_in_pair[j], ids_in_pair[i]))

    # Sample random non-paired concepts as negatives
    concept_ids = list(all_concept_texts.keys())
    random.seed(42)
    hard_neg_count = 0
    for _ in range(60):
        a, b = random.sample(concept_ids, 2)
        if (a, b) not in known_pos:
            train_examples.append(InputExample(
                texts=[all_concept_texts[a], all_concept_texts[b]], label=0.2
            ))
            hard_neg_count += 1

    print(f"Explicit negative pairs: {len(pairs_data['negative_pairs'])}")
    print(f"Hard negative pairs: {hard_neg_count}")
    print(f"Total training examples: {len(train_examples)}")

    # ── Build evaluation set ────────────────────────────────────
    eval_sentences1 = []
    eval_sentences2 = []
    eval_scores = []

    for pair in pairs_data["positive_pairs"][:5]:  # First 5 for eval
        text_a = build_text(pair["concept_a"])
        text_b = build_text(pair["concept_b"])
        eval_sentences1.append(text_a)
        eval_sentences2.append(text_b)
        eval_scores.append(1.0)

    for pair in pairs_data["negative_pairs"][:3]:  # First 3 negatives for eval
        text_a = build_neg_text(pair["concept_a"])
        text_b = build_neg_text(pair["concept_b"])
        eval_sentences1.append(text_a)
        eval_sentences2.append(text_b)
        eval_scores.append(0.0)

    evaluator = EmbeddingSimilarityEvaluator(
        eval_sentences1, eval_sentences2, eval_scores,
        name="phil-concept-eval",
    )

    # ── Pre-training baseline ───────────────────────────────────
    print("\n--- Pre-training baseline ---")
    model = SentenceTransformer("intfloat/multilingual-e5-base")

    # Measure baseline discrimination
    baseline_pos, baseline_neg = measure_discrimination(
        model, pairs_data, all_concept_texts
    )
    print(f"Baseline positive avg: {baseline_pos:.3f}")
    print(f"Baseline negative avg: {baseline_neg:.3f}")
    print(f"Baseline separation:   {baseline_pos - baseline_neg:.3f}")

    # ── Training ────────────────────────────────────────────────
    print("\n--- Fine-tuning ---")
    random.shuffle(train_examples)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
    train_loss = losses.CosineSimilarityLoss(model)

    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=10,
        warmup_steps=10,
        output_path=str(MODEL_OUT),
        show_progress_bar=True,
    )
    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.1f}s")

    # ── Post-training evaluation ────────────────────────────────
    print("\n--- Post-training evaluation ---")
    model = SentenceTransformer(str(MODEL_OUT))

    post_pos, post_neg = measure_discrimination(
        model, pairs_data, all_concept_texts
    )
    print(f"Fine-tuned positive avg: {post_pos:.3f}")
    print(f"Fine-tuned negative avg: {post_neg:.3f}")
    print(f"Fine-tuned separation:   {post_pos - post_neg:.3f}")

    print(f"\nImprovement in separation: {(post_pos - post_neg) - (baseline_pos - baseline_neg):.3f}")

    # ── Detailed per-pair results ───────────────────────────────
    print("\n--- Per-pair results (fine-tuned) ---")
    prefix = "query: "

    for pair in pairs_data["positive_pairs"]:
        text_a = prefix + build_text(pair["concept_a"])
        text_b = prefix + build_text(pair["concept_b"])
        emb_a = model.encode(text_a, normalize_embeddings=True)
        emb_b = model.encode(text_b, normalize_embeddings=True)
        score = float(np.dot(emb_a, emb_b))
        a_term = pair["concept_a"].get("term_en", pair["concept_a"].get("term_ja", ""))
        b_term = pair["concept_b"].get("term_en", pair["concept_b"].get("term_de", ""))
        print(f"  [+] {a_term:25s} <-> {b_term:25s}  {score:.3f}")

    for pair in pairs_data["negative_pairs"]:
        text_a = prefix + build_neg_text(pair["concept_a"])
        text_b = prefix + build_neg_text(pair["concept_b"])
        emb_a = model.encode(text_a, normalize_embeddings=True)
        emb_b = model.encode(text_b, normalize_embeddings=True)
        score = float(np.dot(emb_a, emb_b))
        a_term = pair["concept_a"].get("term", "")
        b_term = pair["concept_b"].get("term", "")
        print(f"  [-] {a_term:25s} <-> {b_term:25s}  {score:.3f}")

    # Pairwise discrimination
    all_pos_scores = []
    all_neg_scores = []
    for pair in pairs_data["positive_pairs"]:
        text_a = prefix + build_text(pair["concept_a"])
        text_b = prefix + build_text(pair["concept_b"])
        emb_a = model.encode(text_a, normalize_embeddings=True)
        emb_b = model.encode(text_b, normalize_embeddings=True)
        all_pos_scores.append(float(np.dot(emb_a, emb_b)))
    for pair in pairs_data["negative_pairs"]:
        text_a = prefix + build_neg_text(pair["concept_a"])
        text_b = prefix + build_neg_text(pair["concept_b"])
        emb_a = model.encode(text_a, normalize_embeddings=True)
        emb_b = model.encode(text_b, normalize_embeddings=True)
        all_neg_scores.append(float(np.dot(emb_a, emb_b)))

    correct = sum(1 for p in all_pos_scores for n in all_neg_scores if p > n)
    total = len(all_pos_scores) * len(all_neg_scores)
    print(f"\nPairwise discrimination: {correct}/{total} = {correct/total:.1%}")


def measure_discrimination(model, pairs_data, all_concept_texts):
    """Measure avg positive and negative similarity."""
    prefix = "query: "
    pos_scores = []
    for pair in pairs_data["positive_pairs"]:
        text_a = prefix + build_text(pair["concept_a"])
        text_b = prefix + build_text(pair["concept_b"])
        emb_a = model.encode(text_a, normalize_embeddings=True)
        emb_b = model.encode(text_b, normalize_embeddings=True)
        pos_scores.append(float(np.dot(emb_a, emb_b)))

    neg_scores = []
    for pair in pairs_data["negative_pairs"]:
        text_a = prefix + build_neg_text(pair["concept_a"])
        text_b = prefix + build_neg_text(pair["concept_b"])
        emb_a = model.encode(text_a, normalize_embeddings=True)
        emb_b = model.encode(text_b, normalize_embeddings=True)
        neg_scores.append(float(np.dot(emb_a, emb_b)))

    return np.mean(pos_scores), np.mean(neg_scores)


if __name__ == "__main__":
    main()
