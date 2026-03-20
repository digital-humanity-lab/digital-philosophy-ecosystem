"""Detailed concept comparison."""

from __future__ import annotations

from philmap.concept import Concept, ConceptDiff
from philmap.embedding.embedder import ConceptEmbedder


def _extract_key_phrases(concept: Concept) -> set[str]:
    phrases: set[str] = set()
    for desc in concept.descriptions:
        for token in desc.definition.lower().split():
            if len(token) > 3:
                phrases.add(token.strip(".,;:()"))
        for ctx in desc.usage_contexts:
            for token in ctx.lower().split():
                if len(token) > 3:
                    phrases.add(token.strip(".,;:()"))
    return phrases


def _generate_diff_narrative(
    a: Concept, b: Concept,
    shared: set[str], only_a: set[str], only_b: set[str],
    sims: dict[str, float],
) -> str:
    sim = sims.get("composite", 0.0)
    if sim > 0.8:
        rel = "strongly analogous"
    elif sim > 0.6:
        rel = "moderately similar"
    elif sim > 0.4:
        rel = "loosely related"
    else:
        rel = "largely distinct"

    lines = [
        f"{a.primary_term} ({a.tradition.name}) and "
        f"{b.primary_term} ({b.tradition.name}) are {rel} "
        f"(composite similarity: {sim:.3f}).",
    ]

    best_facet = max(sims, key=lambda k: sims[k])
    worst_facet = min(sims, key=lambda k: sims[k])
    lines.append(
        f"Strongest alignment on '{best_facet}' ({sims[best_facet]:.3f}), "
        f"weakest on '{worst_facet}' ({sims[worst_facet]:.3f})."
    )

    if shared:
        lines.append(f"Shared themes: {', '.join(list(shared)[:8])}.")
    if only_a:
        lines.append(f"Distinctive to {a.primary_term}: {', '.join(list(only_a)[:5])}.")
    if only_b:
        lines.append(f"Distinctive to {b.primary_term}: {', '.join(list(only_b)[:5])}.")

    return " ".join(lines)


def concept_diff(
    concept_a: Concept,
    concept_b: Concept,
    *,
    embedder: ConceptEmbedder,
) -> ConceptDiff:
    """Detailed comparison of two concepts across facets."""
    emb_a = embedder.embed(concept_a)
    emb_b = embedder.embed(concept_b)

    facet_sims: dict[str, float] = {}
    for facet in ("definition", "usage", "relational"):
        facet_sims[facet] = embedder.similarity(emb_a, emb_b, facet=facet)
    facet_sims["composite"] = embedder.similarity(emb_a, emb_b, facet="composite")

    a_keywords = _extract_key_phrases(concept_a)
    b_keywords = _extract_key_phrases(concept_b)
    shared = a_keywords & b_keywords
    only_a = a_keywords - b_keywords
    only_b = b_keywords - a_keywords

    narrative = _generate_diff_narrative(
        concept_a, concept_b, shared, only_a, only_b, facet_sims,
    )

    return ConceptDiff(
        concept_a=concept_a,
        concept_b=concept_b,
        shared_aspects=sorted(shared),
        unique_to_a=sorted(only_a),
        unique_to_b=sorted(only_b),
        similarity_by_facet=facet_sims,
        overall_similarity=facet_sims["composite"],
        narrative=narrative,
    )
