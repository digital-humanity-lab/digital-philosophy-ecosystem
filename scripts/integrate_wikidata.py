#!/usr/bin/env python3
"""Integrate Wikidata philosophers + 3305 influence relations into philgraph."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "philcore" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "philgraph" / "src"))

from philgraph import PhilGraph, Edge, EdgeType, EdgeProperties, ConsensusLevel
from philgraph.schema import Thinker, Concept, Tradition

DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    print("=" * 70)
    print("Wikidata Full Integration into PhilGraph")
    print("=" * 70)

    # ── Load Wikidata data ──────────────────────────────────────
    with open(DATA_DIR / "wikidata" / "philosophers.json") as f:
        philosophers = json.load(f)
    with open(DATA_DIR / "wikidata" / "influences.json") as f:
        influences = json.load(f)
    with open(DATA_DIR / "wikidata" / "movements.json") as f:
        movements = json.load(f)

    print(f"Loaded: {len(philosophers)} philosophers, "
          f"{len(influences)} influences, {len(movements)} movements")

    # ── Build graph ─────────────────────────────────────────────
    g = PhilGraph()

    # Also ingest seed data first
    g.ingest("manual", yaml_paths=[str(DATA_DIR / "seed" / "kyoto_school.yaml")])
    seed_summary = g.summary()
    print(f"\nSeed data: {seed_summary['total_nodes']} nodes, "
          f"{seed_summary['total_edges']} edges")

    # ── Add movements as Traditions ─────────────────────────────
    print("\nIngesting movements...")
    movement_count = 0
    for mv in movements:
        qid = mv["qid"]
        if not mv["label_en"]:
            continue
        node = Tradition(
            uid=f"tradition:wd:{qid}",
            label=mv["label_en"],
            labels_i18n={k: v for k, v in [("ja", mv.get("label_ja", ""))] if v},
            external_ids={"wikidata": qid},
            provenance=["wikidata"],
        )
        g.add_node(node)
        movement_count += 1
    print(f"  Added {movement_count} traditions/movements")

    # ── Add philosophers as Thinkers ────────────────────────────
    print("Ingesting philosophers...")
    thinker_count = 0
    qid_to_uid = {}
    for phil in philosophers:
        qid = phil["qid"]
        if not phil["label_en"]:
            continue

        uid = f"thinker:wd:{qid}"
        labels_i18n = {}
        for lang_key in ["label_ja", "label_zh", "label_de"]:
            lang = lang_key.split("_")[1]
            if phil.get(lang_key):
                labels_i18n[lang] = phil[lang_key]

        # Check if this thinker already exists (from seed data)
        existing = g.backend.get_node(uid)
        if existing:
            qid_to_uid[qid] = uid
            continue

        # Check by wikidata external ID
        existing_uid = g.resolve_external_id(qid, "wikidata")
        if existing_uid:
            qid_to_uid[qid] = existing_uid
            continue

        node = Thinker(
            uid=uid,
            label=phil["label_en"],
            labels_i18n=labels_i18n,
            birth_year=phil.get("birth_year"),
            death_year=phil.get("death_year"),
            nationality=phil.get("nationality", ""),
            external_ids={"wikidata": qid},
            provenance=["wikidata"],
        )
        g.add_node(node)
        qid_to_uid[qid] = uid
        thinker_count += 1
    print(f"  Added {thinker_count} thinkers (+ seed data)")

    # ── Add influence relations ─────────────────────────────────
    print("Ingesting influence relations...")
    influence_count = 0
    skipped = 0
    for inf in influences:
        src_qid = inf["influencer_qid"]
        tgt_qid = inf["philosopher_qid"]

        src_uid = qid_to_uid.get(src_qid, f"thinker:wd:{src_qid}")
        tgt_uid = qid_to_uid.get(tgt_qid, f"thinker:wd:{tgt_qid}")

        # Only add if both nodes exist
        if g.get_node(src_uid) is None or g.get_node(tgt_uid) is None:
            skipped += 1
            continue

        try:
            g.add_edge(Edge(
                source_uid=src_uid,
                target_uid=tgt_uid,
                edge_type=EdgeType.INFLUENCES,
                properties=EdgeProperties(
                    confidence=0.8,
                    consensus=ConsensusLevel.ESTABLISHED,
                    provenance="wikidata",
                    notes=f"Wikidata P737: {inf['influencer_label']} -> {inf['philosopher_label']}",
                ),
            ))
            influence_count += 1
        except ValueError:
            skipped += 1

    print(f"  Added {influence_count} influence edges (skipped {skipped})")

    # ── Summary ─────────────────────────────────────────────────
    summary = g.summary()
    print(f"\n{'=' * 70}")
    print(f"FINAL GRAPH SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total nodes: {summary['total_nodes']}")
    print(f"Total edges: {summary['total_edges']}")
    print(f"\nNodes by type:")
    for ntype, count in sorted(summary['nodes_by_type'].items()):
        print(f"  {ntype}: {count}")
    print(f"\nEdges by type:")
    for etype, count in sorted(summary['edges_by_type'].items()):
        print(f"  {etype}: {count}")

    # ── Validation checks ───────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("VALIDATION CHECKS")
    print(f"{'=' * 70}")

    # Check known lineages
    lineages = [
        ("Plato -> Aristotle",
         [("Plato", "Aristotle")]),
        ("Husserl -> Heidegger",
         [("Edmund Husserl", "Martin Heidegger"), ("Husserl", "Heidegger")]),
        ("Confucius -> Mencius",
         [("Confucius", "Mencius")]),
        ("Kant -> Hegel",
         [("Immanuel Kant", "Georg Wilhelm Friedrich Hegel"), ("Kant", "Hegel")]),
        ("Nagarjuna -> Nishida (via seed)",
         []),  # This is in seed data
    ]

    for name, label_pairs in lineages:
        found = False
        if not label_pairs:
            # Check seed data paths
            paths = g.find_path("thinker:nagarjuna", "thinker:nishida-kitaro")
            if paths:
                found = True
                labels = [g.get_node(uid).label for uid in paths[0]]
                print(f"  [PASS] {name}: {' -> '.join(labels)}")
            else:
                print(f"  [FAIL] {name}: no path found")
            continue

        for src_label, tgt_label in label_pairs:
            # Find UIDs by label
            src_uid = None
            tgt_uid = None
            for uid, node in g.iter_nodes("Thinker"):
                if src_label.lower() in node.label.lower():
                    src_uid = uid
                if tgt_label.lower() in node.label.lower():
                    tgt_uid = uid
            if src_uid and tgt_uid:
                edges = g.backend.get_edges(
                    source_uid=src_uid, target_uid=tgt_uid,
                    edge_type=EdgeType.INFLUENCES,
                )
                if edges:
                    found = True
                    break
                # Try reverse direction (influenced_by vs influences)
                edges_rev = g.backend.get_edges(
                    source_uid=tgt_uid, target_uid=src_uid,
                    edge_type=EdgeType.INFLUENCES,
                )
                if edges_rev:
                    found = True
                    break
        status = "PASS" if found else "FAIL"
        print(f"  [{status}] {name}")

    # Temporal consistency check
    print("\nTemporal consistency:")
    inconsistent = 0
    total_checked = 0
    for edge in g.iter_edges():
        if edge.edge_type == EdgeType.INFLUENCES:
            src = g.get_node(edge.source_uid)
            tgt = g.get_node(edge.target_uid)
            if (isinstance(src, Thinker) and isinstance(tgt, Thinker)
                    and src.birth_year and tgt.birth_year):
                total_checked += 1
                if src.birth_year > tgt.birth_year + 100:
                    inconsistent += 1
    consistency_rate = (total_checked - inconsistent) / max(total_checked, 1)
    print(f"  Checked {total_checked} influence edges with dates")
    print(f"  Inconsistent: {inconsistent}")
    print(f"  Consistency rate: {consistency_rate:.1%}")

    # Top influential philosophers
    print("\nTop 15 most influential philosophers (by out-degree):")
    influence_out = {}
    for edge in g.iter_edges():
        if edge.edge_type == EdgeType.INFLUENCES:
            influence_out[edge.source_uid] = influence_out.get(edge.source_uid, 0) + 1
    top = sorted(influence_out.items(), key=lambda x: x[1], reverse=True)[:15]
    for uid, count in top:
        node = g.get_node(uid)
        label = node.label if node else uid
        years = ""
        if isinstance(node, Thinker) and node.birth_year:
            years = f" ({node.birth_year}-{node.death_year or '?'})"
        print(f"  {count:3d} influences: {label}{years}")

    # Top influenced philosophers
    print("\nTop 15 most influenced philosophers (by in-degree):")
    influence_in = {}
    for edge in g.iter_edges():
        if edge.edge_type == EdgeType.INFLUENCES:
            influence_in[edge.target_uid] = influence_in.get(edge.target_uid, 0) + 1
    top_in = sorted(influence_in.items(), key=lambda x: x[1], reverse=True)[:15]
    for uid, count in top_in:
        node = g.get_node(uid)
        label = node.label if node else uid
        print(f"  {count:3d} influenced by: {label}")

    # Path discovery
    print("\nPath discovery examples:")
    path_queries = [
        ("Socrates", "Thomas Aquinas"),
        ("Confucius", "Zhu Xi"),
        ("Plato", "Martin Heidegger"),
    ]
    for src_name, tgt_name in path_queries:
        src_uid = tgt_uid = None
        for uid, node in g.iter_nodes("Thinker"):
            if src_name.lower() in node.label.lower():
                src_uid = uid
            if tgt_name.lower() in node.label.lower():
                tgt_uid = uid
        if src_uid and tgt_uid:
            paths = g.find_path(src_uid, tgt_uid, max_depth=6)
            if paths:
                path_labels = [g.get_node(u).label for u in paths[0]]
                print(f"  {src_name} -> {tgt_name}: {' -> '.join(path_labels)}")
            else:
                print(f"  {src_name} -> {tgt_name}: no path found (max_depth=6)")
        else:
            missing = []
            if not src_uid: missing.append(src_name)
            if not tgt_uid: missing.append(tgt_name)
            print(f"  {src_name} -> {tgt_name}: node(s) not found: {missing}")

    # Export
    print("\nExporting graph...")
    g.io.export_jsonld(str(DATA_DIR / "wikidata" / "full_graph.jsonld"))
    g.io.export_graphml(str(DATA_DIR / "wikidata" / "full_graph.graphml"))
    print(f"  Exported to data/wikidata/full_graph.{{jsonld,graphml}}")

    print(f"\n{'=' * 70}")
    print("INTEGRATION COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
