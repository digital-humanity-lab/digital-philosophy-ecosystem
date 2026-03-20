#!/usr/bin/env python3
"""Fetch philosopher data and influence relations from Wikidata SPARQL."""

import json
import sys
from pathlib import Path

import httpx

ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "philgraph/0.1.0 (digital-philosophy-ecosystem)"}
OUT_DIR = Path(__file__).parent.parent / "data" / "wikidata"


def sparql_query(query: str) -> list[dict]:
    resp = httpx.get(
        ENDPOINT,
        params={"query": query, "format": "json"},
        headers=HEADERS,
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()["results"]["bindings"]


def fetch_philosophers(limit: int = 2000) -> list[dict]:
    """Fetch philosophers with birth/death years, nationality, labels."""
    query = f"""
    SELECT ?item ?itemLabel ?itemDescription
           ?birthYear ?deathYear ?nationalityLabel
           ?itemLabel_ja ?itemLabel_zh ?itemLabel_de
    WHERE {{
      ?item wdt:P106 wd:Q4964182 .
      OPTIONAL {{ ?item wdt:P569 ?birth . BIND(YEAR(?birth) AS ?birthYear) }}
      OPTIONAL {{ ?item wdt:P570 ?death . BIND(YEAR(?death) AS ?deathYear) }}
      OPTIONAL {{ ?item wdt:P27 ?nationality . }}
      OPTIONAL {{ ?item rdfs:label ?itemLabel_ja . FILTER(LANG(?itemLabel_ja) = "ja") }}
      OPTIONAL {{ ?item rdfs:label ?itemLabel_zh . FILTER(LANG(?itemLabel_zh) = "zh") }}
      OPTIONAL {{ ?item rdfs:label ?itemLabel_de . FILTER(LANG(?itemLabel_de) = "de") }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
    }}
    LIMIT {limit}
    """
    print(f"Fetching up to {limit} philosophers...")
    rows = sparql_query(query)
    philosophers = []
    for row in rows:
        qid = row.get("item", {}).get("value", "").split("/")[-1]
        if not qid:
            continue
        philosophers.append({
            "qid": qid,
            "label_en": row.get("itemLabel", {}).get("value", ""),
            "label_ja": row.get("itemLabel_ja", {}).get("value", ""),
            "label_zh": row.get("itemLabel_zh", {}).get("value", ""),
            "label_de": row.get("itemLabel_de", {}).get("value", ""),
            "description": row.get("itemDescription", {}).get("value", ""),
            "birth_year": _int(row, "birthYear"),
            "death_year": _int(row, "deathYear"),
            "nationality": row.get("nationalityLabel", {}).get("value", ""),
        })
    print(f"  Got {len(philosophers)} philosophers")
    return philosophers


def fetch_influences(limit: int = 5000) -> list[dict]:
    """Fetch P737 (influenced by) relations between philosophers."""
    query = f"""
    SELECT ?philosopher ?philosopherLabel ?influencer ?influencerLabel
    WHERE {{
      ?philosopher wdt:P106 wd:Q4964182 .
      ?philosopher wdt:P737 ?influencer .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
    }}
    LIMIT {limit}
    """
    print(f"Fetching influence relations (P737)...")
    rows = sparql_query(query)
    influences = []
    for row in rows:
        influences.append({
            "philosopher_qid": row["philosopher"]["value"].split("/")[-1],
            "philosopher_label": row.get("philosopherLabel", {}).get("value", ""),
            "influencer_qid": row["influencer"]["value"].split("/")[-1],
            "influencer_label": row.get("influencerLabel", {}).get("value", ""),
        })
    print(f"  Got {len(influences)} influence relations")
    return influences


def fetch_philosophical_movements(limit: int = 500) -> list[dict]:
    """Fetch philosophical schools/movements."""
    query = f"""
    SELECT ?item ?itemLabel ?itemDescription
           ?itemLabel_ja
    WHERE {{
      ?item wdt:P31/wdt:P279* wd:Q179805 .
      OPTIONAL {{ ?item rdfs:label ?itemLabel_ja . FILTER(LANG(?itemLabel_ja) = "ja") }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
    }}
    LIMIT {limit}
    """
    print(f"Fetching philosophical movements...")
    rows = sparql_query(query)
    movements = []
    for row in rows:
        qid = row.get("item", {}).get("value", "").split("/")[-1]
        movements.append({
            "qid": qid,
            "label_en": row.get("itemLabel", {}).get("value", ""),
            "label_ja": row.get("itemLabel_ja", {}).get("value", ""),
            "description": row.get("itemDescription", {}).get("value", ""),
        })
    print(f"  Got {len(movements)} movements")
    return movements


def _int(row: dict, key: str) -> int | None:
    val = row.get(key, {}).get("value")
    if val:
        try:
            return int(float(val))
        except (ValueError, TypeError):
            pass
    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    philosophers = fetch_philosophers(2000)
    with open(OUT_DIR / "philosophers.json", "w") as f:
        json.dump(philosophers, f, ensure_ascii=False, indent=2)

    influences = fetch_influences(5000)
    with open(OUT_DIR / "influences.json", "w") as f:
        json.dump(influences, f, ensure_ascii=False, indent=2)

    movements = fetch_philosophical_movements(500)
    with open(OUT_DIR / "movements.json", "w") as f:
        json.dump(movements, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {OUT_DIR}/")
    print(f"  philosophers.json: {len(philosophers)} entries")
    print(f"  influences.json:   {len(influences)} relations")
    print(f"  movements.json:    {len(movements)} movements")


if __name__ == "__main__":
    main()
