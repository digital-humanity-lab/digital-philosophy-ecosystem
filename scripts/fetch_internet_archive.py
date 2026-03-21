#!/usr/bin/env python3
"""Fetch philosophy text metadata from Internet Archive."""
import json, time
from pathlib import Path
import httpx

OUT = Path(__file__).parent.parent / "data" / "internet_archive"
SEARCH_URL = "https://archive.org/advancedsearch.php"

QUERIES = [
    ("eastern_philosophy", "subject:philosophy AND subject:eastern AND mediatype:texts"),
    ("indian_philosophy", "subject:Indian philosophy AND mediatype:texts"),
    ("chinese_philosophy", "subject:Chinese philosophy AND mediatype:texts"),
    ("japanese_philosophy", "subject:Japanese philosophy AND mediatype:texts"),
    ("islamic_philosophy", "subject:Islamic philosophy AND mediatype:texts"),
    ("african_philosophy", "subject:African philosophy AND mediatype:texts"),
    ("ancient_philosophy", "subject:philosophy AND subject:ancient AND mediatype:texts"),
    ("kant", "creator:Kant AND subject:philosophy AND mediatype:texts"),
    ("hegel", "creator:Hegel AND subject:philosophy AND mediatype:texts"),
    ("buddhist_philosophy", "subject:Buddhist philosophy AND mediatype:texts"),
]

def search_ia(query, rows=100):
    params = {
        "q": query, "output": "json", "rows": rows,
        "fl[]": ["identifier", "title", "creator", "date", "subject", "language", "downloads"],
    }
    try:
        resp = httpx.get(SEARCH_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", {}).get("docs", [])
    except Exception as e:
        print(f"  Error: {e}")
        return []

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for name, query in QUERIES:
        print(f"Searching IA: {name}...")
        docs = search_ia(query, rows=100)
        all_results[name] = docs
        print(f"  Found {len(docs)} items")
        time.sleep(1)

    with open(OUT / "ia_philosophy.json", "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in all_results.values())
    print(f"\nTotal: {total} items saved to {OUT}/")

if __name__ == "__main__":
    main()
