#!/usr/bin/env python3
"""Fetch philosophy papers and authors from OpenAlex API."""
import json, time
from pathlib import Path
import httpx

OUT = Path(__file__).parent.parent / "data" / "openalex"
BASE = "https://api.openalex.org"
PARAMS = {"mailto": "digital-philosophy@example.com"}

def fetch_pages(endpoint, filters, max_items=2000):
    items, cursor = [], "*"
    while len(items) < max_items:
        params = {**PARAMS, "filter": filters, "per_page": 200, "cursor": cursor}
        resp = httpx.get(f"{BASE}/{endpoint}", params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            break
        items.extend(results)
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(0.2)
    return items[:max_items]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Fetching philosophy works from OpenAlex...")
    works = fetch_pages("works", "concepts.id:C138885662", max_items=2000)
    with open(OUT / "philosophy_works.json", "w") as f:
        json.dump([{
            "id": w.get("id",""), "title": w.get("title",""),
            "year": w.get("publication_year"),
            "doi": w.get("doi"),
            "cited_by": w.get("cited_by_count", 0),
            "authors": [a.get("author",{}).get("display_name","") for a in w.get("authorships",[])],
            "concepts": [c.get("display_name","") for c in w.get("concepts",[])[:5]],
            "abstract": w.get("abstract_inverted_index") is not None,
        } for w in works], f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(works)} works")

    print("Fetching philosophy journals...")
    resp = httpx.get(f"{BASE}/sources", params={**PARAMS, "filter": "concepts.id:C138885662", "per_page": 100}, timeout=60)
    journals = resp.json().get("results", [])
    with open(OUT / "philosophy_journals.json", "w") as f:
        json.dump([{"id": j.get("id",""), "name": j.get("display_name",""),
                    "works_count": j.get("works_count",0)} for j in journals], f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(journals)} journals")

    print(f"Done: {OUT}/")

if __name__ == "__main__":
    main()
