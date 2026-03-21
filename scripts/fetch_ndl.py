#!/usr/bin/env python3
"""Fetch Japanese philosophical texts metadata from NDL (National Diet Library)."""
import json, time
from pathlib import Path
from xml.etree import ElementTree as ET
import httpx

OUT = Path(__file__).parent.parent / "data" / "ndl"
SRU_BASE = "https://ndlsearch.ndl.go.jp/api/sru"

QUERIES = [
    ("nishida_kitaro", "西田幾多郎"),
    ("watsuji_tetsuro", "和辻哲郎"),
    ("nishitani_keiji", "西谷啓治"),
    ("tanabe_hajime", "田辺元"),
    ("philosophy_japanese", "日本哲学"),
    ("kyoto_school", "京都学派"),
    ("zen_philosophy", "禅 哲学"),
    ("buddhist_philosophy_jp", "仏教哲学"),
]

def search_ndl(query, max_records=50):
    params = {
        "operation": "searchRetrieve",
        "query": f'anywhere="{query}"',
        "recordSchema": "dc",
        "maximumRecords": max_records,
    }
    try:
        resp = httpx.get(SRU_BASE, params=params, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"  Error: {e}")
        return []

    records = []
    try:
        root = ET.fromstring(resp.text)
        ns = {"srw": "http://www.loc.gov/zing/srw/",
              "dc": "http://purl.org/dc/elements/1.1/"}
        for rec in root.findall(".//srw:record", ns):
            data = rec.find(".//srw:recordData", ns)
            if data is None:
                continue
            entry = {}
            for elem in data.iter():
                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                if tag in ["title", "creator", "date", "subject", "identifier", "publisher", "language"]:
                    entry.setdefault(tag, []).append(elem.text or "")
            if entry:
                records.append(entry)
    except ET.ParseError:
        pass
    return records

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for name, query in QUERIES:
        print(f"Searching NDL: {query}...")
        records = search_ndl(query, max_records=50)
        all_results[name] = records
        print(f"  Found {len(records)} records")
        time.sleep(1)

    with open(OUT / "ndl_records.json", "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in all_results.values())
    print(f"\nTotal: {total} records saved to {OUT}/")

if __name__ == "__main__":
    main()
