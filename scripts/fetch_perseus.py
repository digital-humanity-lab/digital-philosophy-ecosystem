#!/usr/bin/env python3
"""Fetch classical philosophical texts from Perseus Digital Library CTS API."""
import json, time
from pathlib import Path
from xml.etree import ElementTree as ET
import httpx

OUT = Path(__file__).parent.parent / "data" / "perseus"
CTS_BASE = "https://scaife-cts.perseus.org/api/cts"

# Key philosophical works with their CTS URNs
WORKS = {
    "plato_republic": "urn:cts:greekLit:tlg0059.tlg030",
    "plato_apology": "urn:cts:greekLit:tlg0059.tlg002",
    "plato_phaedo": "urn:cts:greekLit:tlg0059.tlg004",
    "aristotle_metaphysics": "urn:cts:greekLit:tlg0086.tlg025",
    "aristotle_nicomachean": "urn:cts:greekLit:tlg0086.tlg010",
    "aristotle_politics": "urn:cts:greekLit:tlg0086.tlg035",
    "marcus_aurelius_meditations": "urn:cts:greekLit:tlg0562.tlg001",
    "epictetus_discourses": "urn:cts:greekLit:tlg0557.tlg001",
}

def fetch_valid_refs(urn):
    try:
        resp = httpx.get(CTS_BASE, params={"request": "GetValidReff", "urn": urn, "level": "1"}, timeout=30)
        if resp.status_code != 200:
            return []
        root = ET.fromstring(resp.text)
        refs = []
        for ref in root.iter():
            if ref.text and ref.text.strip().startswith("urn:"):
                refs.append(ref.text.strip())
        return refs
    except Exception as e:
        print(f"  Error getting refs for {urn}: {e}")
        return []

def fetch_passage(urn):
    try:
        resp = httpx.get(CTS_BASE, params={"request": "GetPassage", "urn": urn}, timeout=30)
        if resp.status_code != 200:
            return ""
        root = ET.fromstring(resp.text)
        texts = []
        for elem in root.iter():
            if elem.text:
                texts.append(elem.text.strip())
        return " ".join(texts)
    except Exception:
        return ""

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, urn in WORKS.items():
        print(f"Fetching {name}...")
        refs = fetch_valid_refs(urn)
        if not refs:
            # Try alternative: just get the first few sections directly
            text = fetch_passage(urn)
            results[name] = {"urn": urn, "refs": 0, "sample_text": text[:2000]}
            print(f"  Direct fetch: {len(text)} chars")
        else:
            # Fetch first 5 sections
            sections = []
            for ref in refs[:5]:
                text = fetch_passage(ref)
                if text:
                    sections.append({"ref": ref, "text": text[:1000]})
                time.sleep(0.5)
            results[name] = {"urn": urn, "refs": len(refs), "sections": sections}
            print(f"  {len(refs)} refs, fetched {len(sections)} sections")
        time.sleep(1)

    with open(OUT / "classical_texts.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {OUT}/classical_texts.json")

if __name__ == "__main__":
    main()
