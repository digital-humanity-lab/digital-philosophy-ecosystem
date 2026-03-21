#!/usr/bin/env python3
"""Fetch PhilPapers taxonomy via OAI-PMH (no API key needed for metadata)."""
import json, time
from pathlib import Path
from xml.etree import ElementTree as ET
import httpx

OUT = Path(__file__).parent.parent / "data" / "philpapers"
OAI_BASE = "https://philpapers.org/oai2/oai2.php"

def fetch_oai_records(max_records=2000):
    records, token = [], None
    while len(records) < max_records:
        params = {"verb": "ListRecords", "metadataPrefix": "oai_dc"}
        if token:
            params = {"verb": "ListRecords", "resumptionToken": token}
        try:
            resp = httpx.get(OAI_BASE, params=params, timeout=60)
            resp.raise_for_status()
        except Exception as e:
            print(f"  OAI request failed: {e}")
            break
        root = ET.fromstring(resp.text)
        ns = {"oai": "http://www.openarchives.org/OAI/2.0/",
              "dc": "http://purl.org/dc/elements/1.1/"}
        for rec in root.findall(".//oai:record", ns):
            meta = rec.find(".//oai:metadata", ns)
            if meta is None:
                continue
            dc = meta.find(".//{http://www.openarchives.org/OAI/2.0/oai_dc/}dc")
            if dc is None:
                continue
            entry = {}
            for field in ["title", "creator", "date", "subject", "description", "identifier"]:
                els = dc.findall(f"dc:{field}", ns)
                entry[field] = [e.text for e in els if e.text]
            records.append(entry)
        # resumptionToken
        rt = root.find(".//oai:resumptionToken", ns)
        if rt is not None and rt.text:
            token = rt.text
            time.sleep(1)
        else:
            break
    return records[:max_records]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Fetching PhilPapers records via OAI-PMH...")
    records = fetch_oai_records(2000)
    with open(OUT / "records.json", "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(records)} records")
    # Extract unique subjects for taxonomy
    subjects = {}
    for r in records:
        for s in r.get("subject", []):
            subjects[s] = subjects.get(s, 0) + 1
    top_subjects = sorted(subjects.items(), key=lambda x: x[1], reverse=True)[:200]
    with open(OUT / "taxonomy_subjects.json", "w") as f:
        json.dump(top_subjects, f, ensure_ascii=False, indent=2)
    print(f"  Extracted {len(subjects)} unique subjects")

if __name__ == "__main__":
    main()
