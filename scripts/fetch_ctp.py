#!/usr/bin/env python3
"""Fetch classical Chinese philosophical texts from Chinese Text Project API."""

import json
import time
from pathlib import Path

import httpx

BASE_URL = "https://api.ctext.org"
OUT_DIR = Path(__file__).parent.parent / "data" / "ctp"

# Core philosophical texts to fetch
TEXTS = {
    "confucianism": [
        "ctp:analects",        # 論語
        "ctp:mengzi",          # 孟子
        "ctp:daxue",           # 大学
        "ctp:zhongyong",       # 中庸
        "ctp:xunzi",           # 荀子
    ],
    "daoism": [
        "ctp:dao-de-jing",     # 道徳経
        "ctp:zhuangzi",        # 荘子
    ],
    "legalism": [
        "ctp:han-feizi",       # 韓非子
    ],
    "mohism": [
        "ctp:mozi",            # 墨子
    ],
}


def get_text_info(urn: str) -> dict | None:
    """Get metadata about a text."""
    try:
        resp = httpx.get(
            f"{BASE_URL}/gettextinfo",
            params={"urn": urn},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  Error fetching info for {urn}: {e}")
        return None


def get_text_content(urn: str) -> dict | None:
    """Get text content for a chapter/section."""
    try:
        resp = httpx.get(
            f"{BASE_URL}/gettext",
            params={"urn": urn},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  Error fetching text {urn}: {e}")
        return None


def fetch_text_with_chapters(urn: str) -> dict:
    """Fetch text info and first few chapters."""
    info = get_text_info(urn)
    if not info:
        return {"urn": urn, "error": "failed to fetch info"}

    result = {
        "urn": urn,
        "title": info.get("title", ""),
        "title_en": info.get("title_en", ""),
        "dynasty": info.get("dynasty", {}),
        "chapters": [],
    }

    # Get list of subsections
    subsections = info.get("subsections", [])
    if not subsections:
        # Try getting the text directly
        content = get_text_content(urn)
        if content:
            result["full_text"] = content
        return result

    # Fetch first 5 chapters to keep reasonable
    for sub in subsections[:5]:
        sub_urn = sub if isinstance(sub, str) else sub.get("urn", "")
        if not sub_urn:
            continue
        time.sleep(0.5)  # Rate limiting
        content = get_text_content(sub_urn)
        if content:
            result["chapters"].append({
                "urn": sub_urn,
                "content": content,
            })

    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_texts = {}

    for tradition, urns in TEXTS.items():
        print(f"\n=== {tradition} ===")
        tradition_texts = []
        for urn in urns:
            print(f"Fetching {urn}...")
            text_data = fetch_text_with_chapters(urn)
            tradition_texts.append(text_data)
            time.sleep(1.0)  # Rate limiting between texts

        all_texts[tradition] = tradition_texts
        with open(OUT_DIR / f"{tradition}.json", "w") as f:
            json.dump(tradition_texts, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(tradition_texts)} texts to {tradition}.json")

    # Summary
    with open(OUT_DIR / "manifest.json", "w") as f:
        summary = {
            tradition: [
                {"urn": t["urn"], "title": t.get("title", ""),
                 "chapters_fetched": len(t.get("chapters", []))}
                for t in texts
            ]
            for tradition, texts in all_texts.items()
        }
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nAll data saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
