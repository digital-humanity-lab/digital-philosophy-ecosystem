#!/usr/bin/env python3
"""Fetch SEP dataset and philosophy datasets from HuggingFace."""
import json
from pathlib import Path

OUT = Path(__file__).parent.parent / "data" / "huggingface"

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        return

    # Stanford Encyclopedia of Philosophy
    print("Loading SEP dataset from HuggingFace...")
    try:
        ds = load_dataset("AiresPucrs/stanford-encyclopedia-philosophy", split="train")
        # Extract first 500 entries with title and first 500 chars of text
        entries = []
        for i, item in enumerate(ds):
            if i >= 1000:
                break
            entries.append({
                "title": item.get("title", ""),
                "text_preview": (item.get("text", "") or "")[:500],
                "url": item.get("url", ""),
            })
        with open(OUT / "sep_entries.json", "w") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(entries)} SEP entries")
    except Exception as e:
        print(f"  SEP dataset error: {e}")

    print(f"\nDone: {OUT}/")

if __name__ == "__main__":
    main()
