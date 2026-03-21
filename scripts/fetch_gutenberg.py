#!/usr/bin/env python3
"""Fetch public domain philosophy texts from Project Gutenberg via Gutendex."""
import json, time
from pathlib import Path
import httpx

OUT = Path(__file__).parent.parent / "data" / "gutenberg"
GUTENDEX = "https://gutendex.com/books"

PHILOSOPHY_WORKS = [
    {"id": 1497, "title": "Republic", "author": "Plato"},
    {"id": 1656, "title": "Apology", "author": "Plato"},
    {"id": 5827, "title": "Critique of Pure Reason", "author": "Kant"},
    {"id": 4280, "title": "Critique of Practical Reason", "author": "Kant"},
    {"id": 5740, "title": "Meditations", "author": "Marcus Aurelius"},
    {"id": 3600, "title": "An Essay Concerning Human Understanding", "author": "Locke"},
    {"id": 4705, "title": "A Treatise of Human Nature", "author": "Hume"},
    {"id": 10615, "title": "Meditations on First Philosophy", "author": "Descartes"},
    {"id": 1232, "title": "The Prince", "author": "Machiavelli"},
    {"id": 7370, "title": "Beyond Good and Evil", "author": "Nietzsche"},
    {"id": 38427, "title": "Thus Spake Zarathustra", "author": "Nietzsche"},
    {"id": 30821, "title": "On Liberty", "author": "Mill"},
    {"id": 11224, "title": "Leviathan", "author": "Hobbes"},
    {"id": 9662, "title": "The Nicomachean Ethics", "author": "Aristotle"},
]

def fetch_text(gutenberg_id):
    """Fetch plain text from Gutenberg mirrors."""
    urls = [
        f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt",
        f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt",
    ]
    for url in urls:
        try:
            resp = httpx.get(url, timeout=30, follow_redirects=True)
            if resp.status_code == 200:
                return resp.text
        except Exception:
            continue
    return None

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for work in PHILOSOPHY_WORKS:
        print(f"Fetching: {work['author']} - {work['title']}...")
        text = fetch_text(work["id"])
        if text:
            # Save full text
            fname = f"{work['id']}_{work['author'].lower().replace(' ','_')}.txt"
            with open(OUT / fname, "w") as f:
                f.write(text)
            results.append({
                "id": work["id"], "title": work["title"],
                "author": work["author"], "chars": len(text),
                "file": fname,
            })
            print(f"  Saved: {len(text)} chars")
        else:
            print(f"  Failed to fetch")
        time.sleep(1)

    with open(OUT / "manifest.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(results)} texts to {OUT}/")

if __name__ == "__main__":
    main()
