"""
Add ONE new detailed knowledge entry directly to the Chroma collection,
without re-ingesting the whole KNOWLEDGE_BASE.md.

Use this for the "more knowledge" (detailed findings, full narratives,
numbers, debugging trails) that would otherwise make KNOWLEDGE_BASE.md
grow unboundedly. KNOWLEDGE_BASE.md itself should only get a short
pointer/summary line for the same event -- see CLAUDE.md.

Usage:
    python add_kb.py --title "Some finding (2026-08-01)" --file entry.md
    python add_kb.py --title "Some finding (2026-08-01)" --text "..."
"""
import argparse
import re
from pathlib import Path

import chromadb

DB_DIR = Path(__file__).resolve().parent / "db"
COLLECTION_NAME = "kb_sections"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="path to a text/markdown file with the entry body")
    group.add_argument("--text", help="entry body given inline")
    args = parser.parse_args()

    body = Path(args.file).read_text(encoding="utf-8") if args.file else args.text

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    date_match = re.search(r"\((\d{4}-\d{2}-\d{2})", args.title)
    date = date_match.group(1) if date_match else ""

    # id = slug of title, collision-safe via count suffix
    existing = collection.count()
    entry_id = f"manual-{existing:04d}"

    collection.add(
        ids=[entry_id],
        documents=[f"## {args.title}\n\n{body}"],
        metadatas=[{"title": args.title, "date": date, "order": existing}],
    )
    print(f"Added entry '{args.title}' as {entry_id} to Chroma collection "
          f"'{COLLECTION_NAME}'. Total entries now: {collection.count()}")


if __name__ == "__main__":
    main()
