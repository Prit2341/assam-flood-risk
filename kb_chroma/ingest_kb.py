"""
Ingest KNOWLEDGE_BASE.md into a local persistent ChromaDB collection.

KNOWLEDGE_BASE.md stays the source of truth for CORE/base knowledge
(mandatory rules, standing findings someone should read top-to-bottom).
This script chunks it by '## ' section and loads every section into
Chroma so older, more detailed findings can be semantically searched
instead of read in full every session.

Re-run any time KNOWLEDGE_BASE.md changes -- it wipes and rebuilds the
collection each run (idempotent, cheap: local embedding model, no API
calls, a few thousand lines takes seconds).

Usage:
    python ingest_kb.py
"""
import re
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parent.parent
KB_PATH = ROOT / "KNOWLEDGE_BASE.md"
DB_DIR = Path(__file__).resolve().parent / "db"
COLLECTION_NAME = "kb_sections"


def split_into_sections(text: str):
    """Split on top-level '## ' headers. Keeps the '# ' title as a
    zero-th preamble section. Multi-line headers (a '## ' line
    immediately followed by a continuation '## ' line with no body
    text between them) are joined into one section, since several
    entries in this KB wrap long titles across two '## ' lines."""
    lines = text.splitlines()
    sections = []
    current_title = "Preamble"
    current_body = []

    def flush():
        body = "\n".join(current_body).strip()
        if body:
            sections.append((current_title, body))

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## "):
            flush()
            title = line[3:].strip()
            # merge an immediately-following '## ' continuation line into the title
            if i + 1 < len(lines) and lines[i + 1].startswith("## "):
                title += " " + lines[i + 1][3:].strip()
                i += 1
            current_title = title
            current_body = [line]
        else:
            current_body.append(line)
        i += 1
    flush()
    return sections


def main():
    text = KB_PATH.read_text(encoding="utf-8")
    sections = split_into_sections(text)
    print(f"Parsed {len(sections)} sections from {KB_PATH}")

    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # Only replace the '.md'-derived sections (ids starting "section-").
    # Entries added directly via add_kb.py (ids starting "manual-") are
    # left untouched so re-running this script doesn't lose them.
    existing = collection.get()
    old_section_ids = [i for i in existing["ids"] if i.startswith("section-")]
    if old_section_ids:
        collection.delete(ids=old_section_ids)

    ids, docs, metas = [], [], []
    for idx, (title, body) in enumerate(sections):
        # crude date extraction, e.g. "(2026-07-30)" -- helps filtering/sorting later
        date_match = re.search(r"\((\d{4}-\d{2}-\d{2})", title)
        date = date_match.group(1) if date_match else ""
        ids.append(f"section-{idx:04d}")
        docs.append(body)
        metas.append({"title": title, "date": date, "order": idx})

    collection.add(ids=ids, documents=docs, metadatas=metas)
    print(f"Ingested {len(ids)} sections into Chroma collection "
          f"'{COLLECTION_NAME}' at {DB_DIR}")


if __name__ == "__main__":
    main()
