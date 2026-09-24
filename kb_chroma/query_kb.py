"""
Semantic search over the ingested KNOWLEDGE_BASE.md sections.

Usage:
    python query_kb.py "your question or topic here" [--n 5]
"""
import argparse
import sys
from pathlib import Path

import chromadb

sys.stdout.reconfigure(encoding="utf-8")

DB_DIR = Path(__file__).resolve().parent / "db"
COLLECTION_NAME = "kb_sections"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(query_texts=[args.query], n_results=args.n)

    for rank, (doc, meta, dist) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0]),
        start=1,
    ):
        print(f"\n{'=' * 80}")
        print(f"#{rank}  [{meta['title']}]  (distance={dist:.4f})")
        print("=" * 80)
        print(doc[:1200] + ("..." if len(doc) > 1200 else ""))


if __name__ == "__main__":
    main()
