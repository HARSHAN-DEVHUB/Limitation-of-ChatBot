from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.sqlite import SQLiteStore
from src.embedding.embedder import LocalEmbedder
from src.index.faiss_index import FaissStore


def main() -> None:
    db = SQLiteStore()
    chunks = db.fetch_chunks()
    if not chunks:
        print("No chunks found. Run ingest first.")
        return

    embedder = LocalEmbedder()
    vectors = embedder.encode([chunk["text"] for chunk in chunks])
    FaissStore().save(vectors, chunks)
    print(f"Indexed {len(chunks)} chunks")


if __name__ == "__main__":
    main()
