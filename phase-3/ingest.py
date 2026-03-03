"""
Phase 3 - Step 1: Ingest notes into ChromaDB

Usage:
    uv run python phase-3/ingest.py

What it does:
    1. Read all .md files from phase-3/notes/
    2. Chunk text (fixed-size with overlap)
    3. Embed each chunk using ChromaDB's default local embedding function
       (sentence-transformers/all-MiniLM-L6-v2, runs locally, no API needed)
    4. Store into ChromaDB at ./phase-3/chroma_db/

Why local embedding?
    The remote proxy API blocks Chinese text (returns HTML error page).
    ChromaDB's built-in embedding function works offline and supports
    multilingual content via the all-MiniLM-L6-v2 model.
"""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
NOTES_DIR = Path(__file__).parent / "notes"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "notes"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Local embedding function (downloads model on first run, ~90MB)
_embed_fn = DefaultEmbeddingFunction()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping fixed-size chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def ingest():
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Drop and recreate collection for a clean ingest
    try:
        chroma.delete_collection(COLLECTION_NAME)
        print(f"[ingest] Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass
    # Use local embedding function for the collection
    collection = chroma.create_collection(
        COLLECTION_NAME,
        embedding_function=_embed_fn,
    )

    md_files = sorted(NOTES_DIR.glob("*.md"))
    if not md_files:
        print(f"[ingest] No .md files found in {NOTES_DIR}")
        return

    all_ids: list[str] = []
    all_docs: list[str] = []
    all_metadatas: list[dict] = []

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        print(f"[ingest] {md_file.name}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            chunk_id = f"{md_file.stem}_chunk_{i}"
            all_ids.append(chunk_id)
            all_docs.append(chunk)
            all_metadatas.append({"source": md_file.name, "chunk_index": i})

    print("[ingest] Embedding and storing (local model, first run may take 10-30s)...")
    # ChromaDB auto-embeds when embedding_function is set on the collection
    collection.add(
        ids=all_ids,
        documents=all_docs,
        metadatas=all_metadatas,
    )
    print(f"\n[ingest] Done! Stored {len(all_ids)} chunks into '{COLLECTION_NAME}'")
    print(f"[ingest] ChromaDB path: {CHROMA_DIR.resolve()}")


if __name__ == "__main__":
    ingest()
