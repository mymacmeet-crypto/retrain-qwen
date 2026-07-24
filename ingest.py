#!/usr/bin/env python3
"""
Ingestion script — indexes documents from the configured dataset path into
ChromaDB.  Reruns are safe: only new or modified files are re-indexed;
deleted files are removed from the collection.

Usage:
    python ingest.py              # full incremental run
    python ingest.py --reset      # wipe the collection and re-index everything
    python ingest.py --stats      # print collection statistics and exit
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm

from src.config import get_config
from src.document_loader import load_document, scan_directory
from src.embeddings import OllamaEmbedder
from src.text_splitter import TextSplitter
from src.vector_store import VectorStore

STATE_FILE = Path(__file__).parent / "data" / "ingestion_state.json"


# ---------------------------------------------------------------------------
# State management (tracks mtime/size per file to detect changes)
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def file_changed(path: Path, state: dict) -> bool:
    key = str(path.resolve())
    if key not in state:
        return True
    stat = path.stat()
    entry = state[key]
    return stat.st_mtime != entry.get("mtime") or stat.st_size != entry.get("size")


# ---------------------------------------------------------------------------
# Chunk ID generation (deterministic, stable across reruns)
# ---------------------------------------------------------------------------

def chunk_id(source: str, chunk_index: int) -> str:
    h = hashlib.md5(source.encode()).hexdigest()[:12]
    return f"{h}_{chunk_index:04d}"


# ---------------------------------------------------------------------------
# Main ingestion logic
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG knowledge base.")
    parser.add_argument("--reset", action="store_true", help="Wipe collection and re-index everything.")
    parser.add_argument("--stats", action="store_true", help="Print stats and exit.")
    args = parser.parse_args()

    cfg = get_config()
    ollama_cfg = cfg["ollama"]
    doc_cfg = cfg["documents"]
    chunk_cfg = cfg["chunking"]
    vs_cfg = cfg["vector_store"]

    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Salesforce RAG — Ingestion")
    print(f"{'='*60}")

    embedder = OllamaEmbedder(
        model=ollama_cfg["embedding_model"],
        base_url=ollama_cfg["base_url"],
        timeout=ollama_cfg["timeout"],
    )

    if not embedder.check_available():
        print(f"\n[ERROR] Embedding model '{ollama_cfg['embedding_model']}' not found in Ollama.")
        print(f"  Pull it with:  ollama pull {ollama_cfg['embedding_model']}")
        sys.exit(1)

    store = VectorStore(
        persist_directory=vs_cfg["persist_directory"],
        collection_name=vs_cfg["collection_name"],
    )

    if args.stats:
        _print_stats(store, load_state())
        return

    if args.reset:
        print("\n[RESET] Wiping existing collection and state …")
        # Delete & recreate collection
        import chromadb
        client = chromadb.PersistentClient(path=vs_cfg["persist_directory"])
        try:
            client.delete_collection(vs_cfg["collection_name"])
        except Exception:
            pass
        store = VectorStore(
            persist_directory=vs_cfg["persist_directory"],
            collection_name=vs_cfg["collection_name"],
        )
        state = {}
        save_state(state)
        print("[RESET] Done.")
    else:
        state = load_state()

    # ------------------------------------------------------------------
    # Scan filesystem
    # ------------------------------------------------------------------
    print(f"\nScanning: {doc_cfg['dataset_path']}")
    all_files = scan_directory(doc_cfg["dataset_path"], doc_cfg["supported_extensions"])
    print(f"Found {len(all_files)} supported files.")

    # Detect deleted files (in state but no longer on disk)
    disk_paths = {str(p.resolve()) for p in all_files}
    deleted = [s for s in state if s not in disk_paths]
    if deleted:
        print(f"\nRemoving {len(deleted)} deleted file(s) from index …")
        for src in deleted:
            store.delete_by_source(src)
            del state[src]
        save_state(state)

    # Identify files that need (re-)indexing
    to_index = [p for p in all_files if file_changed(p, state)]
    skipped = len(all_files) - len(to_index)
    print(f"{skipped} file(s) unchanged — skipping.")
    print(f"{len(to_index)} file(s) to index.\n")

    if not to_index:
        print("Nothing to do. Knowledge base is up to date.")
        _print_stats(store, state)
        return

    # ------------------------------------------------------------------
    # Index files
    # ------------------------------------------------------------------
    splitter = TextSplitter(
        chunk_size=chunk_cfg["chunk_size"],
        chunk_overlap=chunk_cfg["chunk_overlap"],
        min_chunk_size=chunk_cfg["min_chunk_size"],
    )
    batch_size = ollama_cfg["embedding_batch_size"]

    total_chunks = 0
    errors = 0
    t0 = time.time()

    for path in tqdm(to_index, desc="Indexing files", unit="file"):
        src_key = str(path.resolve())

        # Remove stale chunks for this file before reinserting
        if src_key in state:
            store.delete_by_source(src_key)

        doc = load_document(path)
        if doc is None:
            errors += 1
            # Mark as seen so we don't retry an unreadable file every run
            stat = path.stat()
            state[src_key] = {"mtime": stat.st_mtime, "size": stat.st_size, "chunks": 0, "skip": "unreadable"}
            continue

        chunks = splitter.split(doc)
        if not chunks:
            # File loaded but produced no indexable content (e.g. JS-rendered shell)
            stat = path.stat()
            state[src_key] = {"mtime": stat.st_mtime, "size": stat.st_size, "chunks": 0, "skip": "no_content"}
            continue

        # Embed in mini-batches
        texts = [c.text for c in chunks]
        try:
            embeddings = embedder.embed_batch(texts, batch_size=batch_size)
        except Exception as exc:
            tqdm.write(f"  [WARN] Embedding failed for {path.name}: {exc}")
            errors += 1
            continue

        ids = [chunk_id(doc.source, c.chunk_index) for c in chunks]
        metadatas = [
            {
                "source": c.source,
                "filename": c.filename,
                "file_type": c.file_type,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
                "char_start": c.char_start,
                "char_end": c.char_end,
            }
            for c in chunks
        ]

        store.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

        stat = path.stat()
        state[src_key] = {
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "chunks": len(chunks),
        }
        total_chunks += len(chunks)

    save_state(state)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Ingestion complete in {elapsed:.1f}s")
    print(f"  Chunks added/updated : {total_chunks}")
    print(f"  Files with errors    : {errors}")
    _print_stats(store, state)


def _print_stats(store: VectorStore, state: dict) -> None:
    indexed = {k: v for k, v in state.items() if isinstance(v, dict) and v.get("chunks", 0) > 0}
    skipped = {k: v for k, v in state.items() if isinstance(v, dict) and v.get("chunks", 0) == 0}
    total_ch = sum(v.get("chunks", 0) for v in indexed.values())
    print(f"\n--- Collection stats ---")
    print(f"  Total chunks in DB   : {store.count()}")
    print(f"  Files indexed        : {len(indexed)}")
    print(f"  Total chunks         : {total_ch}")
    print(f"  Files skipped        : {len(skipped)}  (JS-rendered shells / empty)")
    print()


if __name__ == "__main__":
    main()
