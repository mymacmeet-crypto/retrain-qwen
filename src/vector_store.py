"""ChromaDB-backed vector store with cosine similarity."""

from dataclasses import dataclass
from pathlib import Path

import chromadb


@dataclass
class SearchResult:
    text: str
    source: str
    filename: str
    chunk_index: int
    total_chunks: int
    distance: float   # cosine distance: 0 = identical, 2 = opposite
    doc_id: str


class VectorStore:
    """
    Thin wrapper around a ChromaDB persistent collection.

    Embeddings are supplied by the caller (OllamaEmbedder), so we use
    ChromaDB in embedding-free mode (no EmbeddingFunction attached).
    """

    def __init__(self, persist_directory: str, collection_name: str):
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            # cosine distance: lower = more similar
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert(
        self,
        ids: list,
        embeddings: list,
        documents: list,
        metadatas: list,
    ) -> None:
        """Insert or replace chunks.  Safe to call on already-indexed IDs."""
        self._col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def delete_by_source(self, source: str) -> None:
        """Remove all chunks that came from a given source file path."""
        result = self._col.get(where={"source": source}, include=[])
        if result["ids"]:
            self._col.delete(ids=result["ids"])

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def query(self, embedding: list, top_k: int = 5) -> list:
        """Return the top_k most similar chunks for a query embedding."""
        n = min(top_k, self.count())
        if n == 0:
            return []

        res = self._col.query(
            query_embeddings=[embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        for doc_id, doc, meta, dist in zip(
            res["ids"][0],
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        ):
            results.append(SearchResult(
                text=doc,
                source=meta.get("source", ""),
                filename=meta.get("filename", ""),
                chunk_index=int(meta.get("chunk_index", 0)),
                total_chunks=int(meta.get("total_chunks", 0)),
                distance=dist,
                doc_id=doc_id,
            ))
        return results

    def get_all_sources(self) -> set:
        """Return the set of source paths currently in the collection."""
        result = self._col.get(include=["metadatas"])
        return {m.get("source", "") for m in result["metadatas"]}

    def count(self) -> int:
        return self._col.count()
