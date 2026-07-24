"""Core RAG pipeline: embed query → retrieve → build prompt → generate."""

import sys
from typing import Iterator

import requests

from .config import get_config
from .embeddings import OllamaEmbedder
from .vector_store import VectorStore, SearchResult

_CHARS_PER_TOKEN = 4


class RAGPipeline:
    def __init__(self, config: dict = None):
        cfg = config or get_config()
        ollama = cfg["ollama"]
        vs = cfg["vector_store"]

        self._retrieval_cfg = cfg["retrieval"]
        self._rag_cfg = cfg["rag"]
        self._llm_model = ollama["llm_model"]
        self._base_url = ollama["base_url"].rstrip("/")
        self._timeout = ollama["timeout"]

        self._embedder = OllamaEmbedder(
            model=ollama["embedding_model"],
            base_url=ollama["base_url"],
            timeout=ollama["timeout"],
        )
        self._store = VectorStore(
            persist_directory=vs["persist_directory"],
            collection_name=vs["collection_name"],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = None) -> dict:
        """
        Run the full RAG pipeline for a question.

        Returns a dict with keys:
          question, answer, sources, chunks_used, retrieved_chunks
        """
        if self._store.count() == 0:
            return self._empty_store_response(question)

        top_k = top_k or self._retrieval_cfg["top_k"]
        q_emb = self._embedder.embed(question)
        results = self._store.query(q_emb, top_k=top_k)

        context_parts, sources = self._build_context(results)

        if not context_parts:
            return self._no_results_response(question)

        context = "\n\n---\n\n".join(context_parts)
        prompt = self._build_prompt(question, context)
        answer = self._generate(prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunks_used": len(context_parts),
            "retrieved_chunks": [
                {
                    "filename": r.filename,
                    "chunk_index": r.chunk_index,
                    "total_chunks": r.total_chunks,
                    "distance": round(r.distance, 4),
                    "preview": (r.text[:200] + "…") if len(r.text) > 200 else r.text,
                }
                for r in results[: len(context_parts)]
            ],
        }

    def stream_query(self, question: str, top_k: int = None) -> Iterator[str]:
        """
        Generator that yields answer tokens as they arrive from Ollama.
        Use for interactive CLI output.
        """
        if self._store.count() == 0:
            yield "Knowledge base is empty — run ingest.py first."
            return

        top_k = top_k or self._retrieval_cfg["top_k"]
        q_emb = self._embedder.embed(question)
        results = self._store.query(q_emb, top_k=top_k)
        context_parts, sources = self._build_context(results)

        if not context_parts:
            yield "I cannot find this information in the available OmniStudio documentation."
            return

        context = "\n\n---\n\n".join(context_parts)
        prompt = self._build_prompt(question, context)

        yield from self._stream_generate(prompt)
        yield f"\n\nSources: {', '.join(sources)}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context(self, results: list) -> tuple:
        """
        Pack retrieved chunks into context up to max_context_tokens.
        Returns (context_parts, deduplicated_source_list).
        """
        max_chars = self._rag_cfg["max_context_tokens"] * _CHARS_PER_TOKEN
        parts, sources, total_chars = [], [], 0

        for r in results:
            block = f"[Source: {r.filename}]\n{r.text}"
            if total_chars + len(block) > max_chars:
                break
            parts.append(block)
            if r.filename not in sources:
                sources.append(r.filename)
            total_chars += len(block)

        return parts, sources

    def _build_prompt(self, question: str, context: str) -> str:
        system = self._rag_cfg["system_prompt"].strip()
        return (
            f"{system}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            f"ANSWER:"
        )

    def _generate(self, prompt: str) -> str:
        r = requests.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self._rag_cfg["temperature"],
                    "num_ctx": 8192,
                },
            },
            timeout=self._timeout,
        )
        r.raise_for_status()
        return r.json()["response"].strip()

    def _stream_generate(self, prompt: str) -> Iterator[str]:
        with requests.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._llm_model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": self._rag_cfg["temperature"],
                    "num_ctx": 8192,
                },
            },
            stream=True,
            timeout=self._timeout,
        ) as resp:
            resp.raise_for_status()
            import json
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break

    # ------------------------------------------------------------------
    # Edge-case responses
    # ------------------------------------------------------------------

    def _empty_store_response(self, question: str) -> dict:
        return {
            "question": question,
            "answer": "The knowledge base is empty. Please run ingest.py first.",
            "sources": [],
            "chunks_used": 0,
            "retrieved_chunks": [],
        }

    def _no_results_response(self, question: str) -> dict:
        return {
            "question": question,
            "answer": (
                "I cannot find this information in the available OmniStudio documentation."
            ),
            "sources": [],
            "chunks_used": 0,
            "retrieved_chunks": [],
        }
