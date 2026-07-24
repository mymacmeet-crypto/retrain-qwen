"""Ollama-backed embedding client."""

import requests
from typing import Optional


class OllamaEmbedder:
    """
    Wraps Ollama's /api/embed endpoint to produce dense embeddings.

    Ollama /api/embed accepts a list under "input" and returns
    {"embeddings": [[float, ...], ...]}.  Falls back to the legacy
    /api/embeddings endpoint (single "prompt" → "embedding") when the
    newer endpoint is unavailable.
    """

    def __init__(self, model: str, base_url: str, timeout: int = 120):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._dim: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list:
        """Return a single embedding vector for text."""
        return self._embed_batch_api([text])[0]

    def embed_batch(self, texts: list, batch_size: int = 8) -> list:
        """Return embeddings for a list of texts, in batches."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(self._embed_batch_api(batch))
        return all_embeddings

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed("dimension probe"))
        return self._dim

    def check_available(self) -> bool:
        """Return True if the embedding model is listed in Ollama."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=10)
            r.raise_for_status()
            names = [m["name"] for m in r.json().get("models", [])]
            return any(self.model.split(":")[0] in n for n in names)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_batch_api(self, texts: list) -> list:
        """
        Try the modern /api/embed endpoint first; fall back to legacy
        /api/embeddings if it returns a 404.
        """
        try:
            return self._call_embed(texts)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                # Older Ollama build — use legacy single-item endpoint
                return [self._call_embeddings_legacy(t) for t in texts]
            raise

    def _call_embed(self, texts: list) -> list:
        r = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise ValueError(f"Empty embeddings response: {data}")
        return embeddings

    def _call_embeddings_legacy(self, text: str) -> list:
        r = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["embedding"]
