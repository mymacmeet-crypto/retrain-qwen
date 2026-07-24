"""Recursive character-level text splitter with token-aware sizing."""

from dataclasses import dataclass
from typing import Optional

from .document_loader import Document


@dataclass
class Chunk:
    text: str
    source: str
    filename: str
    chunk_index: int
    total_chunks: int  # filled in after splitting; 0 until finalised
    char_start: int
    char_end: int
    file_type: str


# ~4 characters per token is a reliable approximation for English prose.
_CHARS_PER_TOKEN = 4

# Ordered split points: try widest natural boundary first.
_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


def _tok(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


class TextSplitter:
    """
    Splits a Document into overlapping Chunks whose token count stays
    within [min_chunk_size, chunk_size].  Overlap is achieved by
    repeating the tail of the previous chunk at the head of the next.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        min_chunk_size: int = 50,
    ):
        self._max_chars = chunk_size * _CHARS_PER_TOKEN
        self._overlap_chars = chunk_overlap * _CHARS_PER_TOKEN
        self._min_chars = min_chunk_size * _CHARS_PER_TOKEN

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self, doc: Document) -> list:
        raw = self._split_text(doc.content, _SEPARATORS)
        raw = [c.strip() for c in raw if len(c.strip()) >= self._min_chars]

        total = len(raw)
        chunks = []
        search_from = 0

        for i, text in enumerate(raw):
            # Best-effort character position for metadata
            pos = doc.content.find(text[:60], search_from)
            if pos == -1:
                pos = search_from
            end = pos + len(text)
            search_from = max(search_from, end - self._overlap_chars)

            chunks.append(Chunk(
                text=text,
                source=doc.source,
                filename=doc.filename,
                chunk_index=i,
                total_chunks=total,
                char_start=pos,
                char_end=end,
                file_type=doc.file_type,
            ))

        return chunks

    # ------------------------------------------------------------------
    # Internal splitting logic
    # ------------------------------------------------------------------

    def _split_text(self, text: str, separators: list) -> list:
        """Recursively split on progressively finer separators."""
        if len(text) <= self._max_chars:
            return [text]

        for sep in separators:
            parts = text.split(sep)
            if len(parts) == 1:
                continue  # separator not found; try next

            merged = self._merge_splits(parts, sep)
            if len(merged) > 1:
                result = []
                for chunk in merged:
                    if len(chunk) > self._max_chars and separators:
                        # Recurse with finer separators
                        next_seps = separators[separators.index(sep) + 1:]
                        result.extend(self._split_text(chunk, next_seps))
                    else:
                        result.append(chunk)
                return self._add_overlap(result)

        # Hard character-level fallback
        return self._hard_split(text)

    def _merge_splits(self, parts: list, sep: str) -> list:
        """Greedily merge small parts into chunks ≤ max_chars."""
        chunks = []
        current = ""
        for part in parts:
            joined = (current + sep + part) if current else part
            if len(joined) <= self._max_chars:
                current = joined
            else:
                if current:
                    chunks.append(current)
                current = part
        if current:
            chunks.append(current)
        return chunks

    def _add_overlap(self, chunks: list) -> list:
        """Prepend the tail of chunk[i-1] to chunk[i]."""
        if self._overlap_chars == 0 or len(chunks) <= 1:
            return chunks
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-self._overlap_chars:]
            result.append(tail + "\n" + chunks[i])
        return result

    def _hard_split(self, text: str) -> list:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self._max_chars
            chunks.append(text[start:end])
            start = end - self._overlap_chars
        return chunks
