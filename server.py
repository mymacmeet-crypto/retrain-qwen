#!/usr/bin/env python3
"""
OpenAI-compatible FastAPI server wrapping the Salesforce OmniStudio RAG pipeline.

Usage:
    pip install fastapi uvicorn[standard]
    python server.py                  # default: http://localhost:8001
    python server.py --port 8002

Add to OpenWebUI:
    Admin → Connections → Add OpenAI connection
    URL : http://localhost:8001
    Key : any non-empty string (e.g. "rag")

ChromaDB hot-swap (no restart needed):
    curl -X POST http://localhost:8001/admin/switch-db \
         -H "Content-Type: application/json" \
         -d '{"path": "/Volumes/ExternalSSD/chroma_apex"}'

ChromaDB status:
    curl http://localhost:8001/admin/db-status
"""

import argparse
import json
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline
from src.config import get_config
from src.vector_store import VectorStore

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="Salesforce OmniStudio RAG", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared pipeline + a lock so a swap mid-request doesn't corrupt anything
_pipeline: RAGPipeline | None = None
_pipeline_lock = threading.Lock()

MODEL_ID = "salesforce-omni-rag"


@app.on_event("startup")
def load_pipeline():
    global _pipeline
    print("Loading RAG pipeline …", flush=True)
    _pipeline = RAGPipeline()
    cfg = get_config()
    print(f"  LLM       : {cfg['ollama']['llm_model']}")
    print(f"  Embeds    : {cfg['ollama']['embedding_model']}")
    print(f"  ChromaDB  : {_pipeline._store._col.metadata.get('persist_directory', 'unknown')}")
    print(f"  Chunks    : {_pipeline._store.count():,}")
    print("Ready.", flush=True)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[Message]
    stream: bool = True
    top_k: int | None = None


class SwitchDbRequest(BaseModel):
    path: str          # absolute path to the ChromaDB directory
    collection: str = "salesforce_knowledge"   # collection name inside that DB


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _extract_question(messages: list[Message]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content.strip()
    raise HTTPException(status_code=400, detail="No user message found in request.")


# ---------------------------------------------------------------------------
# Standard routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    cfg = get_config()
    return {
        "status": "ok",
        "model":   MODEL_ID,
        "llm":     cfg["ollama"]["llm_model"],
        "chunks":  _pipeline._store.count() if _pipeline else 0,
    }


@app.get("/v1/models")
@app.get("/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(req: ChatRequest):
    question = _extract_question(req.messages)
    top_k    = req.top_k

    if req.stream:
        return StreamingResponse(
            _stream_sse(question, top_k),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    with _pipeline_lock:
        result = _pipeline.query(question, top_k=top_k)

    answer  = result["answer"]
    sources = result.get("sources", [])
    if sources:
        answer += f"\n\nSources: {', '.join(sources)}"

    return {
        "id":      f"chatcmpl-{uuid.uuid4().hex}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   MODEL_ID,
        "choices": [
            {
                "index":         0,
                "message":       {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------

async def _stream_sse(question: str, top_k: int | None) -> AsyncIterator[str]:
    cid     = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def _chunk(content: str, finish: str | None = None) -> str:
        payload = {
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0,
                         "delta": {"content": content} if content else {},
                         "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': MODEL_ID, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

    try:
        with _pipeline_lock:
            tokens = list(_pipeline.stream_query(question, top_k=top_k))
        for token in tokens:
            yield _chunk(token)
    except Exception as exc:
        yield _chunk(f"\n[Error: {exc}]", finish="stop")
        yield "data: [DONE]\n\n"
        return

    yield _chunk("", finish="stop")
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Admin routes — ChromaDB hot-swap
# ---------------------------------------------------------------------------

@app.get("/admin/db-status")
def db_status():
    """Show which ChromaDB is currently connected and how many chunks it has."""
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet.")
    cfg = get_config()
    return {
        "current_db_path":   cfg["vector_store"]["persist_directory"],
        "collection":        cfg["vector_store"]["collection_name"],
        "chunks":            _pipeline._store.count(),
        "llm_model":         cfg["ollama"]["llm_model"],
    }


@app.post("/admin/switch-db")
def switch_db(req: SwitchDbRequest):
    """
    Hot-swap to a different ChromaDB directory without restarting the server.

    The fine-tuned model stays loaded in Ollama — only the knowledge base changes.
    In-flight requests finish against the old DB; new requests use the new DB.

    Example:
        curl -X POST http://localhost:8001/admin/switch-db \\
             -H "Content-Type: application/json" \\
             -d '{"path": "/Volumes/ExternalSSD/chroma_apex"}'
    """
    global _pipeline

    db_path = Path(req.path)
    if not db_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Path does not exist: {db_path}\n"
                   "Make sure the external disk is mounted and the path is correct.",
        )

    # Verify it looks like a ChromaDB directory
    if not (db_path / "chroma.sqlite3").exists():
        raise HTTPException(
            status_code=400,
            detail=f"No ChromaDB found at {db_path}\n"
                   "Expected to find chroma.sqlite3 in that directory.",
        )

    old_path   = get_config()["vector_store"]["persist_directory"]
    old_chunks = _pipeline._store.count() if _pipeline else 0

    # Build new store and swap it into the pipeline under the lock
    new_store = VectorStore(
        persist_directory=str(db_path),
        collection_name=req.collection,
    )
    new_chunks = new_store.count()

    with _pipeline_lock:
        _pipeline._store = new_store

    print(f"[DB SWAP] {old_path} ({old_chunks} chunks) → {db_path} ({new_chunks} chunks)",
          flush=True)

    return {
        "status":      "switched",
        "previous_db": str(old_path),
        "previous_chunks": old_chunks,
        "new_db":      str(db_path),
        "new_chunks":  new_chunks,
        "collection":  req.collection,
    }


@app.get("/admin/list-dbs")
def list_dbs(root: str = "/Volumes"):
    """
    Scan a directory for valid ChromaDB folders.
    Helps you find which DBs are available on a mounted external disk.

    Example:
        curl "http://localhost:8001/admin/list-dbs?root=/Volumes/ExternalSSD"
    """
    root_path = Path(root)
    if not root_path.exists():
        return {"root": str(root_path), "databases": [], "note": "Path not found"}

    found = []
    for candidate in sorted(root_path.rglob("chroma.sqlite3")):
        db_dir = candidate.parent
        try:
            store  = VectorStore(persist_directory=str(db_dir),
                                 collection_name="salesforce_knowledge")
            chunks = store.count()
        except Exception:
            chunks = -1
        found.append({"path": str(db_dir), "chunks": chunks})

    return {"root": str(root_path), "databases": found}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Salesforce RAG — OpenAI-compatible server")
    parser.add_argument("--host",   default="0.0.0.0")
    parser.add_argument("--port",   type=int, default=8001)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run("server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
