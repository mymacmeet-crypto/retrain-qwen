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
"""

import argparse
import json
import sys
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

# ---------------------------------------------------------------------------
# App + CORS (OpenWebUI needs cross-origin access)
# ---------------------------------------------------------------------------

app = FastAPI(title="Salesforce OmniStudio RAG", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared pipeline instance (loads ChromaDB once at startup)
_pipeline: RAGPipeline | None = None

MODEL_ID = "salesforce-omni-rag"


@app.on_event("startup")
def load_pipeline():
    global _pipeline
    print("Loading RAG pipeline …", flush=True)
    _pipeline = RAGPipeline()
    cfg = get_config()
    print(f"  LLM     : {cfg['ollama']['llm_model']}")
    print(f"  Embeds  : {cfg['ollama']['embedding_model']}")
    print(f"  Chunks  : {_pipeline._store.count()} in ChromaDB")
    print("Ready.", flush=True)


# ---------------------------------------------------------------------------
# Request / response schemas (OpenAI subset)
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[Message]
    stream: bool = True
    top_k: int | None = None  # optional override; pass via OpenWebUI system param


# ---------------------------------------------------------------------------
# Helper: extract the latest user question from the messages list
# ---------------------------------------------------------------------------

def _extract_question(messages: list[Message]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content.strip()
    raise HTTPException(status_code=400, detail="No user message found in request.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_ID}


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
    top_k = req.top_k

    if req.stream:
        return StreamingResponse(
            _stream_sse(question, top_k),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming fallback
    result = _pipeline.query(question, top_k=top_k)
    answer = result["answer"]
    sources = result.get("sources", [])
    if sources:
        answer += f"\n\nSources: {', '.join(sources)}"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ---------------------------------------------------------------------------
# SSE streaming generator
# ---------------------------------------------------------------------------

async def _stream_sse(question: str, top_k: int | None) -> AsyncIterator[str]:
    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def _chunk(content: str, finish: str | None = None) -> str:
        payload = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content} if content else {},
                    "finish_reason": finish,
                }
            ],
        }
        return f"data: {json.dumps(payload)}\n\n"

    # Opening delta with role
    role_payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(role_payload)}\n\n"

    try:
        for token in _pipeline.stream_query(question, top_k=top_k):
            yield _chunk(token)
    except Exception as exc:
        yield _chunk(f"\n[Error: {exc}]", finish="stop")
        yield "data: [DONE]\n\n"
        return

    yield _chunk("", finish="stop")
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Salesforce RAG — OpenAI-compatible server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8001, help="Bind port (default: 8001)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    args = parser.parse_args()

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
