#!/usr/bin/env python3
"""
Interactive RAG query interface for Salesforce OmniStudio knowledge.

Usage:
    python query.py                          # interactive REPL
    python query.py "How do I create an OmniScript?"
    python query.py --top-k 8 "What is a FlexCard?"
    python query.py --no-stream "Explain Integration Procedures"
    python query.py --verbose "How to debug DataRaptor?"
"""

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config
from src.rag_pipeline import RAGPipeline


_BANNER = """\
╔══════════════════════════════════════════════════════════╗
║   Salesforce OmniStudio RAG  ·  Powered by Qwen 2.5 7B  ║
║   Type 'exit' or Ctrl-C to quit  ·  '/help' for hints   ║
╚══════════════════════════════════════════════════════════╝"""

_HELP = """\
Commands:
  /help          — show this help
  /top <n>       — change top-k retrieval count  (e.g. /top 8)
  /verbose       — toggle verbose chunk display
  /stream        — toggle streaming output
  exit / quit    — exit the REPL

Tips:
  • Ask about OmniStudio components: OmniScript, FlexCard,
    DataRaptor, Integration Procedure, OmniOut, etc.
  • Be specific: "How do I add a step to an OmniScript?" works
    better than "explain OmniStudio"."""


def run_single_query(pipeline: RAGPipeline, question: str,
                     top_k: int, stream: bool, verbose: bool) -> None:
    print(f"\nQ: {question}\n")
    print("─" * 60)

    if stream:
        print("A: ", end="", flush=True)
        for token in pipeline.stream_query(question, top_k=top_k):
            print(token, end="", flush=True)
        print()
    else:
        result = pipeline.query(question, top_k=top_k)
        _print_result(result, verbose)


def run_repl(pipeline: RAGPipeline, top_k: int, stream: bool, verbose: bool) -> None:
    print(_BANNER)
    cfg = get_config()
    print(f"\nModel  : {cfg['ollama']['llm_model']}")
    print(f"Embeds : {cfg['ollama']['embedding_model']}")
    print(f"Top-k  : {top_k}  |  Stream: {stream}  |  Verbose: {verbose}\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        if question == "/help":
            print(_HELP)
            continue
        if question.startswith("/top "):
            try:
                top_k = int(question.split()[1])
                print(f"Top-k set to {top_k}")
            except (ValueError, IndexError):
                print("Usage: /top <number>")
            continue
        if question == "/verbose":
            verbose = not verbose
            print(f"Verbose {'on' if verbose else 'off'}")
            continue
        if question == "/stream":
            stream = not stream
            print(f"Streaming {'on' if stream else 'off'}")
            continue

        print()
        run_single_query(pipeline, question, top_k, stream, verbose)
        print()


def _print_result(result: dict, verbose: bool) -> None:
    answer = result["answer"]
    # Wrap long lines for readability
    for para in answer.split("\n"):
        if len(para) > 100:
            print(textwrap.fill(para, width=100))
        else:
            print(para)

    print(f"\nSources ({result['chunks_used']} chunk(s) used):")
    for src in result["sources"]:
        print(f"  • {src}")

    if verbose and result.get("retrieved_chunks"):
        print("\n─── Retrieved chunks ───")
        for c in result["retrieved_chunks"]:
            print(f"  [{c['chunk_index']}/{c['total_chunks']}] {c['filename']} "
                  f"(distance={c['distance']})")
            print(f"    {c['preview']}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Query the Salesforce OmniStudio RAG knowledge base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("question", nargs="?", help="Question to ask (omit for interactive mode)")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--verbose", action="store_true", help="Show retrieved chunk previews")
    args = parser.parse_args()

    cfg = get_config()
    top_k = args.top_k or cfg["retrieval"]["top_k"]
    stream = not args.no_stream

    pipeline = RAGPipeline()

    if args.question:
        run_single_query(pipeline, args.question, top_k, stream, args.verbose)
    else:
        run_repl(pipeline, top_k, stream, args.verbose)


if __name__ == "__main__":
    main()
