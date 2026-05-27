"""
main.py — Run the full RAG agent pipeline
Milestones 1 → 4 in sequence, or pick one via --mode flag.

Usage:
  python main.py --mode ingest              # ingest PDFs only
  python main.py --mode chat                # chat with v2 (production)
  python main.py --mode eval                # run RAGAS eval, compare v1 vs v2
  python main.py --mode all                 # run everything end to end
"""

import argparse
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# ── Sanity check API keys ────────────────────────────────────────────────────
def check_env():
    missing = [k for k in ["OPENAI_API_KEY"] if not os.getenv(k)]
    if missing:
        print(f"ERROR: Missing env vars: {', '.join(missing)}")
        print("Copy .env.example → .env and fill in your keys.")
        sys.exit(1)
    warnings = [k for k in ["COHERE_API_KEY", "LANGCHAIN_API_KEY"] if not os.getenv(k)]
    if warnings:
        print(f"WARNING: {', '.join(warnings)} not set. Reranking / tracing will be skipped.\n")


# ── Mode: ingest ─────────────────────────────────────────────────────────────
def run_ingest():
    from src.ingest import run_ingestion
    print("\n[MILESTONE 1] Ingesting documents from data/...")
    vs = run_ingestion(data_dir="data")
    print("Ingestion complete.\n")
    return vs


# ── Mode: chat (v2 production chain) ─────────────────────────────────────────
def run_chat(vectorstore):
    from src.query_v2 import build_production_rag_chain
    print("\n[MILESTONE 2] Building production chain (HyDE + rerank)...")
    chain, retriever = build_production_rag_chain(vectorstore)
    print("Ready. Type your question (or 'quit' to exit).\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        print("\nAssistant:", chain.invoke(q), "\n")


# ── Mode: eval ───────────────────────────────────────────────────────────────
def run_eval(vectorstore):
    from src.query import build_rag_chain
    from src.query_v2 import build_production_rag_chain
    from src.eval import compare_pipelines

    print("\n[MILESTONE 3] Running RAGAS evaluation — this takes a few minutes...")

    chain_v1, ret_v1 = build_rag_chain(vectorstore)
    chain_v2, ret_v2 = build_production_rag_chain(vectorstore)

    compare_pipelines(chain_v1, ret_v1, chain_v2, ret_v2)


# ── Mode: all ────────────────────────────────────────────────────────────────
def run_all():
    vs = run_ingest()
    run_eval(vs)
    run_chat(vs)


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    check_env()

    parser = argparse.ArgumentParser(description="RAG Agent")
    parser.add_argument(
        "--mode",
        choices=["ingest", "chat", "eval", "all"],
        default="chat",
        help="Which mode to run (default: chat)",
    )
    args = parser.parse_args()

    if args.mode == "ingest":
        run_ingest()
    elif args.mode == "chat":
        vs = run_ingest()
        run_chat(vs)
    elif args.mode == "eval":
        vs = run_ingest()
        run_eval(vs)
    elif args.mode == "all":
        run_all()
