"""
MILESTONE 3 — Eval suite using RAGAS
This is the file that makes you stand out in interviews.
Run this to get concrete numbers: faithfulness, answer relevancy,
context precision, context recall.
Shows the improvement from v1 (basic) → v2 (HyDE + rerank).
"""
# RAGAS 0.4.3 compatibility patch for modern langchain-community
import sys
import types
try:
    import langchain_google_vertexai
    mock_vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
    mock_vertexai.ChatVertexAI = langchain_google_vertexai.ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = mock_vertexai
except ImportError:
    pass

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset
from dotenv import load_dotenv
import json, os

load_dotenv()


def _get_score(results, key: str) -> float:
    """Helper to extract average float score from RAGAS EvaluationResult or dictionaries."""
    try:
        val = results[key]
    except Exception:
        val = None

    if val is None:
        if isinstance(results, dict):
            val = results.get(key, 0.0)
        else:
            val = getattr(results, key, 0.0)

    if isinstance(val, (list, tuple)):
        valid_vals = [v for v in val if v is not None]
        if not valid_vals:
            return 0.0
        return sum(valid_vals) / len(valid_vals)

    return float(val) if val is not None else 0.0


# ── Eval dataset ────────────────────────────────────────────────────────────
# In a real project: generate 20-50 Q&A pairs from your actual documents.
# These are the "ground truth" answers you compare your RAG against.
# Replace with your own questions once you have real PDFs loaded.
EVAL_QUESTIONS = [
    {
        "question": "What is NF2 normalization?",
        "ground_truth": "NF2 (Non-First Normal Form) is an advanced database model that allows nested relations and complex attributes, going beyond the flat table structure of traditional relational databases.",
    },
    {
        "question": "What entities are in the auction system schema?",
        "ground_truth": "The auction system schema includes Member, Auction, Bid, and Rating entities, connected through foreign keys.",
    },
    {
        "question": "What is the car retailer schema about?",
        "ground_truth": "The car retailer schema models car models and their attributes including nested relationships for interested customers.",
    },
    {
        "question": "What are structured user-defined types in SQL:2003?",
        "ground_truth": "Structured UDTs in SQL:2003 allow defining row types for complex objects like bids and ratings, enabling object-relational features in SQL.",
    },
    {
        "question": "How are foreign keys used in the auction schema?",
        "ground_truth": "Foreign keys connect the Member, Auction, Bid and Rating entities to establish relationships between them in the auction system.",
    },
]


def run_chain_on_questions(chain, retriever, questions: list[dict]) -> Dataset:
    """Run the RAG chain on all eval questions and collect answers + contexts."""
    rows = []
    for item in questions:
        q      = item["question"]
        answer = chain.invoke(q)

        # Retrieve the contexts used (for context metrics)
        try:
            ctx_docs = retriever.invoke(q)
            contexts = [d.page_content for d in ctx_docs]
        except Exception:
            contexts = []

        rows.append({
            "question":    q,
            "answer":      answer,
            "contexts":    contexts,
            "ground_truth": item["ground_truth"],
        })
    return Dataset.from_list(rows)


def evaluate_pipeline(chain, retriever, questions=None, label="pipeline"):
    """
    Run RAGAS evaluation and print a results table.
    Four metrics explained:
      - faithfulness:       does the answer stick to what the chunks say? (no hallucination)
      - answer_relevancy:   is the answer actually answering the question?
      - context_precision:  are the retrieved chunks actually useful?
      - context_recall:     did we retrieve all the chunks needed to answer?
    """
    if questions is None:
        questions = EVAL_QUESTIONS

    print(f"\nEvaluating [{label}] on {len(questions)} questions...")

    dataset = run_chain_on_questions(chain, retriever, questions)

    llm_wrapper  = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    emb_wrapper  = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm_wrapper,
        embeddings=emb_wrapper,
    )

    print(f"\n{'='*50}")
    print(f"  RAGAS Results - {label}")
    print(f"{'='*50}")
    print(f"  Faithfulness       : {_get_score(results, 'faithfulness'):.3f}  (no hallucination score)")
    print(f"  Answer relevancy   : {_get_score(results, 'answer_relevancy'):.3f}  (answers the question?)")
    print(f"  Context precision  : {_get_score(results, 'context_precision'):.3f}  (useful chunks?)")
    print(f"  Context recall     : {_get_score(results, 'context_recall'):.3f}  (all needed chunks found?)")
    print(f"{'='*50}\n")

    return results


def compare_pipelines(chain_v1, retriever_v1, chain_v2, retriever_v2, questions=None):
    """
    Compare v1 (basic) vs v2 (HyDE + rerank).
    Prints the improvement table you show in your portfolio and interviews.
    """
    if questions is None:
        questions = EVAL_QUESTIONS

    r1 = evaluate_pipeline(chain_v1, retriever_v1, questions, label="v1 basic")
    r2 = evaluate_pipeline(chain_v2, retriever_v2, questions, label="v2 HyDE+rerank")

    print("\n  IMPROVEMENT (v1 -> v2)")
    print(f"  {'='*44}")
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        score1 = _get_score(r1, metric)
        score2 = _get_score(r2, metric)
        delta = score2 - score1
        arrow = "(+)" if delta > 0 else "(-)"
        print(f"  {metric:<22}: {score1:.3f} -> {score2:.3f}  {arrow} {abs(delta):.3f}")
    print(f"  {'='*44}\n")

    return r1, r2


if __name__ == "__main__":
    print("Import and call compare_pipelines() from main.py after building both chains.")
