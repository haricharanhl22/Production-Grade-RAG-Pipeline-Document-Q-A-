# RAG Pipeline — Document Q&A with HyDE, Reranking, and RAGAS Eval

An end-to-end Retrieval-Augmented Generation pipeline for document question
answering. The project compares a baseline retriever against an improved
version (HyDE query expansion + Cohere reranking) and quantifies the
difference using RAGAS metrics.

The point of the project was to move past tutorial-style RAG — where you embed
chunks, retrieve top-k, and trust the output — and actually measure what helps.

## What it does

- Ingests PDFs, chunks them with a recursive splitter, embeds with
  `text-embedding-3-small`, and stores in Qdrant.
- Answers questions over the corpus with citations back to the source page.
- Runs a RAGAS evaluation comparing `v1` (basic retrieval) against `v2`
  (HyDE expansion + Cohere rerank).
- Logs every retrieval and generation step to LangSmith.

## Stack

- **LangChain** — orchestration
- **OpenAI** `text-embedding-3-small` for embeddings, `gpt-4o-mini` for generation
- **Qdrant** — vector store (in-memory or Docker)
- **Cohere** `rerank-english-v3.0` — reranking
- **RAGAS** — evaluation
- **LangSmith** — tracing

## Pipeline

```
Ingestion
  PDFs → recursive chunking → embedding → Qdrant

Query (v2)
  Question → multi-query / HyDE expansion
          → retrieve top-8 from Qdrant
          → Cohere rerank → top-3
          → gpt-4o-mini with citation prompt
          → answer + [source: file, page X]

Evaluation
  RAGAS metrics: faithfulness, answer_relevancy, context_precision
```

## Eval results

On the sample document set:

```
v1 (basic)            v2 (HyDE + rerank)
Faithfulness         0.71    →    0.89   (+0.18)
Answer relevancy     0.74    →    0.91   (+0.17)
Context precision    0.68    →    0.86   (+0.18)
```

Most of the lift comes from reranking — HyDE on its own helped less than
expected on this corpus, probably because the questions were already
specific enough that hypothetical answers didn't add much retrieval signal.
The reranker, by contrast, consistently pushed relevant chunks to the top.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add OPENAI_API_KEY (required)
# Add COHERE_API_KEY for reranking (free tier at cohere.com)
# Add LANGCHAIN_API_KEY for tracing (free at smith.langchain.com)

cp your_documents.pdf data/
```

## Usage

```bash
python main.py --mode chat       # ask questions interactively
python main.py --mode eval       # run RAGAS evaluation
python main.py --mode all        # ingest + eval + chat
```

## Layout

```
.
├── main.py              # entry point
├── src/
│   ├── ingest.py        # load, chunk, embed, store
│   ├── query.py         # baseline retrieval + LLM
│   ├── query_v2.py      # HyDE + reranker
│   └── eval.py          # RAGAS evaluation
├── data/                # PDFs go here
├── evals/               # eval datasets and saved results
└── .env.example
```

## Notes

This was built as a learning project for retrieval engineering — the
interesting parts are in `query_v2.py` and `eval.py`. The pipeline isn't
production-deployed; it's a controlled comparison meant to make the eval
numbers reproducible.
