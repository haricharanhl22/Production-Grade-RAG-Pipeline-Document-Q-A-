"""
MILESTONE 1 — Basic query pipeline
Takes a question → retrieves top-k chunks → sends to LLM → returns cited answer
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY
the context below. If the answer is not in the context, say "I don't have enough
information to answer that."

For every fact you state, add a citation like [Source: page X] using the
metadata from the context.

Context:
{context}

Question: {question}

Answer:"""


def format_docs(docs):
    """Format retrieved docs with source metadata for the prompt."""
    parts = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source: {source}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_rag_chain(vectorstore, k: int = 4):
    """
    Build a simple RAG chain.
    k = number of chunks to retrieve per query.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm       = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt    = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def ask(chain, question: str) -> str:
    """Ask a question and return the answer."""
    answer = chain.invoke(question)
    return answer


if __name__ == "__main__":
    # Quick smoke test — wire up with a real vectorstore in main.py
    print("Import this module and call build_rag_chain(vectorstore)")
