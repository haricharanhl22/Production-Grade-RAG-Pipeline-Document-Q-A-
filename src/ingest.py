"""
MILESTONE 1 — Ingestion pipeline
Loads PDFs → chunks them → embeds → stores in Qdrant (local, in-memory)
"""

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
import os

load_dotenv()

COLLECTION_NAME = "rag_docs"
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 64


def load_documents(data_dir: str = "data"):
    """Load all PDFs from the data/ folder."""
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from {data_dir}/")
    return docs


def chunk_documents(docs):
    """
    RecursiveCharacterTextSplitter tries to split on paragraphs,
    then sentences, then words — respects natural text boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def build_vectorstore(chunks):
    """Embed chunks and store in Qdrant (local in-memory for dev)."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Local in-memory Qdrant — no Docker needed for development
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    vectorstore.add_documents(chunks)
    print(f"Stored {len(chunks)} chunks in Qdrant collection '{COLLECTION_NAME}'")
    return vectorstore


def run_ingestion(data_dir: str = "data"):
    docs   = load_documents(data_dir)
    chunks = chunk_documents(docs)
    vs     = build_vectorstore(chunks)
    return vs


if __name__ == "__main__":
    run_ingestion()
