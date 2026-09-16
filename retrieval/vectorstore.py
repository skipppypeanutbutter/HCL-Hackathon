"""
Semantic retrieval using Supabase Postgres and pgvector.

The retriever:

1. Embeds the user's query with all-MiniLM-L6-v2.
2. Compares it against document_chunks.embedding.
3. Uses pgvector cosine distance: embedding <=> query_vector.
4. Returns the top-k nearest chunks as LangChain Document objects.

The embedding model used here must match the model used during ingestion.
all-MiniLM-L6-v2 produces 384-dimensional embeddings.
"""

import os
import threading
from typing import List

from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from rag.db import get_pool


EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

DEFAULT_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    """
    Load the embedding model once per Python process.

    The lock prevents concurrent requests from attempting to initialise
    the SentenceTransformer model simultaneously.
    """

    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(EMBEDDING_MODEL)

    return _model


def _embed_query(query: str) -> list[float]:
    """
    Convert a query into an embedding.

    Encoding is serialised because concurrent SentenceTransformer calls
    can be unstable with the Apple Silicon MPS backend.
    """

    if not query or not query.strip():
        raise ValueError("Retrieval query must not be empty.")

    model = _get_model()

    with _model_lock:
        embedding = model.encode(
            query.strip(),
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    return embedding.tolist()


def _to_pgvector_literal(embedding: list[float]) -> str:
    """
    Convert a Python embedding into pgvector text format.

    Example:
        [0.12, -0.34, 0.56]

    The generated value is passed as a SQL parameter and explicitly
    cast with ::vector.
    """

    return "[" + ",".join(str(float(value)) for value in embedding) + "]"


class SupabaseRetriever:
    """
    Minimal LangChain-compatible retriever backed by Supabase pgvector.

    Calling invoke(query) returns a list of LangChain Document objects
    ordered from most relevant to least relevant.
    """

    def __init__(self, k: int = DEFAULT_TOP_K):
        if k < 1:
            raise ValueError("k must be at least 1.")

        self.k = k

    def invoke(self, query: str) -> List[Document]:
        query_embedding = _embed_query(query)
        embedding_literal = _to_pgvector_literal(query_embedding)

        sql = """
            WITH query_vector AS (
                SELECT %s::vector AS embedding
            )
            SELECT
                dc.chunk_text,
                dc.chunk_index,
                d.title,
                d.doc_type,
                d.source_path,
                dc.embedding <=> q.embedding AS distance,
                1 - (dc.embedding <=> q.embedding) AS similarity
            FROM document_chunks dc
            JOIN documents d
                ON d.id = dc.document_id
            CROSS JOIN query_vector q
            WHERE dc.embedding IS NOT NULL
            ORDER BY distance ASC
            LIMIT %s
        """

        pool = get_pool()
        connection = pool.getconn()

        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    sql,
                    (embedding_literal, self.k),
                )
                rows = cursor.fetchall()

        except Exception:
            # Prevent an aborted transaction from being returned to the pool.
            connection.rollback()
            raise

        finally:
            # SELECT starts a transaction in psycopg2 by default.
            # Rolling it back safely closes that read-only transaction.
            connection.rollback()
            pool.putconn(connection)

        documents: list[Document] = []

        for (
            chunk_text,
            chunk_index,
            title,
            doc_type,
            source_path,
            distance,
            similarity,
        ) in rows:
            documents.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "source": source_path or title or "unknown",
                        "doc_type": doc_type or "unknown",
                        "title": title or "unknown",
                        "chunk_index": chunk_index,
                        "distance": float(distance),
                        "similarity": float(similarity),
                    },
                )
            )

        return documents


def get_retriever(k: int = DEFAULT_TOP_K) -> SupabaseRetriever:
    """Create a Supabase pgvector retriever."""

    return SupabaseRetriever(k=k)


def build_or_load_vector_store() -> int:
    """
    Health check used during application startup.

    Confirms that Supabase is reachable and returns the number of
    document chunks containing embeddings.
    """

    sql = """
        SELECT COUNT(*)
        FROM document_chunks
        WHERE embedding IS NOT NULL
    """

    pool = get_pool()
    connection = pool.getconn()

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql)
            result = cursor.fetchone()
            return int(result[0])

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.rollback()
        pool.putconn(connection)