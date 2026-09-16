"""
Minimal RAG pipeline: chunk -> embed -> store -> retrieve.

Uses sentence-transformers for embeddings (runs locally, no API key
needed) and a plain numpy array as the vector store. This is enough
for a hackathon-scale demo (hundreds to a few thousand chunks).
If you need more scale, swap VectorStore for FAISS or Chroma without
changing the rest of the app.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks


class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings: np.ndarray | None = None
        self.texts: list[str] = []
        self.metadata: list[dict] = []

    def add(self, texts: list[str], metadata: list[dict]):
        new_embeddings = self.model.encode(texts, normalize_embeddings=True)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        self.texts.extend(texts)
        self.metadata.extend(metadata)

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        if self.embeddings is None or len(self.texts) == 0:
            return []
        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_vec
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "text": self.texts[i],
                "score": float(scores[i]),
                "metadata": self.metadata[i],
            }
            for i in top_indices
        ]


def build_context(results: list[dict]) -> str:
    """Turn retrieved chunks into a single context block for the prompt."""
    return "\n\n".join(f"[{r['metadata'].get('source', 'unknown')}] {r['text']}" for r in results)
