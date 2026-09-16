import os
import json
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from google import genai

load_dotenv()

# --- Supabase & Gemini Setup ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Load MiniLM model (384 dimensions - matching ingestion.py)
print("Loading MiniLM embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

def rewrite_query(original_query: str) -> list[str]:
    """Uses Gemini to generate 3 alternative query variations."""
    prompt = f"""You are an AI assistant specialized in optimizing search queries for vector databases.
Generate 3 alternative versions of the user's query to capture different phrasings, synonyms, or related concepts.

Original Query: {original_query}"""

    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
        )
        lines = response.text.strip().split("\n")
        rewrites = [line.strip().lstrip("0123456789.- ") for line in lines if line.strip()]
        return rewrites[:3]
    except Exception as e:
        print(f"Warning: Failed to rewrite query with Gemini: {e}")
        return []

def retrieve_chunks(query: str, top_k: int = 4) -> list[dict]:
    """Retrieves top_k document chunks from Supabase for a single query string."""
    query_vector = embedding_model.encode(query, show_progress_bar=False).tolist()

    # Option A: Try Supabase RPC match_document_chunks if available
    try:
        rpc_res = supabase.rpc(
            "match_document_chunks",
            {
                "query_embedding": query_vector,
                "match_threshold": 0.0,
                "match_count": top_k
            }
        ).execute()
        if rpc_res.data:
            return rpc_res.data
    except Exception:
        pass  # Fall back to client-side vector search if RPC function does not exist

    # Option B: Client-side vector similarity fallback
    res = supabase.table("document_chunks").select("id, document_id, chunk_index, chunk_text, embedding").execute()
    if not res.data:
        return []

    q_vec = np.array(query_vector)
    scored_chunks = []
    for chunk in res.data:
        emb = chunk.get("embedding")
        if emb:
            c_vec = np.array(emb if isinstance(emb, list) else json.loads(emb))
            # Cosine similarity (MiniLM embeddings are L2 normalized)
            sim = float(np.dot(q_vec, c_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(c_vec) + 1e-10))
            chunk_copy = dict(chunk)
            chunk_copy["similarity"] = sim
            scored_chunks.append(chunk_copy)

    scored_chunks.sort(key=lambda x: x["similarity"], reverse=True)
    return scored_chunks[:top_k]


def evaluate_accuracy_with_llm(user_query: str, retrieved_chunks: list[dict]) -> tuple[float, str]:
    """Uses Gemini LLM-as-a-judge to grade how well retrieved chunks match user query intent (0-100%)."""
    if not retrieved_chunks:
        return 0.0, "No chunks retrieved."

    context = "\n\n".join([f"Chunk {idx+1}: {c['chunk_text']}" for idx, c in enumerate(retrieved_chunks)])
    prompt = f"""You are an expert evaluator assessing the retrieval quality of a vector database RAG system.

User Query: "{user_query}"

Retrieved Context Chunks:
{context}

Based on how relevant, comprehensive, and accurate these retrieved chunks are for answering the user's query, output a single score from 0 to 100 representing the retrieval accuracy/relevance percentage.
Respond ONLY with a valid JSON object in this exact format:
{{"score": 85, "reasoning": "Brief 1-sentence explanation"}}"""

    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
        )
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text.strip())
        return float(data.get("score", 0.0)), data.get("reasoning", "")
    except Exception as e:
        mean_sim = np.mean([c.get("similarity", 0) for c in retrieved_chunks]) * 100
        return round(float(mean_sim), 2), f"Calculated from average vector similarity ({mean_sim:.1f}%)"


def run_comparison(user_query: str, top_k: int = 4):
    print("\n" + "="*70)
    print(f"  ORIGINAL QUERY: \"{user_query}\"")
    print("="*70)

    # 1. Query Rewriting
    print("\n[Step 1] Rewriting query using Gemini (gemini-3.5-flash-lite)...")
    rewritten_queries = rewrite_query(user_query)
    print("  Alternative Queries Generated:")
    for idx, q in enumerate(rewritten_queries, 1):
        print(f"    {idx}. \"{q}\"")

    # 2. Before Rewriting Retrieval
    print("\n[Step 2] Retrieving documents BEFORE Query Rewriting (Original Query Only)...")
    before_chunks = retrieve_chunks(user_query, top_k=top_k)
    
    print(f"\n  --- BEFORE REWRITING RESULTS (Top {len(before_chunks)}) ---")
    for idx, c in enumerate(before_chunks, 1):
        sim = c.get("similarity", 0)
        snippet = c["chunk_text"].replace("\n", " ")[:120]
        print(f"  [{idx}] (Similarity: {sim:.4f}) | Chunk #{c.get('chunk_index', '?')}")
        print(f"      \"{snippet}...\"")

    # 3. After Rewriting Retrieval
    print("\n[Step 3] Retrieving documents AFTER Query Rewriting (Original + 3 Variations)...")
    all_queries = [user_query] + rewritten_queries
    after_chunk_dict = {}

    for q in all_queries:
        chunks = retrieve_chunks(q, top_k=top_k)
        for c in chunks:
            c_id = c["id"]
            if c_id not in after_chunk_dict or c.get("similarity", 0) > after_chunk_dict[c_id].get("similarity", 0):
                after_chunk_dict[c_id] = c

    # Sort combined retrieved chunks by similarity score
    after_chunks = sorted(after_chunk_dict.values(), key=lambda x: x.get("similarity", 0), reverse=True)[:top_k]

    print(f"\n  --- AFTER REWRITING RESULTS (Top {len(after_chunks)}) ---")
    for idx, c in enumerate(after_chunks, 1):
        sim = c.get("similarity", 0)
        snippet = c["chunk_text"].replace("\n", " ")[:120]
        print(f"  [{idx}] (Similarity: {sim:.4f}) | Chunk #{c.get('chunk_index', '?')}")
        print(f"      \"{snippet}...\"")

    # 4. Accuracy & Relevance Evaluation
    print("\n[Step 4] Evaluating Retrieval Accuracy with LLM-as-a-Judge...")
    before_score, before_reason = evaluate_accuracy_with_llm(user_query, before_chunks)
    after_score, after_reason = evaluate_accuracy_with_llm(user_query, after_chunks)

    print("\n" + "="*70)
    print("                      ACCURACY & RETRIEVAL COMPARISON SUMMARY")
    print("="*70)
    print(f"  BEFORE REWRITING ACCURACY : {before_score:.1f}%")
    print(f"    Reasoning: {before_reason}")
    print(f"  AFTER REWRITING ACCURACY  : {after_score:.1f}%")
    print(f"    Reasoning: {after_reason}")

    diff = after_score - before_score
    if diff > 0:
        print(f"\n  -> Query Rewriting IMPROVED retrieval accuracy by +{diff:.1f}%! 🚀")
    elif diff < 0:
        print(f"\n  -> Query Rewriting decreased accuracy by {diff:.1f}%.")
    else:
        print(f"\n  -> Retrieval accuracy remained identical ({before_score:.1f}%).")
    print("="*70 + "\n")


def main():
    print("="*70)
    print("  Supabase Vector DB Query Rewriting Test Harness")
    print("="*70)

    while True:
        try:
            user_prompt = input("\nEnter prompt to test (or 'exit' to quit): ").strip()
            if not user_prompt:
                continue
            if user_prompt.lower() in ("exit", "quit", "q"):
                print("Exiting test harness. Goodbye!")
                break
            
            run_comparison(user_prompt)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
