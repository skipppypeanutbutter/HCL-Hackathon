import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from google import genai

load_dotenv()

# --- Setup Clients ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY) must be set in .env")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY must be set in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Load Embedding Model
print("Loading MiniLM embedding model...")
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
except Exception:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def call_gemini(prompt: str) -> str:
    """Helper function to call Gemini with automatic fallback across models."""
    models = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
    last_err = None
    for model_name in models:
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Gemini API call failed: {last_err}")


def rewrite_query_with_feedback(original_query: str, last_query: str = None, feedback: str = None) -> list[str]:
    """Generates 3 targeted search query variations, taking into account past judge feedback if available."""
    if feedback:
        prompt = f"""You are an expert search optimizer for a vector retrieval system.
The previous search attempt failed to retrieve sufficient context for the user's question.

Original Question: {original_query}
Previous Search Query Used: {last_query}
Evaluator Feedback on Missing Details: {feedback}

Generate 3 NEW, alternative search queries specifically targeted at addressing the missing details.
Rules:
- Focus on specific missing keywords, entities, or legal/financial policy codes.
- Keep variations short, concise, and keyword-dense.
- Output EXACTLY 3 lines, one query per line, without numbers, bullets, or commentary."""
    else:
        prompt = f"""You are an expert search optimizer for vector retrieval systems.
Generate 3 alternative search queries for the user's input.
Rules:
- Keep variations short, concise, and focused on core keywords.
- Preserve exact technical terms, acronyms, or proper nouns.
- Output EXACTLY 3 lines, one query per line, without numbers, bullet points, or commentary.

Original Query: {original_query}"""

    try:
        raw_text = call_gemini(prompt)
        lines = raw_text.strip().split("\n")
        rewrites = [line.strip().lstrip("0123456789.- ") for line in lines if line.strip()]
        return rewrites[:3]
    except Exception as e:
        print(f"Warning: Query rewrite failed: {e}")
        return []


def retrieve_chunks(query: str, top_k: int = 4) -> list[dict]:
    """Retrieves top_k document chunks from Supabase vector storage."""
    query_vector = embedding_model.encode(query, show_progress_bar=False).tolist()

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
        pass

    res = supabase.table("document_chunks").select("id, document_id, chunk_index, chunk_text, embedding").execute()
    if not res.data:
        return []

    q_vec = np.array(query_vector)
    scored_chunks = []
    for chunk in res.data:
        emb = chunk.get("embedding")
        if emb:
            c_vec = np.array(emb if isinstance(emb, list) else json.loads(emb))
            sim = float(np.dot(q_vec, c_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(c_vec) + 1e-10))
            chunk_copy = dict(chunk)
            chunk_copy["similarity"] = sim
            scored_chunks.append(chunk_copy)

    scored_chunks.sort(key=lambda x: x["similarity"], reverse=True)
    return scored_chunks[:top_k]


def reciprocal_rank_fusion(query_results_list: list[list[dict]], top_k: int = 4, k: int = 60) -> list[dict]:
    """Merges multiple vector search rank lists using Reciprocal Rank Fusion (RRF)."""
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for chunk_list in query_results_list:
        for rank, chunk in enumerate(chunk_list, start=1):
            c_id = chunk["id"]
            if c_id not in chunk_map:
                chunk_map[c_id] = dict(chunk)
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (1.0 / (k + rank))

    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    
    fused_chunks = []
    for cid in sorted_ids[:top_k]:
        chunk = chunk_map[cid]
        chunk["rrf_score"] = rrf_scores[cid]
        fused_chunks.append(chunk)

    return fused_chunks


def generate_llm_answer(user_query: str, retrieved_chunks: list[dict]) -> str:
    """Generates final answer using retrieved context."""
    if not retrieved_chunks:
        return "No relevant context retrieved to answer the query."

    context = "\n\n".join([f"[Chunk {idx+1}]: {c['chunk_text']}" for idx, c in enumerate(retrieved_chunks)])
    prompt = f"""Answer the question based ONLY on the context below.

User Question: {user_query}

Context Chunks:
{context}

Answer:"""

    try:
        return call_gemini(prompt)
    except Exception as e:
        return f"Error generating answer: {e}"


def evaluate_against_golden_truth(user_query: str, generated_answer: str, golden_truth: str) -> tuple[float, str]:
    """Uses LLM-as-a-judge to compare generated RAG answer against golden truth ground truth."""
    prompt = f"""You are an objective AI evaluator comparing a RAG system's generated output against a Ground Truth expected answer.

User Query: "{user_query}"
Expected Ground Truth Answer: "{golden_truth}"
Generated RAG Output: "{generated_answer}"

Evaluate the accuracy, completeness, and correctness of the Generated RAG Output against the Ground Truth Answer.
Score the answer from 0 to 100.
Respond ONLY with a valid JSON object in this format:
{{"score": 90, "reasoning": "Brief explanation detailing specific missing facts or inaccuracies if any."}}"""

    try:
        raw_text = call_gemini(prompt)
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text.strip())
        return float(data.get("score", 0.0)), data.get("reasoning", "")
    except Exception as e:
        return 0.0, f"Evaluation error: {e}"


def run_adaptive_rag_pipeline(query: str, golden_truth: str, target_score: float = 80.0, max_retries: int = 3, top_k: int = 4) -> dict:
    """Executes an adaptive self-correcting RAG loop using LLM-as-a-Judge feedback."""
    last_feedback = None
    last_query_used = query
    
    best_attempt = {
        "score": -1.0,
        "answer": "",
        "reasoning": "",
        "chunks": [],
        "attempts_taken": 0
    }

    for attempt in range(1, max_retries + 1):
        print(f"\n  [Loop Attempt {attempt}/{max_retries}] Rewrite & Retrieval Phase...")
        
        # 1. Generate query variations based on feedback
        rewritten_queries = rewrite_query_with_feedback(query, last_query=last_query_used, feedback=last_feedback)
        search_queries = [query] + rewritten_queries
        print(f"    Queries Used: {search_queries}")

        # 2. Retrieve chunks across all queries and fuse with RRF
        all_chunk_lists = [retrieve_chunks(q, top_k=top_k) for q in search_queries]
        retrieved_chunks = reciprocal_rank_fusion(all_chunk_lists, top_k=top_k)

        # 3. Generate Answer
        generated_answer = generate_llm_answer(query, retrieved_chunks)

        # 4. Evaluate with LLM-as-a-Judge against Golden Truth
        score, reasoning = evaluate_against_golden_truth(query, generated_answer, golden_truth)
        print(f"    Judge Score: {score:.1f}% | Feedback: {reasoning}")

        # Track best score across attempts
        if score > best_attempt["score"]:
            best_attempt = {
                "score": score,
                "answer": generated_answer,
                "reasoning": reasoning,
                "chunks": retrieved_chunks,
                "attempts_taken": attempt
            }

        # Threshold Exit Condition
        if score >= target_score:
            print(f"  -> Target score reached ({score:.1f}% >= {target_score:.1f}%)! Ending loop.")
            break
        else:
            print(f"  -> Score below target threshold ({target_score:.1f}%). Retrying with feedback...")
            last_feedback = reasoning
            last_query_used = search_queries[1] if len(search_queries) > 1 else query

    return best_attempt


def evaluate_dataset(excel_path: str = "datasets/golden_dataset_for_RAG_evaluation.xlsx", top_k: int = 4, target_score: float = 80.0, max_retries: int = 3):
    """Loads Golden Dataset Excel file and runs full adaptive evaluation comparison."""
    print("Loading Golden Dataset Excel sheet...")
    df = pd.read_excel(excel_path, sheet_name="Golden Dataset")
    
    valid_rows = df.dropna(subset=["Sample query", "Expected answer"])
    results = []
    
    for idx, row in valid_rows.iterrows():
        sn = row.get("S/N", idx)
        topic = row.get("Topic of the query", "General")
        query = row.get("Sample query")
        golden_truth = row.get("Expected answer")

        if str(sn).upper() == "EX":
            continue

        print("\n" + "="*80)
        print(f"  [Item {sn}] Topic: {topic}")
        print(f"  Query: \"{query}\"")
        print("="*80)

        # Baseline: Single-pass Original Query Only
        print("\n--- BASELINE (Single Query, No Rewriting Loop) ---")
        base_chunks = retrieve_chunks(query, top_k=top_k)
        base_answer = generate_llm_answer(query, base_chunks)
        base_score, base_reason = evaluate_against_golden_truth(query, base_answer, golden_truth)
        print(f"  Baseline Score: {base_score:.1f}% | Reasoning: {base_reason}")

        # Adaptive Loop with LLM Feedback
        print("\n--- ADAPTIVE FEEDBACK LOOP (Self-Correcting RAG) ---")
        best_run = run_adaptive_rag_pipeline(
            query=query, 
            golden_truth=golden_truth, 
            target_score=target_score, 
            max_retries=max_retries, 
            top_k=top_k
        )

        results.append({
            "S/N": sn,
            "Topic": topic,
            "Query": query,
            "Golden Truth": golden_truth,
            "Baseline_Score": base_score,
            "Baseline_Reasoning": base_reason,
            "Adaptive_Score": best_run["score"],
            "Adaptive_Reasoning": best_run["reasoning"],
            "Attempts_Required": best_run["attempts_taken"],
            "Score_Delta": best_run["score"] - base_score
        })

    # Summary Output
    res_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("                        OVERALL EVALUATION SUMMARY")
    print("="*80)
    print(f"Average Baseline Score             : {res_df['Baseline_Score'].mean():.2f}%")
    print(f"Average Adaptive Loop Score         : {res_df['Adaptive_Score'].mean():.2f}%")
    print(f"Average Score Improvement           : {res_df['Score_Delta'].mean():+.2f}%")
    print(f"Average Attempts Needed per Query   : {res_df['Attempts_Required'].mean():.2f}")
    print("="*80)


if __name__ == "__main__":
    evaluate_dataset()