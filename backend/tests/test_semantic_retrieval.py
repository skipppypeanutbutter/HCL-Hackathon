"""
Tests for semantic retrieval.

Two groups, on purpose:

1. Embedding unit tests: no DB needed, just check the embedding model
   ranks a genuinely relevant sentence above an unrelated one. These
   run anywhere, in seconds, and catch "did I break the encoding step"
   separately from "is retrieval against my actual data working".

2. Integration tests: hit the real Supabase project through
   agent.tools.vector_search. These need SUPABASE_DB_URL set and the
   data already ingested (see README). They're skipped automatically
   if that env var isn't set, so this file doesn't fail in an
   environment with no DB configured.

Run with: pytest tests/test_semantic_retrieval.py -v
"""

import os
import pytest
from sentence_transformers import SentenceTransformer, util

from agent.tools import vector_search

HAS_DB = "SUPABASE_DB_URL" in os.environ
skip_if_no_db = pytest.mark.skipif(not HAS_DB, reason="SUPABASE_DB_URL not set, skipping integration test")

model = SentenceTransformer("all-MiniLM-L6-v2")


def cosine_similarity(text_a: str, text_b: str) -> float:
    embeddings = model.encode([text_a, text_b], normalize_embeddings=True)
    return float(util.cos_sim(embeddings[0], embeddings[1]))


# ---------------------------------------------------------------------------
# 1. Embedding unit tests (no DB required)
# ---------------------------------------------------------------------------

def test_relevant_text_scores_higher_than_irrelevant_text():
    query = "client wants to reduce portfolio risk"
    relevant = "The client has requested to de-risk their holdings ahead of retirement."
    irrelevant = "The quarterly fund performance report shows a 3% return."

    score_relevant = cosine_similarity(query, relevant)
    score_irrelevant = cosine_similarity(query, irrelevant)

    assert score_relevant > score_irrelevant


def test_paraphrased_query_still_matches():
    # Same underlying meaning, different wording. If this fails, the
    # embedding model isn't capturing semantics the way retrieval needs.
    original = "Is this product suitable for a conservative investor?"
    paraphrase = "Would a client with low risk tolerance be allowed to buy this fund?"
    unrelated = "The office is closed for a public holiday next Monday."

    score_paraphrase = cosine_similarity(original, paraphrase)
    score_unrelated = cosine_similarity(original, unrelated)

    assert score_paraphrase > score_unrelated
    # A loose sanity threshold, not a tuned target. Adjust if it turns
    # out too strict or too loose for your actual phrasing.
    assert score_paraphrase > 0.4


def test_identical_text_scores_near_one():
    text = "Private Equity Co-Investment Vehicle Fund IV is a Complex/illiquid product."
    score = cosine_similarity(text, text)
    assert score > 0.99


# ---------------------------------------------------------------------------
# 2. Integration tests against the real Supabase data
# ---------------------------------------------------------------------------

@skip_if_no_db
def test_vector_search_returns_results_for_a_basic_policy_question():
    results = vector_search("What are the KYC requirements for onboarding a new client?", top_k=5)
    assert len(results) > 0


@skip_if_no_db
def test_vector_search_finds_the_suitability_policy_for_a_suitability_question():
    results = vector_search(
        "Is the APEX Autocallable Note suitable for a Conservative client?",
        top_k=6,
    )
    titles = [r["title"] for r in results]
    # This should surface either the suitability policy or the fund's
    # own fact sheet (SRI 7/7). If neither shows up in top_k, retrieval
    # quality on this question needs attention before the demo.
    assert any("suitability" in t.lower() or "exotic" in t.lower() for t in titles), (
        f"Expected the suitability policy or the APEX fact sheet in results, got: {titles}"
    )


@skip_if_no_db
def test_vector_search_client_filter_only_returns_matching_client():
    # Adjust this client_id if CL013 doesn't exist in your ingested data.
    results = vector_search("portfolio de-risking request", top_k=5, client_id="CL013")
    for r in results:
        related = (r["metadata"] or {}).get("related_clients", [])
        assert "CL013" in related, f"Got a chunk not tagged for CL013: {r['title']} -> {related}"


@skip_if_no_db
def test_vector_search_similarity_scores_are_descending():
    # match_document_chunks should return results ordered by relevance.
    results = vector_search("client risk profiling requirements", top_k=6)
    scores = [r["similarity"] for r in results]
    assert scores == sorted(scores, reverse=True)