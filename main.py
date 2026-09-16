"""
Entry point. Run with: uvicorn main:app --reload

Three routes are stubbed out:
- POST /ingest        add a document to the knowledge base
- POST /query         ask a question, get a RAG-grounded answer
- POST /risk-profile  example structured-output endpoint

Delete or repurpose whichever routes don't match your actual use case
once it's assigned. The point of this file is to have working
plumbing on day one, not a finished product.
"""

import json
from fastapi import FastAPI, HTTPException

from schemas import (
    IngestRequest,
    QueryRequest,
    QueryResponse,
    RiskProfileRequest,
    RiskProfileResponse,
)
from rag_pipeline import VectorStore, chunk_text, build_context
from llm import call_llm, call_llm_structured

app = FastAPI(title="Hackathon Boilerplate")
store = VectorStore()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest):
    chunks = chunk_text(req.text)
    metadata = [{"doc_id": req.doc_id, "source": req.source} for _ in chunks]
    store.add(chunks, metadata)
    return {"ingested_chunks": len(chunks)}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    results = store.search(req.question, top_k=req.top_k)
    if not results:
        raise HTTPException(status_code=404, detail="No documents ingested yet")

    context = build_context(results)
    system_prompt = (
        "You are a helpful assistant answering questions using only the "
        "provided context. If the context doesn't contain the answer, say so."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {req.question}"
    answer = call_llm(system_prompt, user_prompt)

    sources = sorted({r["metadata"].get("source", "unknown") for r in results})
    return QueryResponse(answer=answer, sources=sources)


@app.post("/risk-profile", response_model=RiskProfileResponse)
def risk_profile(req: RiskProfileRequest):
    system_prompt = "You are a private banking assistant that assesses client risk profiles."
    user_prompt = (
        f"Client details: age {req.age}, annual income {req.annual_income}, "
        f"investment horizon {req.investment_horizon_years} years, "
        f"existing portfolio value {req.existing_portfolio_value}. "
        f"Notes: {req.notes or 'none'}."
    )
    schema_hint = (
        "risk_category (string: conservative/moderate/aggressive), "
        "reasoning (string), "
        "suggested_allocation (object mapping asset class to percentage, values sum to 100)"
    )
    raw = call_llm_structured(system_prompt, user_prompt, schema_hint)
    parsed = json.loads(raw)
    return RiskProfileResponse(**parsed)
