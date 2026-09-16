"""
FastAPI backend for the Wealth Advisor Assistant.

Equivalent to your Django views.py + urls.py, but as a standalone API
service that the Streamlit frontend calls over HTTP instead of Django
rendering the page itself. Run with:

    uvicorn app.main:app --reload --port 8000

(run from the project root so the `rag` and `retrieval` packages import
correctly - same reason your Django app relied on being inside the project).
"""
import logging
import time
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rag.agent import agent_executor
from rag.citations import extract_citations
from rag.sql_store import ping as ping_sql_store
from retrieval.vectorstore import build_or_load_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Wealth Advisor Assistant API")

# Streamlit runs on a different port during local dev (8501 by default),
# so it's a different origin as far as the browser's CORS rules go.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    text: str
    session_id: str = "default"


class Citation(BaseModel):
    """
    One retrieved passage, as re-derived from the tool output itself
    (rag/citations.py) rather than trusted from the model's own inline
    [tag] formatting - see that module's docstring for why. "used" tells
    the frontend whether the model actually cited this passage in its
    final answer, vs. retrieved-but-not-used.
    """
    tag: str
    source: str
    doc_type: str
    used: bool


class QueryResponse(BaseModel):
    success: bool
    input: str
    output: str
    intermediate_steps: List[str] = []
    citations: List[Citation] = []
    response_time: float


@app.post("/api/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Async endpoint - equivalent to your `query_chatbot_api`. FastAPI's
    async support pairs with `agent_executor.ainvoke` the same way your
    Django async view paired with it.
    """
    start_time = time.time()
    text = request.text.strip()

    if not text:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "text is required"},
        )

    try:
        logger.info(f"Processing query: {text[:80]}...")
        result = await agent_executor.ainvoke({"input": text})

        output = result.get("output", "No response generated")
        raw_steps = result.get("intermediate_steps", [])
        intermediate_steps = [str(step) for step in raw_steps]

        # Ground-truth citation list built from what retrieve_documents
        # actually returned this turn, not just the [tag]s the model
        # happened to type correctly - see rag/citations.py.
        citations = extract_citations(raw_steps, output)

        return QueryResponse(
            success=True,
            input=text,
            output=output,
            intermediate_steps=intermediate_steps,
            citations=citations,
            response_time=round(time.time() - start_time, 2),
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Error: {e}"},
        )


@app.get("/api/health")
def health_check():
    """Equivalent to your Django health_check view."""
    status = {"status": "healthy", "fastapi": "running", "agent": "initialized"}
    try:
        chunk_count = build_or_load_vector_store()
        status["vector_store"] = f"connected ({chunk_count} chunks)"
    except Exception as e:
        # most common causes: DATABASE_URL unset/wrong, or the Supabase
        # project's IPv6-only "direct connection" string was used instead
        # of the IPv4-friendly "session pooler" one
        status["vector_store"] = f"error: {e}"
        status["status"] = "unhealthy"

    try:
        client_count = ping_sql_store()
        status["sql_store"] = f"connected ({client_count} clients)"
    except Exception as e:
        status["sql_store"] = f"error: {e}"
        status["status"] = "unhealthy"

    return status