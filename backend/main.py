"""
FastAPI wrapper around the agent, for a live demo.

Run with: uvicorn main:app --reload
Then either hit http://localhost:8000/docs for the interactive Swagger
UI (easiest for a demo, no frontend needed), or POST to /ask directly.

CORS is wide open (allow_origins=["*"]) since this is a localhost demo,
not something exposed to the internet. Tighten it if that changes.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    AskRequest,
    AskResponse,
    ClientSummary,
    ClientDetail,
    HoldingOut,
    TransactionOut,
    DocumentOut,
)
from agent.agent import ask
from db.connection import run_query

app = FastAPI(title="Wealth Management RAG Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    result = ask(req.question)
    return AskResponse(
        answer=result["answer"],
        tool_calls=[{"tool": c["tool"], "input": c["input"]} for c in result["tool_calls"]],
    )


# ---------------------------------------------------------------------------
# Data endpoints. Plain reads against the structured tables, no LLM
# involved, for a frontend to browse client data directly.
# ---------------------------------------------------------------------------

@app.get("/clients", response_model=list[ClientSummary])
def list_clients():
    rows = run_query(
        "select client_id, name, jurisdiction, risk_profile, accredited_investor "
        "from clients order by client_id"
    )
    return rows


@app.get("/clients/{client_id}", response_model=ClientDetail)
def get_client(client_id: str):
    rows = run_query("select * from clients where client_id = %s", (client_id,))
    if not rows:
        raise HTTPException(status_code=404, detail=f"No client found with id {client_id}")
    return rows[0]


@app.get("/clients/{client_id}/portfolio", response_model=list[HoldingOut])
def get_client_portfolio(client_id: str):
    rows = run_query(
        "select product_name, sri, holding_value, currency, pct_of_portfolio, as_of_date "
        "from portfolio_holdings where client_id = %s order by pct_of_portfolio desc",
        (client_id,),
    )
    return rows


@app.get("/clients/{client_id}/transactions", response_model=list[TransactionOut])
def get_client_transactions(client_id: str):
    rows = run_query(
        "select txn_date, txn_type, product_name, amount, currency, status "
        "from transactions where client_id = %s order by txn_date desc",
        (client_id,),
    )
    return rows


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(doc_type: str | None = None, client_id: str | None = None):
    query = "select id, doc_type, title, source_path, metadata from documents where 1=1"
    params = []

    if doc_type:
        query += " and doc_type = %s"
        params.append(doc_type)
    if client_id:
        # related_clients is stored as a JSON array inside metadata,
        # jsonb containment checks whether client_id is one of its elements.
        query += " and metadata->'related_clients' @> %s::jsonb"
        params.append(f'"{client_id}"')

    query += " order by id"
    return run_query(query, tuple(params))
