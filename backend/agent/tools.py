"""
The two tools the agent can call. Keep tool logic here, separate from
the agent loop, so you can test each one directly against the DB
before wiring up the LLM.
"""

from sentence_transformers import SentenceTransformer

from db.connection import run_query

model = SentenceTransformer("all-MiniLM-L6-v2")

# Read-only guard: the agent should only ever SELECT. This is a
# hackathon safety net, not a production-grade sanitizer.
FORBIDDEN_KEYWORDS = ("insert", "update", "delete", "drop", "alter", "truncate")


def sql_query(query: str) -> list[dict]:
    """Run a read-only SQL query against the structured tables
    (clients, portfolio_holdings, transactions) and return the rows.
    """
    lowered = query.lower()
    if not lowered.strip().startswith("select"):
        return [{"error": "Only SELECT queries are allowed."}]
    if any(word in lowered for word in FORBIDDEN_KEYWORDS):
        return [{"error": "Query contains a forbidden keyword."}]

    try:
        return run_query(query)
    except Exception as e:
        return [{"error": str(e)}]


def vector_search(question: str, top_k: int = 6, client_id: str | None = None) -> list[dict]:
    """Semantic search over document_chunks (policies, factsheets,
    call notes, emails, complaints, forms). Optionally filter to
    chunks whose parent document mentions a specific client_id.
    """
    embedding = model.encode([question], normalize_embeddings=True)[0].tolist()

    rows = run_query(
        "select * from match_document_chunks(%s::vector, %s)",
        (embedding, top_k * 3 if client_id else top_k),
    )

    if client_id:
        rows = [r for r in rows if client_id in (r["metadata"] or {}).get("related_clients", [])]
        rows = rows[:top_k]

    return rows


# Tool definitions in Anthropic's tool-use format, imported by agent.py.
TOOL_DEFINITIONS = [
    {
        "name": "sql_query",
        "description": (
            "Run a read-only SQL SELECT against the structured banking data. "
            "Tables: clients(client_id, name, jurisdiction, kyc_status, risk_profile, "
            "accredited_investor, raw_profile jsonb), "
            "portfolio_holdings(client_id, product_name, sri, holding_value, currency, "
            "pct_of_portfolio, as_of_date), "
            "transactions(client_id, txn_date, txn_type, product_name, amount, currency, status). "
            "Use this for numeric aggregation, filtering, and lookups over clients, "
            "holdings, and transactions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A SELECT statement."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "vector_search",
        "description": (
            "Semantic search over unstructured documents: policies, fund fact sheets, "
            "RM call notes, client correspondence emails, complaint letters, and risk "
            "acknowledgement forms. Use this for policy language, narrative context, "
            "and anything not captured in the structured tables. Pass client_id to "
            "narrow results to documents that mention a specific client."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "client_id": {"type": "string", "description": "Optional, e.g. CL013"},
            },
            "required": ["question"],
        },
    },
]
