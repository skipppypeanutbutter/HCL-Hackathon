"""
Tools available to the RAG agent.

This is the "single agent, multiple tools" pattern discussed in the plan:
the LLM decides which tool a question needs, instead of a hardcoded
if/else branch. The grounding + abstention rule lives in the agent's
system prompt (agent.py), not here - these tools just return raw data
(or "NO_RESULTS") and let the model decide what that means.

- retrieve_documents: semantic search over unstructured docs (fund fact
  sheets, policy, call notes, correspondence) - built from Person A/B's
  ingestion + vector store work.
- lookup_client_portfolio / lookup_client_transactions: exact structured
  lookups against the client CSV/JSON, for when semantic search over
  chunked text is the wrong tool (you want one exact record, not a
  similarity match).
"""
import json
import os
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

# from retrieval.vectorstore import get_retriever

CLIENTS_PATH = Path(os.getenv("CLIENTS_PATH", "data/clients_portfolio.json"))
TRANSACTIONS_PATH = Path(os.getenv("TRANSACTIONS_PATH", "data/transactions.csv"))

_clients_df = None
_transactions_df = None


def _load_clients() -> pd.DataFrame:
    """
    Handles both shapes:
      - a bare list of client records: [{...}, {...}]
      - a wrapper object with metadata + a "clients" array, e.g.:
        {"dataset_note": "...", "region_focus": "APAC", "clients": [{...}, {...}]}
    The hackathon dataset repo uses the second shape - json_normalize on the
    raw dict alone (without unwrapping "clients" first) produces a single
    row with "dataset_note"/"region_focus"/"clients" as columns, which is
    why client_id lookups were silently missing everything.
    """
    global _clients_df
    if _clients_df is None:
        with open(CLIENTS_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        records = raw["clients"] if isinstance(raw, dict) and "clients" in raw else raw
        _clients_df = pd.json_normalize(records)
    return _clients_df


def _load_transactions() -> pd.DataFrame:
    global _transactions_df
    if _transactions_df is None:
        _transactions_df = pd.read_csv(TRANSACTIONS_PATH)
    return _transactions_df


# @tool
# def retrieve_documents(query: str) -> str:
#     """
#     Search fund fact sheets, policy documents, RM call notes, and client
#     correspondence for passages relevant to the query. Returns the top
#     matching passages with their source filenames so the answer can cite
#     them. Use this for product, policy, or general (non-client-specific)
#     questions.
#     """
#     retriever = get_retriever(k=5)
#     results = retriever.invoke(query)
#     if not results:
#         return "NO_RESULTS: nothing relevant was found in the document store."

#     formatted = []
#     for i, doc in enumerate(results, 1):
#         source = doc.metadata.get("source", "unknown")
#         doc_type = doc.metadata.get("doc_type", "")
#         formatted.append(f"[{i}] (source: {source}, type: {doc_type})\n{doc.page_content}")
#     return "\n\n".join(formatted)


@tool
def lookup_client_portfolio(client_id: str) -> str:
    """
    Look up one client's exact record - holdings, risk profile, and KYC
    attributes - by client_id. Use this instead of retrieve_documents
    whenever the question is about a SPECIFIC client's data: it reads the
    structured client record directly rather than searching text.
    """
    df = _load_clients()
    id_col = "client_id" if "client_id" in df.columns else df.columns[0]
    match = df[df[id_col].astype(str) == str(client_id)]
    if match.empty:
        return f"NO_RESULTS: no client found with id '{client_id}'."
    return match.to_json(orient="records", indent=2)


@tool
def lookup_client_transactions(client_id: str, limit: int = 10) -> str:
    """
    Look up a specific client's recent transactions (subscriptions,
    redemptions, coupons) by client_id. Use alongside
    lookup_client_portfolio when the question is about recent activity
    rather than current holdings.
    """
    df = _load_transactions()
    id_col = "client_id" if "client_id" in df.columns else df.columns[0]
    match = df[df[id_col].astype(str) == str(client_id)].tail(limit)
    if match.empty:
        return f"NO_RESULTS: no transactions found for client '{client_id}'."
    return match.to_json(orient="records", indent=2)


@tool
def find_clients_by_risk(risk_level: str = "high", limit: int = 5) -> str:
    """
    List clients ranked by risk appetite - the tool does the sorting, not you.

    Set risk_level="high" for questions like "which client has a high /
    aggressive risk appetite" - returns clients with the HIGHEST
    risk_score_1_to_10 first.

    Set risk_level="low" for questions like "which client has a low /
    conservative risk appetite" - returns clients with the LOWEST
    risk_score_1_to_10 first.

    The first item in the returned list is always the answer to "which
    single client has the highest/lowest risk appetite" - do not re-sort,
    re-rank, or otherwise re-derive the answer from the list yourself.
    Use this instead of guessing a client_id for whole-book questions.
    """
    df = _load_clients()
    if "risk_score_1_to_10" not in df.columns:
        return "NO_RESULTS: risk_score_1_to_10 field not found in client data."

    ascending = risk_level.strip().lower() not in ("high", "aggressive")
    sorted_df = df.sort_values("risk_score_1_to_10", ascending=ascending)

    cols = [c for c in ["client_id", "name", "risk_profile", "risk_score_1_to_10", "relationship_manager"]
            if c in sorted_df.columns]
    return sorted_df[cols].head(limit).to_json(orient="records", indent=2)


# TOOLS = [retrieve_documents, lookup_client_portfolio, lookup_client_transactions, find_clients_by_risk]
TOOLS = [lookup_client_portfolio, lookup_client_transactions, find_clients_by_risk]