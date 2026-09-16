"""Request and response models for the agent and data endpoints."""

from pydantic import BaseModel
from datetime import date


class AskRequest(BaseModel):
    question: str


class ToolCallLog(BaseModel):
    tool: str
    input: dict


class AskResponse(BaseModel):
    answer: str
    tool_calls: list[ToolCallLog]


class ClientSummary(BaseModel):
    client_id: str
    name: str | None = None
    jurisdiction: str | None = None
    risk_profile: str | None = None
    accredited_investor: bool | None = None


class ClientDetail(ClientSummary):
    kyc_status: str | None = None
    raw_profile: dict | None = None


class HoldingOut(BaseModel):
    product_name: str | None = None
    sri: int | None = None
    holding_value: float | None = None
    currency: str | None = None
    pct_of_portfolio: float | None = None
    as_of_date: date | None = None


class TransactionOut(BaseModel):
    txn_date: date | None = None
    txn_type: str | None = None
    product_name: str | None = None
    amount: float | None = None
    currency: str | None = None
    status: str | None = None


class DocumentOut(BaseModel):
    id: int
    doc_type: str | None = None
    title: str | None = None
    source_path: str | None = None
    metadata: dict | None = None
