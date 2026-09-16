"""
Request and response models.

Keep every model here so the rest of the app just imports shapes
instead of redefining them inline. Adjust the fields once you know
the real use case (client records, portfolios, transactions, etc).
"""

from pydantic import BaseModel, Field
from typing import Optional


class IngestRequest(BaseModel):
    """A single document to add to the knowledge base."""
    doc_id: str
    text: str
    source: Optional[str] = None


class QueryRequest(BaseModel):
    """A question to ask over the ingested documents."""
    question: str
    top_k: int = Field(default=4, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


class RiskProfileRequest(BaseModel):
    """Example structured-output use case: client risk scoring.
    Swap this out for whatever the actual use case turns out to be.
    """
    age: int
    annual_income: float
    investment_horizon_years: int
    existing_portfolio_value: float
    notes: Optional[str] = None


class RiskProfileResponse(BaseModel):
    risk_category: str
    reasoning: str
    suggested_allocation: dict[str, float]
