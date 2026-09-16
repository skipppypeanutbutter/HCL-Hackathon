"""
Structured data ingestion for the Wealth Advisor RAG system.
Manages the shared SQLite in-memory database connection.
"""
import os
import sqlite3
from pathlib import Path
from build_sqlite_db import build_sqlite_db

# Point to project root (one level up from 'SQLite db/')
BASE_DIR = Path(__file__).resolve().parent.parent

CLIENTS_PATH = Path(os.getenv("CLIENTS_PATH", BASE_DIR / "SQLite_db/data/clients_portfolio.csv"))
TRANSACTIONS_PATH = Path(os.getenv("TRANSACTIONS_PATH", BASE_DIR / "SQLite_db/data/transactions.csv"))

_sqlite_conn: sqlite3.Connection | None = None


def get_db_connection() -> sqlite3.Connection:
    """Return the shared in-memory SQLite connection, building it on first call."""
    global _sqlite_conn
    if _sqlite_conn is not None:
        return _sqlite_conn

    _sqlite_conn = build_sqlite_db(CLIENTS_PATH, TRANSACTIONS_PATH)
    return _sqlite_conn