"""
Database builder logic for CSV files.
Creates tables and populates the SQLite database from raw CSV data.
"""
import sqlite3
import textwrap
from pathlib import Path
import pandas as pd


def create_schema(conn: sqlite3.Connection) -> None:
    """Create tables matching the SQLite schema diagram."""
    conn.executescript(textwrap.dedent("""
        CREATE TABLE clients (
            client_id            TEXT PRIMARY KEY,
            name                 TEXT,
            residency_country    TEXT,
            age                  INTEGER,
            investor_status      TEXT,
            risk_profile         TEXT,
            risk_score           INTEGER,
            aum_sgd              REAL,
            investment_objective TEXT,
            kyc_status           TEXT,
            pep_status           TEXT
        );

        CREATE TABLE holdings (
            holding_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id      TEXT REFERENCES clients(client_id) ON DELETE CASCADE,
            product_name   TEXT,
            asset_class    TEXT,
            allocation_pct REAL,
            value_sgd      REAL,
            currency       TEXT
        );

        CREATE TABLE transactions (
            transaction_id   TEXT PRIMARY KEY,
            client_id        TEXT REFERENCES clients(client_id) ON DELETE CASCADE,
            transaction_date TEXT,
            transaction_type TEXT,
            product_name     TEXT,
            amount           REAL,
            currency         TEXT,
            status           TEXT,
            notes            TEXT
        );
    """))


def load_clients_and_holdings(conn: sqlite3.Connection, clients_path: Path) -> None:
    """Load clients_portfolio.csv into clients and holdings tables."""
    resolved_path = clients_path.resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Clients file not found at: {resolved_path}")

    df = pd.read_csv(resolved_path)

    # 1. Map columns for the 'clients' table
    client_col_map = {
        "risk_score_1_to_10": "risk_score",
    }
    df_clients = df.rename(columns=client_col_map)

    expected_client_cols = [
        "client_id", "name", "residency_country", "age", "investor_status",
        "risk_profile", "risk_score", "aum_sgd", "investment_objective",
        "kyc_status", "pep_status"
    ]

    # Extract unique client profiles and insert into 'clients'
    clients_to_insert = df_clients[[c for c in expected_client_cols if c in df_clients.columns]].drop_duplicates(subset=["client_id"])
    clients_to_insert.to_sql("clients", conn, if_exists="append", index=False)

    # 2. Extract holdings rows and insert into 'holdings'
    expected_holding_cols = [
        "client_id", "product_name", "asset_class", "allocation_pct", "value_sgd", "currency"
    ]
    if any(col in df.columns for col in ["product_name", "value_sgd"]):
        holdings_to_insert = df[[c for c in expected_holding_cols if c in df.columns]].dropna(subset=["product_name"])
        holdings_to_insert.to_sql("holdings", conn, if_exists="append", index=False)


def load_transactions(conn: sqlite3.Connection, transactions_path: Path) -> None:
    """Load transactions.csv into the transactions table."""
    resolved_path = transactions_path.resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Transactions file not found at: {resolved_path}")

    df = pd.read_csv(resolved_path)
    if "date" in df.columns and "transaction_date" not in df.columns:
        df = df.rename(columns={"date": "transaction_date"})

    expected_cols = [
        "transaction_id", "client_id", "transaction_date", "transaction_type",
        "product_name", "amount", "currency", "status", "notes"
    ]
    df = df[[col for col in expected_cols if col in df.columns]]
    df.to_sql("transactions", conn, if_exists="append", index=False)


def build_sqlite_db(clients_path: Path, transactions_path: Path) -> sqlite3.Connection:
    """Initialize SQLite database, apply schema, and ingest data from CSV files."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")

    create_schema(conn)
    load_clients_and_holdings(conn, clients_path)
    load_transactions(conn, transactions_path)
    conn.commit()

    return conn