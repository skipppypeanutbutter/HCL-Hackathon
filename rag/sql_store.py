"""
Read-only SQL query layer over the structured tables in Supabase Postgres:
clients, portfolio_holdings, transactions.

This replaces the earlier local-SQLite version. Your teammate built the
real schema directly in Supabase, so there's no separate "ingestion"
step for structured data anymore on this app's side - whatever's in
Supabase IS the store. This module just connects (via rag/db.py) and
runs read-only queries against it.

Schema (from the Supabase schema visualizer):
  clients(client_id PK text, name, jurisdiction, kyc_status, risk_profile,
          accredited_investor bool, raw_profile jsonb)
  portfolio_holdings(id PK, client_id FK -> clients, product_name, sri int4,
          holding_value numeric, currency, pct_of_portfolio numeric, as_of_date)
  transactions(id PK, client_id, txn_date, txn_type, product_name, amount,
          currency, status)

Only clients.risk_profile/kyc_status/jurisdiction/accredited_investor are
flat columns - everything else about a client from the original dataset
(aum_sgd, risk_score_1_to_10, occupation, notes, suitability_flag, etc.)
looks like it lives inside clients.raw_profile as JSONB. get_schema_description()
below samples the actual keys found in raw_profile so the agent's SQL tool
knows what's in there without you having to hardcode a guess - confirm
with your teammate that raw_profile is the full original client record.
"""
import os

import pandas as pd

from rag.db import get_pool

TABLES = ("clients", "portfolio_holdings", "transactions")
JSONB_COLUMNS = {"clients": ["raw_profile"]}


def get_schema_description() -> str:
    """Human-readable column list per table (+ sampled JSONB keys) - goes
    straight into the query_client_database tool's docstring."""
    pool = get_pool()
    conn = pool.getconn()
    try:
        lines = []
        with conn.cursor() as cur:
            for table in TABLES:
                cur.execute(
                    """
                    select column_name, data_type
                    from information_schema.columns
                    where table_schema = 'public' and table_name = %s
                    order by ordinal_position
                    """,
                    (table,),
                )
                cols = cur.fetchall()
                if not cols:
                    continue
                col_desc = ", ".join(f"{name} {dtype}" for name, dtype in cols)
                lines.append(f"{table}({col_desc})")

                for jcol in JSONB_COLUMNS.get(table, []):
                    cur.execute(f"select jsonb_object_keys({jcol}) from {table} limit 1")
                    if cur.fetchone() is None:
                        continue
                    cur.execute(
                        f"select distinct key from {table}, jsonb_object_keys({jcol}) as key limit 60"
                    )
                    keys = sorted(r[0] for r in cur.fetchall())
                    lines.append(f"  {table}.{jcol} is JSONB; keys seen: {', '.join(keys)}")
        return "\n".join(lines)
    finally:
        pool.putconn(conn)


def run_readonly_sql(query: str) -> str:
    """
    Executes a single read-only SELECT and returns rows as JSON.

    Refuses anything that isn't a SELECT (app-level guard), AND sets the
    Postgres session itself to read-only for the query (DB-level guard) -
    belt and suspenders. If your team wants this locked down further,
    have your teammate create a dedicated read-only Postgres role for
    this app's DATABASE_URL instead of using the default postgres user.
    """
    stripped = query.strip().rstrip(";")
    if not stripped.lower().startswith("select"):
        return "ERROR: only SELECT statements are allowed."

    pool = get_pool()
    conn = pool.getconn()
    try:
        conn.set_session(readonly=True, autocommit=True)
        df = pd.read_sql_query(stripped, conn)
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        conn.set_session(readonly=False, autocommit=False)
        pool.putconn(conn)

    if df.empty:
        return "NO_RESULTS: the query returned no rows."
    return df.to_json(orient="records", indent=2, date_format="iso")


def ping() -> int:
    """Health-check hook: proves the DB is reachable. Returns the client count."""
    pool = get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("select count(*) from clients")
            return cur.fetchone()[0]
    finally:
        pool.putconn(conn)