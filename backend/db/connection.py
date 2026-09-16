"""
Shared Postgres connection helper.

Uses a direct connection string (the Supabase 'Session pooler' URI)
rather than the supabase-py client, since raw SQL is what both the
ingestion scripts and the agent's sql_query tool need, and pgvector's
<=> operator isn't exposed through the client library.
"""

import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
load_dotenv()

def get_connection():
    db_url = os.environ["SUPABASE_DB_URL"]
    return psycopg2.connect(db_url)


def run_query(sql: str, params: tuple = ()) -> list[dict]:
    """Run a SELECT and return rows as a list of dicts."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]


def run_write(sql: str, params: tuple = ()):
    """Run an INSERT/UPDATE and commit."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()


def run_write_many(sql: str, rows: list[tuple]):
    """Run the same INSERT for many rows at once."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, rows)
        conn.commit()
