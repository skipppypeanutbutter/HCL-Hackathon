"""
Load clients_portfolio.json and transactions.csv into Supabase.

IMPORTANT: I haven't seen your actual clients_portfolio.json, so the
field names below (CLIENT_FIELD_MAP, HOLDING_FIELD_MAP) are guesses at
common naming. Open the file first, check the real keys, and adjust
the .get() calls to match before running this. That's a five-minute
fix, better than debugging a silent empty-column problem later.

We use the JSON file as the source of truth for both client attributes
and holdings (it's the fuller, nested version). The CSV is the same
holdings data flattened, ingest it instead of the JSON only if the
JSON turns out to be missing fields the CSV has.

Run with: python -m ingestion.parse_structured
"""

import csv
import json
from pathlib import Path

from db.connection import run_write, run_write_many

DATA_DIR = Path("data")  # adjust to wherever the source files live


def load_clients(path: Path):
    with open(path) as f:
        clients = json.load(f)

    client_rows = []
    holding_rows = []

    for client in clients:
        # TODO confirm these keys against the real file
        client_id = client["client_id"]
        client_rows.append((
            client_id,
            client.get("name"),
            client.get("jurisdiction"),
            client.get("kyc_status"),
            client.get("risk_profile"),
            client.get("accredited_investor", False),
            json.dumps(client),
        ))

        for holding in client.get("portfolio", client.get("holdings", [])):
            holding_rows.append((
                client_id,
                holding.get("product_name"),
                holding.get("sri"),
                holding.get("value", holding.get("holding_value")),
                holding.get("currency"),
                holding.get("pct_of_portfolio"),
                holding.get("as_of_date"),
            ))

    run_write_many(
        """
        insert into clients
            (client_id, name, jurisdiction, kyc_status, risk_profile, accredited_investor, raw_profile)
        values (%s, %s, %s, %s, %s, %s, %s)
        on conflict (client_id) do update set raw_profile = excluded.raw_profile
        """,
        client_rows,
    )

    run_write_many(
        """
        insert into portfolio_holdings
            (client_id, product_name, sri, holding_value, currency, pct_of_portfolio, as_of_date)
        values (%s, %s, %s, %s, %s, %s, %s)
        """,
        holding_rows,
    )

    print(f"Loaded {len(client_rows)} clients, {len(holding_rows)} holdings")


def load_transactions(path: Path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # TODO confirm these column names against the real CSV header
            rows.append((
                row.get("client_id"),
                row.get("date") or row.get("txn_date"),
                row.get("type") or row.get("txn_type"),
                row.get("product_name"),
                row.get("amount"),
                row.get("currency"),
                row.get("status"),
            ))

    run_write_many(
        """
        insert into transactions
            (client_id, txn_date, txn_type, product_name, amount, currency, status)
        values (%s, %s, %s, %s, %s, %s, %s)
        """,
        rows,
    )
    print(f"Loaded {len(rows)} transactions")


if __name__ == "__main__":
    load_clients(DATA_DIR / "clients_portfolio.json")
    load_transactions(DATA_DIR / "transactions.csv")
