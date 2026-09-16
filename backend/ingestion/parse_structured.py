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
        data = json.load(f)

    client_rows = []
    holding_rows = []

    # Top-level file is {"dataset_note": ..., "region_focus": ..., "clients": [...]},
    # not a bare list.
    for client in data["clients"]:
        client_id = client["client_id"]

        # No "jurisdiction" key exists in the source. Using residency_country
        # as the closest match; "nationality" (citizenship) is the other
        # candidate. Note residency_country is sometimes a descriptive phrase
        # rather than a clean country name, e.g.
        # "Indonesia (Jakarta), banks with Singapore booking centre".
        jurisdiction = client.get("residency_country")

        # No boolean accredited-investor flag exists. "investor_status" is
        # free text with three observed values: "Accredited Investor (SFA
        # definition met)", "Retail Investor", "Professional Investor (per
        # SFO Cap. 571)". This only flags "Accredited Investor" as True -
        # "Professional Investor" is a distinct (generally higher) SFA/SFO
        # category and is NOT counted here. Confirm whether Professional
        # Investor clients should also be treated as accredited before this
        # touches suitability/compliance logic downstream.
        investor_status = client.get("investor_status", "")
        accredited_investor = "accredited investor" in investor_status.lower()

        client_rows.append((
            client_id,
            client.get("name"),
            jurisdiction,
            client.get("kyc_status"),
            client.get("risk_profile"),
            accredited_investor,
            json.dumps(client),
        ))

        for holding in client.get("portfolio_holdings", []):
            holding_rows.append((
                client_id,
                holding.get("product_name"),
                # No SRI (Summary Risk Indicator) rating field exists
                # anywhere in this file - always None. It may live in the
                # fund factsheet PDFs instead, which this script doesn't read.
                None,
                holding.get("value_sgd"),
                holding.get("currency"),
                holding.get("allocation_pct"),
                # No per-holding as-of date exists. The client-level
                # "last_portfolio_review_date" is a candidate proxy but
                # describes the whole portfolio, not this specific holding -
                # left None rather than implying false precision.
                None,
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
        # Real header: transaction_id,client_id,date,transaction_type,
        # product_name,amount,currency,status,notes.
        # transaction_id and notes aren't captured below, matching the
        # existing 7-column insert - flag if those should be added.
        for row in reader:
            rows.append((
                row.get("client_id"),
                row.get("date"),
                row.get("transaction_type"),
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
