"""
Extract text from the PDFs and the correspondence emails, chunk it,
embed it, and load it into document_chunks.

Client IDs (CL001, CL013, etc) are detected in the text with a simple
regex and stored in documents.metadata->related_clients. This is what
lets the agent later filter vector search to "documents that mention
this client" for the multi-hop questions, without needing manual
tagging of which document belongs to which client.

Run with: python -m ingestion.parse_documents
"""

import json
import re
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from db.connection import run_write, run_write_many

DATA_DIR = Path("data")
CLIENT_ID_PATTERN = re.compile(r"\bCL\d{3}\b")

model = SentenceTransformer("all-MiniLM-L6-v2")


def find_related_clients(text: str) -> list[str]:
    return sorted(set(CLIENT_ID_PATTERN.findall(text)))


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


def insert_document(doc_type: str, title: str, source_path: str, text: str, extra_metadata: dict | None = None):
    related_clients = find_related_clients(text)
    metadata = {"related_clients": related_clients, **(extra_metadata or {})}

    rows = run_write_and_return_id(
        """
        insert into documents (doc_type, title, source_path, metadata)
        values (%s, %s, %s, %s)
        returning id
        """,
        (doc_type, title, source_path, json.dumps(metadata)),
    )
    document_id = rows[0]["id"]

    chunks = chunk_text(text)
    if not chunks:
        return
    embeddings = model.encode(chunks, normalize_embeddings=True)

    chunk_rows = [
        (document_id, i, chunk, embedding.tolist())
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    run_write_many(
        """
        insert into document_chunks (document_id, chunk_index, chunk_text, embedding)
        values (%s, %s, %s, %s)
        """,
        chunk_rows,
    )
    print(f"  {title}: {len(chunks)} chunks, related clients: {related_clients or 'none detected'}")


def run_write_and_return_id(sql: str, params: tuple) -> list[dict]:
    """insert ... returning id needs its own helper since run_write doesn't fetch."""
    from db.connection import get_connection
    import psycopg2.extras

    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            result = [dict(row) for row in cur.fetchall()]
        conn.commit()
    return result


def ingest_pdf(path: Path, doc_type: str):
    reader = PdfReader(path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    insert_document(doc_type=doc_type, title=path.stem, source_path=path.name, text=text)


def ingest_email_threads(path: Path):
    with open(path) as f:
        data = json.load(f)

    # Top-level file is {"dataset_note": ..., "email_threads": [...]}, not a
    # bare list.
    for thread in data["email_threads"]:
        subject = thread.get("subject", "untitled thread")
        messages = thread.get("messages", [])
        # Real message keys are "from", "date", "body" (no "text" key).
        combined_text = "\n\n".join(
            f"From: {m.get('from')}\nDate: {m.get('date')}\n{m.get('body', '')}"
            for m in messages
        )
        insert_document(
            doc_type="email",
            title=subject,
            source_path=f"client_correspondence.json#{subject}",
            text=combined_text,
            # NOTE: there's no "type" key on a thread, so this was always
            # None - left as-is since it's outside the requested key list.
            # Threads do carry "related_client_id" instead, which duplicates
            # find_related_clients()'s regex extraction from the body text -
            # flag if you want it captured here too.
            extra_metadata={"thread_type": thread.get("type")},
        )


if __name__ == "__main__":
    pdf_files = {
        "policy_kyc_onboarding.pdf": "policy",
        "policy_investment_suitability.pdf": "policy",
        "fund_factsheet_safe.pdf": "factsheet",
        "fund_factsheet_global_bond_income.pdf": "factsheet",
        "fund_factsheet_global_reit_basket.pdf": "factsheet",
        "fund_factsheet_balanced_income_growth.pdf": "factsheet",
        "fund_factsheet_pacific_growth_equity.pdf": "factsheet",
        "fund_factsheet_ilp_regular_premium.pdf": "factsheet",
        "fund_factsheet_global_tech_innovation.pdf": "factsheet",
        "fund_factsheet_dci_aud_usd.pdf": "factsheet",
        "fund_factsheet_private_equity_fund_iv.pdf": "factsheet",
        "fund_factsheet_exotic_unsafe.pdf": "factsheet",
        "rm_call_notes_log.pdf": "call_note",
        "client_complaint_letters.pdf": "complaint",
        "complex_product_risk_acknowledgement_forms.pdf": "form",
    }

    print("Ingesting PDFs...")
    for filename, doc_type in pdf_files.items():
        path = DATA_DIR / filename
        if path.exists():
            ingest_pdf(path, doc_type)
        else:
            print(f"  skipped {filename}, not found in {DATA_DIR}")

    print("Ingesting email threads...")
    ingest_email_threads(DATA_DIR / "client_correspondence.json")
