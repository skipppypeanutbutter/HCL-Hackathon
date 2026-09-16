"""
One-off script to check what's actually in document_chunks / documents.

Run from the project root:
    python3 check_chunks.py

Uses the same DATABASE_URL/SUPABASE_DB_URL + connection pool as the app
(rag/db.py), so if this works, the app's vector retrieval will too.
"""
from dotenv import load_dotenv

load_dotenv()

from rag.db import get_pool

pool = get_pool()
conn = pool.getconn()
try:
    with conn.cursor() as cur:
        cur.execute("select count(*) from documents")
        print("documents:", cur.fetchone()[0])

        cur.execute("select count(*) from document_chunks")
        print("document_chunks:", cur.fetchone()[0])

        cur.execute("select distinct doc_type from documents order by doc_type")
        print("doc_types present:", [r[0] for r in cur.fetchall()])

        print("\nfirst 5 chunks:")
        cur.execute(
            """
            select d.title, d.doc_type, dc.chunk_index,
                   left(dc.chunk_text, 100) as preview
            from document_chunks dc
            join documents d on d.id = dc.document_id
            order by dc.id
            limit 5
            """
        )
        for title, doc_type, chunk_index, preview in cur.fetchall():
            print(f"  [{doc_type}] {title} (chunk {chunk_index}): {preview}...")
finally:
    pool.putconn(conn)