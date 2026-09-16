-- Run this in the Supabase SQL Editor after enabling the vector extension.
-- create extension if not exists vector;

create table if not exists clients (
    client_id text primary key,
    name text,
    jurisdiction text,
    kyc_status text,
    risk_profile text,
    accredited_investor boolean,
    raw_profile jsonb          -- full original JSON record, for anything the flat columns miss
);

create table if not exists portfolio_holdings (
    id bigserial primary key,
    client_id text references clients(client_id),
    product_name text,
    sri int,                   -- Specific Risk Indicator, 1-7
    holding_value numeric,
    currency text,
    pct_of_portfolio numeric,
    as_of_date date
);

create table if not exists transactions (
    id bigserial primary key,
    client_id text references clients(client_id),
    txn_date date,
    txn_type text,             -- subscription, redemption, coupon, top_up, pending
    product_name text,
    amount numeric,
    currency text,
    status text
);

-- One row per source document (a PDF, an email thread, etc).
create table if not exists documents (
    id bigserial primary key,
    doc_type text,              -- policy, factsheet, call_note, email, complaint, form
    title text,
    source_path text,
    metadata jsonb              -- e.g. {"related_clients": ["CL013", "CL002"], "sri": 7}
);

-- Chunks of each document, embedded for similarity search.
-- Dimension 384 matches sentence-transformers/all-MiniLM-L6-v2.
-- If you switch embedding models, change this dimension to match.
create table if not exists document_chunks (
    id bigserial primary key,
    document_id bigint references documents(id),
    chunk_index int,
    chunk_text text,
    embedding vector(384)
);

create index if not exists document_chunks_embedding_idx
    on document_chunks using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- Callable from Python as a single query: pass an embedding, get back
-- the closest chunks joined with their parent document's metadata.
create or replace function match_document_chunks(
    query_embedding vector(384),
    match_count int default 6
)
returns table (
    chunk_id bigint,
    document_id bigint,
    chunk_text text,
    doc_type text,
    title text,
    metadata jsonb,
    similarity float
)
language sql stable
as $$
    select
        dc.id as chunk_id,
        dc.document_id,
        dc.chunk_text,
        d.doc_type,
        d.title,
        d.metadata,
        1 - (dc.embedding <=> query_embedding) as similarity
    from document_chunks dc
    join documents d on d.id = dc.document_id
    order by dc.embedding <=> query_embedding
    limit match_count;
$$;
