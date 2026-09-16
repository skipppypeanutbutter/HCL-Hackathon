import os
from dotenv import load_dotenv
from pathlib import Path
from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

load_dotenv()  # Load environment variables from .env file

# --- Supabase Configuration ---
load_dotenv()  # Load environment variables from .env file
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- MiniLM Embedding Model Setup ---
# all-MiniLM-L6-v2 outputs 384-dimensional vectors
print("Loading MiniLM embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Chunking Setup ---
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, 
    strip_headers=False
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=150
)

def chunk_markdown_text(markdown_text: str) -> list[str]:
    """Splits Markdown into context-sized text blocks."""
    header_splits = markdown_splitter.split_text(markdown_text)
    
    final_chunks = []
    for doc in header_splits:
        sub_chunks = text_splitter.split_text(doc.page_content)
        final_chunks.extend(sub_chunks)
        
    return final_chunks

def convert_and_ingest_to_supabase(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_path.glob("**/*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{input_dir}'")
        return

    print(f"Found {len(pdf_files)} PDF(s) to convert, embed, and upload...\n")
    converter = DocumentConverter()

    for idx, pdf_file in enumerate(pdf_files, 1):
        print(f"[{idx}/{len(pdf_files)}] Converting: {pdf_file.name} ...")

        try:
            # 1. Parse PDF with Docling
            result = converter.convert(str(pdf_file))
            markdown_content = result.document.export_to_markdown()

            # Save Markdown file locally
            output_file_name = pdf_file.stem + ".md"
            output_file_path = output_path / output_file_name
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            # 2. Insert document record into `documents` table first
            doc_type = "factsheet" if "factsheet" in pdf_file.name else ("policy" if "policy" in pdf_file.name else "document")
            doc_res = supabase.table("documents").insert({
                "title": pdf_file.stem,
                "doc_type": doc_type,
                "source_path": pdf_file.name,
                "metadata": {}
            }).execute()

            # Retrieve the created document's ID
            document_id = doc_res.data[0]["id"]
            print(f" Created document record (ID: {document_id})")

            # 3. Chunk the Markdown content
            chunks = chunk_markdown_text(markdown_content)
            print(f" Created {len(chunks)} chunks.")

            if not chunks:
                continue

            # 4. Generate Embeddings
            print(" -> Generating MiniLM embeddings...")
            embeddings = embedding_model.encode(chunks, show_progress_bar=False)

            # 5. Prepare payload matching `document_chunks` table (including document_id)
            records_to_insert = [
                {
                    "document_id": document_id,  # Link chunk to parent document
                    "chunk_index": chunk_idx,
                    "chunk_text": chunk_text,
                    "embedding": embedding.tolist()
                }
                for chunk_idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings))
            ]

            # 6. Upload chunks to Supabase
            supabase.table("document_chunks").insert(records_to_insert).execute()
            print(f" -> Successfully uploaded chunks and vectors to Supabase!\n")

        except Exception as e:
            print(f" -> Error processing {pdf_file.name}: {e}\n")

    print("All conversions, embeddings, and Supabase uploads complete!")

if __name__ == "__main__":
    INPUT_PDF_DIR = "datasets"
    OUTPUT_MD_DIR = "dataset_chunked"

    convert_and_ingest_to_supabase(INPUT_PDF_DIR, OUTPUT_MD_DIR)