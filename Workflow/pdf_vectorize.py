"""
PDF Vectorization Script
========================
Extracts text from a PDF, chunks it, generates Gemini embeddings,
and stores everything in a PostgreSQL + pgvector database.

Usage:
    python pdf_vectorize.py <path_to_pdf>
    python pdf_vectorize.py --search "your query here"
"""

import os
import sys
import json
import argparse
import psycopg2
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────
DB_CONFIG = {
    "dbname": os.getenv("VECTOR_DB_NAME", "rfi_db"),
    "user": os.getenv("VECTOR_DB_USER", "rfi_user"),
    "password": os.getenv("VECTOR_DB_PASSWORD", "rfi_user"),
    "host": os.getenv("VECTOR_DB_HOST", "localhost"),
    "port": int(os.getenv("VECTOR_DB_PORT", "5434")),
}

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("❌ GOOGLE_API_KEY is not set in .env")
    sys.exit(1)

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# ── Chunk size (characters) ──────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ── Helper functions ─────────────────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text page-by-page from a PDF using PyMuPDF (fitz)."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        if text.strip():
            pages.append({"page": page_num + 1, "text": text})
    doc.close()
    return pages


def chunk_text(pages: list[dict]) -> list[dict]:
    """Split page texts into overlapping chunks."""
    chunks = []
    chunk_index = 0

    for page_info in pages:
        text = page_info["text"]
        page_num = page_info["page"]
        start = 0

        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_content = text[start:end]

            if chunk_content.strip():
                chunks.append({
                    "chunk_index": chunk_index,
                    "page_number": page_num,
                    "content": chunk_content.strip(),
                })
                chunk_index += 1

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def generate_embedding(text: str) -> list[float] | None:
    """Generate a vector embedding using the Gemini embedding model."""
    try:
        return embeddings_model.embed_query(text)
    except Exception as e:
        print(f"  ⚠️  Embedding error: {e}")
        return None


def store_document(pdf_path: str, chunks: list[dict]) -> int:
    """Insert the document record and return its ID."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO documents (filename, total_pages, total_chunks) VALUES (%s, %s, %s) RETURNING id",
        (os.path.basename(pdf_path), max(c["page_number"] for c in chunks), len(chunks)),
    )
    doc_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return doc_id


def store_chunks(doc_id: int, chunks: list[dict]):
    """Embed each chunk and insert it into the database."""
    conn = get_conn()
    cur = conn.cursor()

    for i, chunk in enumerate(chunks):
        print(f"  ⏳ Embedding chunk {i + 1}/{len(chunks)}...", end="\r")
        embedding = generate_embedding(chunk["content"])
        cur.execute(
            """
            INSERT INTO document_chunks (document_id, chunk_index, page_number, content, embedding)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (doc_id, chunk["chunk_index"], chunk["page_number"], chunk["content"],
             json.dumps(embedding) if embedding else None),
        )

    conn.commit()
    cur.close()
    conn.close()
    print()


def similarity_search(query: str, top_k: int = 5):
    """Search for chunks most similar to the query."""
    embedding = generate_embedding(query)
    if not embedding:
        print("❌ Could not generate embedding for query.")
        return

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            d.filename,
            dc.page_number,
            dc.content,
            (dc.embedding <=> %s::vector) AS distance
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.id
        WHERE dc.embedding IS NOT NULL
        ORDER BY dc.embedding <=> %s::vector
        LIMIT %s;
        """,
        (json.dumps(embedding), json.dumps(embedding), top_k),
    )
    results = cur.fetchall()
    cur.close()
    conn.close()

    if not results:
        print("No results found.")
        return

    print(f"\n{'─' * 60}")
    print(f"  Top {len(results)} results for: \"{query}\"")
    print(f"{'─' * 60}")
    for i, (filename, page, content, distance) in enumerate(results, 1):
        similarity = 1 - distance
        print(f"\n  [{i}]  📄 {filename} — page {page} — similarity: {similarity:.4f}")
        print(f"       {content[:200]}...")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────
def ingest_pdf(pdf_path: str):
    if not os.path.isfile(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    print(f"📄 Reading PDF: {pdf_path}")
    pages = extract_text_from_pdf(pdf_path)
    print(f"   Found {len(pages)} pages with text.")

    print("✂️  Chunking text...")
    chunks = chunk_text(pages)
    print(f"   Created {len(chunks)} chunks.")

    print("💾 Storing document in database...")
    doc_id = store_document(pdf_path, chunks)
    print(f"   Document ID: {doc_id}")

    print("🧠 Generating embeddings & storing chunks...")
    store_chunks(doc_id, chunks)

    print(f"✅ Done! {len(chunks)} chunks stored with embeddings.")
    return doc_id


def main():
    parser = argparse.ArgumentParser(description="PDF Vectorization & Search")
    parser.add_argument("pdf", nargs="?", help="Path to a PDF file to ingest")
    parser.add_argument("--search", "-s", type=str, help="Search query for similarity search")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results to return (default: 5)")
    args = parser.parse_args()

    if args.search:
        similarity_search(args.search, args.top_k)
    elif args.pdf:
        ingest_pdf(args.pdf)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
