"""
PDF Vectorization — Vector Service Core Logic
=============================================
Handles PDF extraction, chunking, embedding, and DB operations.
Used exclusively by the Vector Service microservice.
"""

import os
import sys
import json
import logging
import psycopg2
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
# Inside Docker: VECTOR_DB_HOST=pg_vector, VECTOR_DB_PORT=5432 (set by compose)
# Local dev:     VECTOR_DB_HOST=localhost, VECTOR_DB_PORT=5434 (set by .env)
DB_CONFIG = {
    "dbname":   os.getenv("VECTOR_DB_NAME",     "rfi_db"),
    "user":     os.getenv("VECTOR_DB_USER",     "rfi_user"),
    "password": os.getenv("VECTOR_DB_PASSWORD", "rfi_user"),
    "host":     os.getenv("VECTOR_DB_HOST",     "pg_vector"),
    "port":     int(os.getenv("VECTOR_DB_PORT", "5432")),
}

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set. Vector service cannot start.")

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200


# ── DB Connection ─────────────────────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(**DB_CONFIG)


# ── PDF Extraction & Chunking ─────────────────────────────────────────────────
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
    logger.info(f"Extracted {len(pages)} pages from {pdf_path}")
    return pages


def chunk_text(pages: list[dict]) -> list[dict]:
    """Split page texts into overlapping chunks."""
    chunks = []
    chunk_index = 0
    for page_info in pages:
        text    = page_info["text"]
        page_num = page_info["page"]
        start   = 0
        while start < len(text):
            end           = start + CHUNK_SIZE
            chunk_content = text[start:end]
            if chunk_content.strip():
                chunks.append({
                    "chunk_index": chunk_index,
                    "page_number": page_num,
                    "content":     chunk_content.strip(),
                })
                chunk_index += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────
def generate_embedding(text: str) -> list[float] | None:
    """Generate a vector embedding using the Gemini embedding model."""
    try:
        return embeddings_model.embed_query(text)
    except Exception as e:
        logger.warning(f"Embedding error: {e}")
        return None


# ── Storage ───────────────────────────────────────────────────────────────────
def store_document(user_id: int | None, filename: str, total_pages: int, total_chunks: int) -> int:
    """Insert a document metadata record (linked to a user or None for policy) and return its doc_id."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        "INSERT INTO documents (user_id, filename, total_pages, total_chunks) VALUES (%s, %s, %s, %s) RETURNING id",
        (user_id, filename, total_pages, total_chunks),
    )
    doc_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    logger.info(f"Stored document '{filename}' as doc_id={doc_id} for user_id={user_id}")
    return doc_id


def ingest_policy_document(pdf_path: str):
    """Full flow for indexing a policy PDF: Extract -> Chunk -> Check Exists -> Store with Embeddings."""
    filename = os.path.basename(pdf_path)
    
    # 1. Check if already ingested (where user_id is NULL)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM documents WHERE filename = %s AND user_id IS NULL", (filename,))
    exists = cur.fetchone()
    cur.close()
    conn.close()
    
    if exists:
        logger.info(f"Skipping ingestion: policy '{filename}' already exists in DB.")
        return exists[0]

    logger.info(f"New policy detected: {filename}. Starting ingestion...")
    
    # 2. Process
    try:
        pages = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(pages)
        
        if not chunks:
            logger.warning(f"No text found in policy: {filename}")
            return None
            
        total_pages = max((c["page_number"] for c in chunks), default=1)
        
        # 3. Store metadata
        doc_id = store_document(None, filename, total_pages, len(chunks))
        
        # 4. Store with embeddings
        store_policy_chunks(doc_id, chunks)
        
        logger.info(f"✅ Successfully indexed policy: {filename}")
        return doc_id
    except Exception as e:
        logger.error(f"❌ Failed to index policy {filename}: {e}")
        return None


def store_chunks_no_embedding(doc_id: int, chunks: list[dict], table_name: str = "user_documents"):
    """Insert user-uploaded document chunks as raw text only (no embeddings)."""
    conn = get_conn()
    cur  = conn.cursor()
    for chunk in chunks:
        cur.execute(
            f"""
            INSERT INTO {table_name} (document_id, chunk_index, page_number, content)
            VALUES (%s, %s, %s, %s)
            """,
            (doc_id, chunk["chunk_index"], chunk["page_number"], chunk["content"]),
        )
    conn.commit()
    cur.close()
    conn.close()
    logger.info(f"Stored {len(chunks)} raw-text chunks into '{table_name}' for doc_id={doc_id}")


def store_policy_chunks(doc_id: int, chunks: list[dict]):
    """Embed each chunk and store it in the policy_chunks table (used by admin ingestion)."""
    conn = get_conn()
    cur  = conn.cursor()
    for i, chunk in enumerate(chunks):
        logger.info(f"  Embedding policy chunk {i + 1}/{len(chunks)}...")
        embedding = generate_embedding(chunk["content"])
        cur.execute(
            """
            INSERT INTO policy_chunks (document_id, chunk_index, page_number, content, embedding)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (doc_id, chunk["chunk_index"], chunk["page_number"], chunk["content"],
             json.dumps(embedding) if embedding else None),
        )
    conn.commit()
    cur.close()
    conn.close()
    logger.info(f"Stored {len(chunks)} policy chunks with embeddings for doc_id={doc_id}")


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve_user_text(user_id: int) -> str:
    """Fetch all raw text chunks for this user's uploaded documents."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        """
        SELECT d.filename, ud.chunk_index, ud.content
        FROM user_documents ud
        JOIN documents d ON ud.document_id = d.id
        WHERE d.user_id = %s
        ORDER BY d.id, ud.chunk_index
        """,
        (user_id,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        logger.warning(f"No user document chunks found for user_id={user_id}")
        return "[No user documents found.]"

    sections      = []
    current_file  = None
    for filename, _, content in rows:
        if filename != current_file:
            sections.append(f"\n=== Document: {filename} ===")
            current_file = filename
        sections.append(content)

    logger.info(f"Retrieved {len(rows)} chunks for user_id={user_id}")
    return "\n".join(sections)


def retrieve_policy_chunks(query_text: str, top_k: int = 7) -> str:
    """Embed a query and return the most relevant policy chunks via cosine similarity."""
    embedding = generate_embedding(query_text)
    if not embedding:
        return "[Policy retrieval failed: could not generate query embedding.]"

    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        """
        SELECT
            d.filename,
            pc.page_number,
            pc.content,
            (pc.embedding <=> %s::vector) AS distance
        FROM policy_chunks pc
        JOIN documents d ON pc.document_id = d.id
        WHERE pc.embedding IS NOT NULL
        ORDER BY pc.embedding <=> %s::vector
        LIMIT %s;
        """,
        (json.dumps(embedding), json.dumps(embedding), top_k),
    )
    results = cur.fetchall()
    cur.close()
    conn.close()

    if not results:
        logger.warning("No policy chunks matched the query.")
        return "[No relevant policy chunks found.]"

    sections = []
    for filename, page, content, distance in results:
        similarity = round(1 - distance, 4)
        sections.append(
            f"--- Policy Source: {filename} | Page {page} | Similarity: {similarity} ---\n{content}"
        )
    logger.info(f"Retrieved {len(results)} policy chunks for query.")
    return "\n\n".join(sections)
