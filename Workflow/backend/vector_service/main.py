"""
Vector Service — FastAPI Application
=====================================
Handles all PDF ingestion, text storage, and vector retrieval operations.
Exposes HTTP endpoints consumed by the AI Service.
"""

import os
import sys
import logging
import tempfile
from typing import List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel

import pdf_vectorize

# ── Logging (stdout so Docker captures it) ────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [vector_service] %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vector Service", version="1.0.0")


# ── Models ────────────────────────────────────────────────────────────────────
class UserUpsertRequest(BaseModel):
    full_name:   str
    national_id: str
    phone:       str | None = None
    email:       str | None = None


class PolicyQueryRequest(BaseModel):
    query:  str
    top_k:  int = 7


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "vector_service"}


# ── User Upsert ───────────────────────────────────────────────────────────────
@app.post("/upsert/user")
def upsert_user(data: UserUpsertRequest):
    """Insert or update an applicant in the users table. Returns user_id."""
    logger.info(f"Upserting user national_id={data.national_id}")
    conn = pdf_vectorize.get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("SELECT id FROM users WHERE national_id = %s", (data.national_id,))
        res = cur.fetchone()
        if res:
            user_id = res[0]
            cur.execute(
                "UPDATE users SET full_name=%s, phone=%s, email=%s WHERE id=%s",
                (data.full_name, data.phone, data.email, user_id),
            )
            logger.info(f"Updated existing user id={user_id}")
        else:
            cur.execute(
                "INSERT INTO users (full_name, national_id, phone, email) VALUES (%s, %s, %s, %s) RETURNING id",
                (data.full_name, data.national_id, data.phone, data.email),
            )
            user_id = cur.fetchone()[0]
            logger.info(f"Created new user id={user_id}")
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to upsert user: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()
    return {"user_id": user_id}


# ── User Document Ingestion ───────────────────────────────────────────────────
@app.post("/ingest/documents")
async def ingest_documents(
    user_id: int = Form(...),
    files:   List[UploadFile] = File(...),
):
    """
    Receive PDF files as byte streams, extract text, chunk, and store as
    raw text in user_documents (no embeddings generated).
    """
    logger.info(f"Ingesting {len(files)} document(s) for user_id={user_id}")
    ingested = []
    for upload in files:
        file_bytes = await upload.read()
        logger.info(f"  Processing file: {upload.filename} ({len(file_bytes)} bytes)")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            pages  = pdf_vectorize.extract_text_from_pdf(tmp_path)
            chunks = pdf_vectorize.chunk_text(pages)

            total_pages  = max((c["page_number"] for c in chunks), default=1) if chunks else 1
            total_chunks = len(chunks)

            doc_id = pdf_vectorize.store_document(user_id, upload.filename, total_pages, total_chunks)
            pdf_vectorize.store_chunks_no_embedding(doc_id, chunks)

            logger.info(f"  Stored {total_chunks} chunks for '{upload.filename}' (doc_id={doc_id})")
            ingested.append({"filename": upload.filename, "doc_id": doc_id, "chunks": total_chunks})
        except Exception as e:
            logger.error(f"  Failed to ingest {upload.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to ingest {upload.filename}: {e}")
        finally:
            os.unlink(tmp_path)

    return {"status": "ok", "ingested": ingested}


# ── User Chunk Retrieval ──────────────────────────────────────────────────────
@app.get("/retrieve/user/{user_id}")
def retrieve_user_chunks(user_id: int):
    """Return all raw text chunks for a given user's uploaded documents."""
    logger.info(f"Retrieving user document chunks for user_id={user_id}")
    text = pdf_vectorize.retrieve_user_text(user_id)
    return {"text": text}


# ── Policy Chunk Retrieval (Vector Search) ────────────────────────────────────
@app.post("/retrieve/policy")
def retrieve_policy(request: PolicyQueryRequest):
    """
    Embed the query and return the top-k most relevant policy chunks
    from the policy_chunks table via cosine similarity search.
    """
    logger.info(f"Policy retrieval | top_k={request.top_k} | query: {request.query[:80]}...")
    text = pdf_vectorize.retrieve_policy_chunks(request.query, top_k=request.top_k)
    return {"text": text}
