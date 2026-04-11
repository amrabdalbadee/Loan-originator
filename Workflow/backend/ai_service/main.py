"""
AI Service — FastAPI Application
=================================
Receives evaluation requests from the Frontend.
Orchestrates the full pipeline via AI_logic and serves the HTML report.
"""

import os
import sys
import json
import logging
from typing import List
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
import requests

import AI_logic

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ai_service] %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

VECTOR_SERVICE_URL = os.getenv("VECTOR_SERVICE_URL", "http://vector_service:8001")
REPORT_DIR         = os.getenv("REPORT_DIR", "/tmp/reports")

app = FastAPI(title="AI Service", version="1.0.0")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "ai_service"}


# ── Main Evaluation Endpoint ──────────────────────────────────────────────────
@app.post("/evaluate")
async def evaluate(
    form_data: str              = Form(...),   # JSON string of the applicant form
    files:     List[UploadFile] = File(default=[]),
):
    """
    Full three-input loan evaluation:
      1. JSON form data → fed directly to LLM
      2. Uploaded PDFs  → sent to Vector Service as byte streams, retrieved as raw text
      3. Bank policy    → retrieved from Vector Service via cosine similarity
    Returns structured decision + report URL for the browser to open in a new tab.
    """
    try:
        json_data = json.loads(form_data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON in form_data field.")

    national_id = json_data.get("personal_info", {}).get("national_id", "unknown")
    logger.info(f"=== Evaluation started for national_id={national_id} ===")

    # ── Step 1: Upsert user ───────────────────────────────────────────────────
    try:
        user_id = AI_logic.upsert_user(json_data)
    except Exception as e:
        logger.error(f"Failed to upsert user: {e}")
        raise HTTPException(status_code=502, detail=f"Vector service error (upsert): {e}")

    # ── Step 2: Ingest uploaded PDFs into Vector Service ─────────────────────
    if files:
        logger.info(f"Forwarding {len(files)} file(s) to Vector Service for ingestion")
        try:
            multipart_files = []
            file_contents   = []
            for f in files:
                content = await f.read()
                file_contents.append((f.filename, content))
                multipart_files.append(
                    ("files", (f.filename, content, "application/pdf"))
                )

            resp = requests.post(
                f"{VECTOR_SERVICE_URL}/ingest/documents",
                data={"user_id": user_id},
                files=multipart_files,
                timeout=120,
            )
            resp.raise_for_status()
            logger.info(f"Ingestion response: {resp.json()}")
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            raise HTTPException(status_code=502, detail=f"Vector service error (ingest): {e}")
    else:
        logger.warning("No files uploaded — proceeding without user document context.")

    # ── Step 3: Retrieve all user doc chunks ──────────────────────────────────
    try:
        user_doc_context = AI_logic.get_user_doc_context(user_id)
    except Exception as e:
        logger.error(f"Failed to retrieve user chunks: {e}")
        user_doc_context = "[Could not retrieve user documents.]"

    # ── Step 4: Retrieve relevant policy chunks ───────────────────────────────
    try:
        policy_context = AI_logic.get_policy_context(json_data)
    except Exception as e:
        logger.error(f"Failed to retrieve policy context: {e}")
        policy_context = "[Could not retrieve policy rules.]"

    # ── Step 5: Call LLM ──────────────────────────────────────────────────────
    try:
        decision = AI_logic.call_llm(json_data, user_doc_context, policy_context)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        decision = {
            "is_approved":             False,
            "confidence_score":        0,
            "decision_reasoning":      f"System error during LLM evaluation: {str(e)}",
            "risk_factors":            ["System Error"],
            "recommended_loan_amount": 0.0,
        }

    # ── Step 6: Generate HTML report ──────────────────────────────────────────
    report_filename = AI_logic.generate_report(json_data, decision, REPORT_DIR)
    logger.info(f"=== Evaluation complete for national_id={national_id} ===")

    return {
        "structured_decision": decision,
        "report_filename":     report_filename,
    }


# ── Report Serving Endpoint ───────────────────────────────────────────────────
@app.get("/report/{filename}", response_class=HTMLResponse)
def serve_report(filename: str):
    """Serve a generated HTML credit report — opened in a new browser tab."""
    # Basic path traversal guard
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    report_path = os.path.join(REPORT_DIR, filename)
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail=f"Report '{filename}' not found.")
    with open(report_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
