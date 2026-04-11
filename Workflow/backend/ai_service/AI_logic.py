"""
AI Service — Core Evaluation Logic
====================================
Orchestrates the full three-input loan evaluation pipeline.
Communicates with the Vector Service via HTTP (no direct DB access).
"""

import os
import json
import logging
import requests

logger = logging.getLogger(__name__)

OLLAMA_URL         = os.getenv("OLLAMA_URL",         "http://172.24.77.77:11434")
MODEL              = os.getenv("OLLAMA_MODEL",        "gemma4:e4b")
VECTOR_SERVICE_URL = os.getenv("VECTOR_SERVICE_URL",  "http://vector_service:8001")


# ── Step 1: Upsert applicant via Vector Service ───────────────────────────────
def upsert_user(json_data: dict) -> int:
    info = json_data.get("personal_info", {})
    payload = {
        "full_name":   info.get("full_name"),
        "national_id": info.get("national_id"),
        "phone":       info.get("phone"),
        "email":       info.get("email"),
    }
    logger.info(f"Calling Vector Service to upsert user: {info.get('national_id')}")
    resp = requests.post(f"{VECTOR_SERVICE_URL}/upsert/user", json=payload, timeout=30)
    resp.raise_for_status()
    user_id = resp.json()["user_id"]
    logger.info(f"User upserted → user_id={user_id}")
    return user_id


# ── Step 2: Retrieve user document text via Vector Service ────────────────────
def get_user_doc_context(user_id: int) -> str:
    logger.info(f"Retrieving user document chunks for user_id={user_id}")
    resp = requests.get(f"{VECTOR_SERVICE_URL}/retrieve/user/{user_id}", timeout=30)
    resp.raise_for_status()
    return resp.json()["text"]


# ── Step 3: Retrieve policy context via vector similarity ─────────────────────
def get_policy_context(json_data: dict, top_k: int = 7) -> str:
    employment = json_data.get("employment_details", {})
    loan       = json_data.get("loan_details", {})

    query = (
        f"Loan application: {employment.get('employment_profile', '')} applicant, "
        f"monthly income {employment.get('monthly_income_egp', '')} EGP, "
        f"requested amount {loan.get('requested_amount_egp', '')} EGP, "
        f"duration {loan.get('duration_months', '')} months. "
        f"What are the NBE eligibility requirements, income ratio limits, "
        f"and documentation rules for this type of applicant?"
    )
    logger.info(f"Retrieving policy context (top_k={top_k})")
    resp = requests.post(
        f"{VECTOR_SERVICE_URL}/retrieve/policy",
        json={"query": query, "top_k": top_k},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["text"]


# ── Step 4: Call Ollama LLM ───────────────────────────────────────────────────
def call_llm(json_data: dict, user_doc_context: str, policy_context: str) -> dict:
    system_prompt = """You are a senior credit analyst at NBE (National Bank of Egypt).

Your task is to evaluate a loan application for compliance and creditworthiness.
You will receive three distinct inputs:

1. APPLICANT FORM DATA — structured JSON submitted by the applicant.
2. APPLICANT DOCUMENTS — raw extracted text from the applicant's uploaded Income Proof and Utility Bill PDFs.
3. BANK POLICY — the most relevant NBE lending policy rules retrieved for this application type.

Your job:
- Verify the information in the form data is consistent with the uploaded documents.
- Check the application against the retrieved bank policy rules.
- Identify any risk factors, missing information, or policy violations.
- Produce a final credit decision.

You MUST reply with ONLY a strictly valid JSON object matching this exact structure and nothing else:
{
  "is_approved": true or false,
  "confidence_score": integer between 0 and 100,
  "decision_reasoning": "detailed multi-sentence evaluation",
  "risk_factors": ["list", "of", "identified", "risks"],
  "recommended_loan_amount": float
}
"""

    user_message = f"""=== INPUT 1: APPLICANT FORM DATA (JSON) ===
{json.dumps(json_data, indent=2, ensure_ascii=False)}

=== INPUT 2: APPLICANT UPLOADED DOCUMENTS (Raw Text) ===
{user_doc_context}

=== INPUT 3: RETRIEVED BANK POLICY RULES ===
{policy_context}
"""

    payload = {
        "model":    MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "stream": False,
        "format": "json",
    }

    logger.info(f"Calling Ollama ({MODEL}) at {OLLAMA_URL}")
    response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
    response.raise_for_status()
    reply_text = response.json().get("message", {}).get("content", "")
    reply_dict = json.loads(reply_text)
    logger.info(f"LLM decision: is_approved={reply_dict.get('is_approved')} confidence={reply_dict.get('confidence_score')}")
    return reply_dict

float
# ── Step 5: Generate HTML report ──────────────────────────────────────────────
def generate_report(json_data: dict, decision: dict, report_dir: str) -> str:
    """Build an HTML credit report and write it to report_dir. Returns the file path."""
    is_approved    = decision.get("is_approved", False)
    decision_text  = "APPROVED" if is_approved else "REJECTED"
    decision_class = "approved" if is_approved else "rejected"
    risk_factors_html = "".join(
        [f"<li>{factor}</li>" for factor in decision.get("risk_factors", [])]
    )

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Loan Credit Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; color: #333; margin: 40px; }}
        h1, h2 {{ color: #004b87; }}
        .approved {{ color: green; font-weight: bold; }}
        .rejected {{ color: red; font-weight: bold; }}
        .card {{ border: 1px solid #ccc; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        ul {{ line-height: 1.6; }}
    </style>
</head>
<body>
    <h1>Credit Decision Report</h1>
    <div class="card">
        <h2>Applicant: {json_data.get('personal_info', {}).get('full_name')}</h2>
        <p><strong>National ID:</strong> {json_data.get('personal_info', {}).get('national_id')}</p>
        <p><strong>Requested Amount:</strong> {json_data.get('loan_details', {}).get('requested_amount_egp')} EGP</p>
        <hr>
        <h3>Decision: <span class="{decision_class}">{decision_text}</span></h3>
        <p><strong>Confidence Score:</strong> {decision.get('confidence_score')}%</p>
        <p><strong>Recommended Amount:</strong> {decision.get('recommended_loan_amount')} EGP</p>
        <h4>Reasoning:</h4>
        <p>{decision.get('decision_refloatasoning')}</p>
        <h4>Risk Factors:</h4>
        <ul>{risk_factors_html}</ul>
    </div>
</body>
</html>"""

    national_id = json_data.get("personal_info", {}).get("national_id", "unknown")
    filename    = f"report_{national_id}.html"
    path        = os.path.join(report_dir, filename)
    os.makedirs(report_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Report written to {path}")
    return filename
