import os
import json
import requests
from pydantic import BaseModel
import pdf_vectorize

OLLAMA_URL = "http://172.24.77.77:11434"
MODEL = "gemma4:e4b"

class LoanApprovalDecision(BaseModel):
    is_approved: bool
    confidence_score: int
    decision_reasoning: str
    risk_factors: list[str]
    recommended_loan_amount: float

def get_conn():
    return pdf_vectorize.get_conn()

def insert_or_update_user(json_data):
    conn = get_conn()
    cur = conn.cursor()
    info = json_data.get("personal_info", {})
    national_id = info.get("national_id")
    full_name = info.get("full_name")
    phone = info.get("phone")
    email = info.get("email")

    cur.execute("SELECT id FROM users WHERE national_id = %s", (national_id,))
    res = cur.fetchone()
    if res:
        user_id = res[0]
        cur.execute("""
            UPDATE users SET full_name=%s, phone=%s, email=%s WHERE id=%s
        """, (full_name, phone, email, user_id))
    else:
        cur.execute("""
            INSERT INTO users (full_name, national_id, phone, email)
            VALUES (%s, %s, %s, %s) RETURNING id
        """, (full_name, national_id, phone, email))
        user_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return user_id

def ingest_pdfs_for_user(user_id, pdf_paths):
    conn = get_conn()
    cur = conn.cursor()
    for pdf_path in pdf_paths:
        pages = pdf_vectorize.extract_text_from_pdf(pdf_path)
        chunks = pdf_vectorize.chunk_text(pages)
        
        cur.execute(
            "INSERT INTO documents (user_id, filename, total_pages, total_chunks) VALUES (%s, %s, %s, %s) RETURNING id",
            (user_id, os.path.basename(pdf_path), max((c["page_number"] for c in chunks), default=1) if chunks else 1, len(chunks)),
        )
        doc_id = cur.fetchone()[0]
        pdf_vectorize.store_chunks(doc_id, chunks)
    conn.commit()
    cur.close()
    conn.close()

def retrieve_user_chunks(user_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT d.filename, dc.content
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.id
        WHERE d.user_id = %s
    """, (user_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    contexts = []
    for filename, content in rows:
        contexts.append(f"--- Document: {filename} ---\n{content}")
    return "\n\n".join(contexts)

def generate_report(json_data, decision: LoanApprovalDecision):
    decision_text = "APPROVED" if decision.is_approved else "REJECTED"
    decision_class = "approved" if decision.is_approved else "rejected"
    
    risk_factors_html = "".join([f"<li>{factor}</li>" for factor in decision.risk_factors])
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
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
            <h2>Applicant: {json_data.get('personal_info', {{}}).get('full_name')}</h2>
            <p><strong>National ID:</strong> {json_data.get('personal_info', {{}}).get('national_id')}</p>
            <p><strong>Requested Amount:</strong> {json_data.get('loan_details', {{}}).get('requested_amount_egp')} EGP</p>
            <hr>
            <h3>Decision: <span class="{decision_class}">{decision_text}</span></h3>
            <p><strong>Confidence Score:</strong> {decision.confidence_score}%</p>
            <p><strong>Recommended Amount:</strong> {decision.recommended_loan_amount} EGP</p>
            <h4>Reasoning:</h4>
            <p>{decision.decision_reasoning}</p>
            <h4>Risk Factors:</h4>
            <ul>
                {risk_factors_html}
            </ul>
        </div>
    </body>
    </html>
    """
    report_path = f"report_{json_data.get('personal_info', {{}}).get('national_id', 'unknown')}.html"
    with open(report_path, "w") as f:
        f.write(html_content)
    return report_path

def evaluate_loan_application(json_data, pdf_paths):
    user_id = insert_or_update_user(json_data)
    ingest_pdfs_for_user(user_id, pdf_paths)
    context = retrieve_user_chunks(user_id)
    
    system_prompt = """You are an NBE (National Bank of Egypt) Credit Analyst. 
Evaluate the loan application based on the provided JSON data and the extracted document text.
You MUST reply withONLY a strictly valid JSON object matching the following structure and nothing else:
{
  "is_approved": true or false,
  "confidence_score": integer between 0 and 100,
  "decision_reasoning": "string with the detailed evaluation here",
  "risk_factors": ["list", "of", "strings"],
  "recommended_loan_amount": float
}
"""
    
    user_content = f"JSON Form Data:\n{json.dumps(json_data, indent=2)}\n\nExtracted Document Context:\n{context}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "format": "json"
    }

    try:
        response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
        response.raise_for_status()
        reply_text = response.json().get("message", {}).get("content", "")
        
        reply_dict = json.loads(reply_text)
        decision = LoanApprovalDecision(**reply_dict)
    except Exception as e:
        print("Error evaluating loan:", e)
        decision = LoanApprovalDecision(
            is_approved=False,
            confidence_score=0,
            decision_reasoning=f"Error evaluating loan: {str(e)}",
            risk_factors=["System Error"],
            recommended_loan_amount=0.0
        )
        reply_dict = decision.dict()
        
    report_path = generate_report(json_data, decision)
    
    return {
        "structured_decision": reply_dict,
        "report_path": report_path
    }
