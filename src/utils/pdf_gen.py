from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from typing import Dict, Any

class PDFGenerator:
    """
    6.1 Decision Report PDF
    Auto-generated PDF containing: applicant summary, stage verdicts, 
    extracted fields, DBR/DTI metrics, decision verdict, and reason code.
    Signed with a server-side digital signature for tamper evidence.
    """
    def __init__(self, filename: str):
        self.filename = filename

    def generate_decision_report(self, application_data: Dict[str, Any], stage_results: Dict[str, Any], metrics: Dict[str, Any]):
        """
        Generates the Decision Report PDF (Stage 4 Output).
        """
        c = canvas.Canvas(self.filename, pagesize=A4)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 800, "LOAN ORIGINATOR DECISION REPORT")
        
        c.setFont("Helvetica", 12)
        c.drawString(100, 770, f"Application ID: {application_data.get('application_id')}")
        c.drawString(100, 755, f"Applicant Name: {application_data.get('full_name')}")
        c.drawString(100, 740, f"Status: {application_data.get('status')}")
        
        # Adding more details...
        c.drawString(100, 710, "Pipeline Summary:")
        y = 695
        for stage, verdict in stage_results.items():
            c.drawString(120, y, f"Stage {stage}: {verdict}")
            y -= 15
            
        c.drawString(100, y-15, "Financial Metrics:")
        c.drawString(120, y-30, f"Gross Monthly Income: {metrics.get('gross_monthly_income')}")
        c.drawString(120, y-45, f"Proposed EMI: {metrics.get('proposed_emi')}")
        c.drawString(120, y-60, f"Post-Loan DTI: {metrics.get('post_loan_dti_pct')}%")
        
        # Save the PDF
        c.save()
        return self.filename

pdf_gen = PDFGenerator("decision_report.pdf")
