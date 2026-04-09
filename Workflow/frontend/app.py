import streamlit as st
import json

st.set_page_config(page_title="NBE Loan Application", page_icon="🏦", layout="centered")

st.title("🏦 NBE Loan Application Form")
st.markdown("Please fill out the form below. Once submitted, the output will be displayed in JSON format.")

with st.form("loan_form"):
    st.subheader("👤 Personal Information")
    
    col1, col2 = st.columns(2)
    with col1:
        full_name = st.text_input("Applicant Full Name", placeholder="e.g. Ahmed Mohamed Hassan")
    with col2:
        national_id = st.text_input("National ID (14 digits)", max_chars=14, placeholder="29001011234567")
        
    col3, col4 = st.columns(2)
    with col3:
        phone = st.text_input("Phone Number", placeholder="010XXXXXXXX")
    with col4:
        email = st.text_input("Email Address", placeholder="ahmed@example.com")
        
    st.markdown("---")
    st.subheader("💼 Employment Details")
    
    col5, col6 = st.columns(2)
    with col5:
        employment_type = st.selectbox(
            "Employment Profile",
            ["Employee / Salary Transfer", "Business Owner / Professional"]
        )
    with col6:
        monthly_income = st.number_input("Monthly Net Income (EGP)", min_value=0, step=500, value=15000)
        
    st.markdown("---")
    st.subheader("💰 Loan Details")
    
    col7, col8 = st.columns(2)
    with col7:
        loan_amount = st.number_input("Requested Loan Amount (EGP)", min_value=1000, step=5000, value=100000)
    with col8:
        loan_duration = st.selectbox(
            "Duration (Months)",
            [12, 24, 36, 48, 60, 72, 84]
        )
        
    st.markdown("---")
    st.subheader("📄 Document Upload")
    
    col9, col10 = st.columns(2)
    with col9:
        income_proof_file = st.file_uploader("Income Proof (إثبات دخل)", type=["pdf"])
    with col10:
        utility_bill_file = st.file_uploader("Utility Bill (إيصال مرافق)", type=["pdf"])
        
    # The submit button
    submitted = st.form_submit_button("Generate JSON Payload", type="primary", use_container_width=True)

if submitted:
    # Build dictionary
    data = {
        "personal_info": {
            "full_name": full_name,
            "national_id": national_id,
            "phone": phone,
            "email": email
        },
        "employment_details": {
            "employment_profile": employment_type,
            "monthly_income_egp": monthly_income
        },
        "loan_details": {
            "requested_amount_egp": loan_amount,
            "duration_months": loan_duration
        },
        "documents": {
            "income_proof": income_proof_file.name if income_proof_file else None,
            "utility_bill": utility_bill_file.name if utility_bill_file else None
        }
    }
    
    import tempfile
    import os
    import sys
    import streamlit.components.v1 as components
    
    # Add Workflow folder to path to import AI_logic
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
        
    try:
        import AI_logic
    except ImportError:
        st.error("Failed to import AI_logic.")
        st.stop()
        
    pdf_paths = []
    
    if income_proof_file or utility_bill_file:
        with st.spinner("Processing Documents and Evaluating Application..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                if income_proof_file:
                    ip_path = os.path.join(temp_dir, income_proof_file.name)
                    with open(ip_path, "wb") as f:
                        f.write(income_proof_file.getbuffer())
                    pdf_paths.append(ip_path)
                    
                if utility_bill_file:
                    ub_path = os.path.join(temp_dir, utility_bill_file.name)
                    with open(ub_path, "wb") as f:
                        f.write(utility_bill_file.getbuffer())
                    pdf_paths.append(ub_path)
                    
                result = AI_logic.evaluate_loan_application(data, pdf_paths)
                
            st.success("Form submitted and evaluated successfully!")
            st.subheader("Structured Evaluation")
            st.json(result["structured_decision"])
            
            report_path = result.get("report_path")
            if report_path and os.path.exists(report_path):
                st.markdown(f"**HTML Credit Report Saved At:** `{report_path}`")
                try:
                    with open(report_path, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    components.html(html_content, height=600, scrolling=True)
                except Exception as e:
                    st.error(f"Could not load HTML report: {e}")
    else:
        st.warning("Please upload at least one PDF document to evaluate.")
        st.json(data)
