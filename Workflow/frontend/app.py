import streamlit as st
import requests
import json
import os

# Internal URL: Used by the Python server to talk to the AI service container
AI_SERVICE_INTERNAL_URL = os.getenv("AI_SERVICE_INTERNAL_URL", "http://ai_service:8000")
# External URL: Used by your browser to open links (reports) in a new tab
AI_SERVICE_EXTERNAL_URL = os.getenv("AI_SERVICE_EXTERNAL_URL", "http://localhost:8000")

st.set_page_config(page_title="NBE Loan Application", page_icon="🏦", layout="centered")
st.title("🏦 NBE Loan Application Form")
st.markdown("Please fill out the form below and upload the required documents.")

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
        loan_duration = st.selectbox("Duration (Months)", [12, 24, 36, 48, 60, 72, 84])

    st.markdown("---")
    st.subheader("📄 Document Upload")

    col9, col10 = st.columns(2)
    with col9:
        income_proof_file = st.file_uploader("Income Proof (إثبات دخل)", type=["pdf"])
    with col10:
        utility_bill_file = st.file_uploader("Utility Bill (إيصال مرافق)", type=["pdf"])

    submitted = st.form_submit_button("Evaluate Application", type="primary", use_container_width=True)

if submitted:
    if not income_proof_file and not utility_bill_file:
        st.warning("Please upload at least one PDF document to proceed.")
        st.stop()

    form_data = {
        "personal_info": {
            "full_name":   full_name,
            "national_id": national_id,
            "phone":       phone,
            "email":       email,
        },
        "employment_details": {
            "employment_profile": employment_type,
            "monthly_income_egp": monthly_income,
        },
        "loan_details": {
            "requested_amount_egp": loan_amount,
            "duration_months":      loan_duration,
        },
    }

    with st.spinner("⏳ Evaluating application... this may take a minute."):
        try:
            # Build multipart request
            actual_files = []
            if income_proof_file:
                actual_files.append(
                    ("files", (income_proof_file.name, income_proof_file.getbuffer(), "application/pdf"))
                )
            if utility_bill_file:
                actual_files.append(
                    ("files", (utility_bill_file.name, utility_bill_file.getbuffer(), "application/pdf"))
                )

            # Use INTERNAL URL for the API call
            response = requests.post(
                f"{AI_SERVICE_INTERNAL_URL}/evaluate",
                data={"form_data": json.dumps(form_data)},
                files=actual_files,
                timeout=360,
            )
            response.raise_for_status()
            result = response.json()

        except requests.exceptions.ConnectionError:
            st.error(f"❌ Could not connect to AI Service at `{AI_SERVICE_INTERNAL_URL}`. Is the container running?")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

    st.success("✅ Application evaluated successfully!")

    decision = result.get("structured_decision", {})
    st.subheader("📋 Evaluation Results")
    st.json(decision)

    # ── Open full HTML report in a new browser tab ────────────────────────────
    report_filename = result.get("report_filename")
    if report_filename:
        # Use EXTERNAL URL for the browser link
        report_url = f"{AI_SERVICE_EXTERNAL_URL}/report/{report_filename}"
        st.markdown("---")
        st.link_button(
            label="🔗 Open Full HTML Report in New Tab",
            url=report_url,
            use_container_width=True,
        )
