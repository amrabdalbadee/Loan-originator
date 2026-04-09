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
        }
    }
    
    st.success("Form submitted successfully! Here is your JSON payload:")
    
    # Display the JSON output nicely
    st.json(data)
