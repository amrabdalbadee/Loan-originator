import streamlit as st
import requests
import json
import os

# Internal URL: Used by the Python server to talk to the AI service container
AI_SERVICE_INTERNAL_URL = os.getenv("AI_SERVICE_INTERNAL_URL", "http://ai_service:8000")
# External URL: Used by your browser to open links (reports) in a new tab
AI_SERVICE_EXTERNAL_URL = os.getenv("AI_SERVICE_EXTERNAL_URL", "http://localhost:8000")
# ID Extractor URL
ID_EXTRACTOR_URL = os.getenv("ID_EXTRACTOR_URL", "http://id_extractor_service:8002")

st.set_page_config(page_title="NBE Loan Application", page_icon="🏦", layout="centered")
st.title("🏦 NBE Loan Application Form")
st.markdown("Please fill out the form below and upload the required documents.")

if 'full_name' not in st.session_state:
    st.session_state.full_name = ""
if 'national_id' not in st.session_state:
    st.session_state.national_id = ""

st.subheader("🪪 Step 1: Identity Extraction (Optional)")
st.markdown("Upload your National ID or Passport to automatically pre-fill your application.")

doc_type = st.radio("Select Identity Document:", ["National ID", "Passport"], horizontal=True)

if doc_type == "National ID":
    id_col1, id_col2 = st.columns(2)
    with id_col1:
        id_front_file = st.file_uploader("Upload National ID (Front)", type=['jpg', 'jpeg', 'png'])
    with id_col2:
        id_back_file = st.file_uploader("Upload National ID (Back)", type=['jpg', 'jpeg', 'png'])
    
    if st.button("Extract ID Data"):
        if id_front_file and id_back_file:
            with st.spinner("Extracting..."):
                try:
                    files = [
                        ("id_front", (id_front_file.name, id_front_file.getvalue(), id_front_file.type)),
                        ("id_back", (id_back_file.name, id_back_file.getvalue(), id_back_file.type))
                    ]
                    data = {"doc_type": "National ID"}
                    resp = requests.post(f"{ID_EXTRACTOR_URL}/extract", data=data, files=files)
                    resp.raise_for_status()
                    result = resp.json()
                    
                    if not result.get("is_valid", False):
                        st.error(f"⚠️ Document appears to be expired! (Expiry: {result.get('expiry_date')})")
                    else:
                        st.success("✅ Document is valid.")
                        
                    if result.get("full_name_arabic"):
                        st.session_state.full_name = result["full_name_arabic"]
                    if result.get("national_id_number"):
                        st.session_state.national_id = result["national_id_number"]
                        
                    if result.get("is_valid", False):
                        st.rerun()
                except Exception as e:
                    st.error(f"Error during extraction: {e}")
        else:
            st.warning("Please upload both front and back images.")

else:
    passport_file = st.file_uploader("Upload Passport Image", type=['jpg', 'jpeg', 'png'])
    if st.button("Extract Passport Data"):
        if passport_file:
            with st.spinner("Extracting..."):
                try:
                    files = [
                        ("passport", (passport_file.name, passport_file.getvalue(), passport_file.type))
                    ]
                    data = {"doc_type": "Passport"}
                    resp = requests.post(f"{ID_EXTRACTOR_URL}/extract", data=data, files=files)
                    resp.raise_for_status()
                    result = resp.json()
                    
                    if not result.get("is_valid", False):
                        st.error(f"⚠️ Document appears to be expired! (Expiry: {result.get('expiry_date')})")
                    else:
                        st.success("✅ Document is valid.")
                        
                    name = result.get("full_name_arabic") or result.get("full_name_latin")
                    if name:
                        st.session_state.full_name = name
                    if result.get("national_id_number"):
                        st.session_state.national_id = result["national_id_number"]
                        
                    if result.get("is_valid", False):
                        st.rerun()
                except Exception as e:
                    st.error(f"Error during extraction: {e}")
        else:
            st.warning("Please upload the passport image.")

st.markdown("---")

with st.form("loan_form"):
    st.subheader("👤 Personal Information")

    col1, col2 = st.columns(2)
    with col1:
        full_name = st.text_input("Applicant Full Name", value=st.session_state.full_name, placeholder="e.g. Ahmed Mohamed Hassan")
    with col2:
        national_id = st.text_input("National ID (14 digits)", value=st.session_state.national_id, max_chars=14, placeholder="29001011234567")

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

st.markdown("---")
st.subheader("✍️ Step 4: Signature Verification")
st.markdown("Upload a document to verify its signatures against the stored reference.")

doc_sig_file = st.file_uploader("Upload Document for Signature Extraction", type=['jpg', 'jpeg', 'png'], key="doc_sig")

if st.button("Verify Signature", type="primary", use_container_width=True):
    if doc_sig_file:
        with st.spinner("✍️ Verifying signatures..."):
            try:
                SIGNATURE_SERVICE_URL = os.getenv("SIGNATURE_SERVICE_URL", "http://signature_service:8003")
                files = {
                    "document": (doc_sig_file.name, doc_sig_file.getvalue(), doc_sig_file.type)
                }
                # Hardcoded threshold as requested
                data = {"threshold": 0.85}
                resp = requests.post(f"{SIGNATURE_SERVICE_URL}/verify", files=files, data=data)
                resp.raise_for_status()
                result = resp.json()
                
                if result.get("status") == "success":
                    detections = result.get("detections", [])
                    if not detections:
                        st.warning("No signatures detected in the document.")
                    else:
                        st.success(f"Detections completed. Found {len(detections)} signature(s).")
                        
                        cols = st.columns(len(detections) if len(detections) < 4 else 4)
                        for i, det in enumerate(detections):
                            with cols[i % 4]:
                                import base64
                                from PIL import Image
                                import io
                                
                                # Decode the base64 crop
                                img_data = base64.b64decode(det['crop_base64'])
                                img = Image.open(io.BytesIO(img_data))
                                
                                st.image(img, caption=f"Detection {i+1}")
                                status = "✅ GENUINE" if det['is_genuine'] else "❌ FORGED"
                                st.markdown(f"**{status}**")
                                st.markdown(f"Score: `{det['similarity']:.4f}`")
                else:
                    st.error("Signature verification failed.")
            except Exception as e:
                st.error(f"Error during signature verification: {e}")
    else:
        st.warning("Please upload a document to verify.")
