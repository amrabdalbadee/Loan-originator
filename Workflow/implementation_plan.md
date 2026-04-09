# Integrate UI JSON and Uploaded Documents with Gemma

This plan outlines how to connect the Streamlit `app.py` with the Ollama chat script, renaming it to `AI_logic.py`, and implementing the vectorization -> LLM pipeline with a robust user-centric schema and structured reporting.

## User Review Required
> [!IMPORTANT]
> I have incorporated your reference code's design! The AI evaluation will now return a beautifully structured output and generate an interactive HTML report similar to your provided Stage 3 logic. 
> The LLM will use a highly tuned system prompt specifically designed for evaluating the applicant based on the uploaded documents and JSON form fields.
> Do we have the green light to execute this plan?

## Proposed Changes

### Database Schema Update
#### [MODIFY] [init-vectordb.sql](file:///home/abdelrahman-click/Desktop/IPULSE/Loan-originator/Workflow/init-vectordb.sql)
- Create a `users` table to track applicants (e.g., `id`, `full_name`, `national_id` (UNIQUE), `phone`, `email`).
- Modify the `documents` table to include a `user_id` foreign key referencing the `users` table.
- This allows the system to easily filter embedded chunks tied strictly to the current applicant during the Gemma evaluation phase.

### Core Logic
#### [NEW] [AI_logic.py](file:///home/abdelrahman-click/Desktop/IPULSE/Loan-originator/Workflow/AI_logic.py)
- Rename `ollama_chat.py` to `AI_logic.py`.
- Refactor to include a primary `evaluate_loan_application(json_data, pdf_paths)` workflow.
- **Database Ingestion**: Insert/update the applicant into the `users` table via `national_id`. Store uploaded PDFs into `documents` linked to their `user_id`. Extract text, generate embeddings natively, and store chunks.
- **Context Retrieval**: Retrieve chunks from the DB filtered by `user_id`.
- **Prompt & LLM Calling**: 
  - Write a robust System Prompt outlining the Persona (NBE Credit Analyst), the rules of evaluation, and mandating a structured JSON response (matching the Pydantic models from your reference code).
  - Construct the `messages` array combining the JSON form data and the extracted document contexts.
  - Call the `gemma4:e4b` model on your local Ollama server.
- **Structured Output**: Parse the LLM's raw response into structured Pydantic objects for safety.
- **HTML Report generation**: Automatically generate the detailed loan decision HTML file from the model's output (following your beautiful reference template UI style).

#### [DELETE] [ollama_chat.py](file:///home/abdelrahman-click/Desktop/IPULSE/Loan-originator/Workflow/ollama_chat.py)
- Renamed and fully refactored into `AI_logic.py`.

### Frontend Interface
#### [MODIFY] [app.py](file:///home/abdelrahman-click/Desktop/IPULSE/Loan-originator/Workflow/frontend/app.py)
- Accept and validate the PDF uploads for 'Income Proof' and 'Utility Bill'.
- Temporarily save these files and pass their exact local paths to `evaluate_loan_application` alongside the fully compiled JSON form object.
- Display a neat Streamlit loading UI.
- Render the final output, showing JSON statistics alongside an embedded view or link to the generated HTML credit report.

## Verification Plan
### Automations
- Validate database table creation/integrity through schema introspection.
- Validate that Gemma responds effectively to structured formatting directives.

### Manual Verification
- Run the Streamlit web app end-to-end and execute a mock upload to confirm all components (Embeddings -> DB -> Retrieval -> Gemma -> HTML Report) fire seamlessly.
