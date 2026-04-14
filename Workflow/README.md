# 🔄 Loan Originator Workflow

An end-to-end, microservices-based AI pipeline for automated loan application processing, identity verification, and signature validation.

## 🚀 Overview

The **Workflow** is the core implementation of the Loan Originator AI Platform. It orchestrates multiple specialized AI services to process loan applications from document upload to final credit decision.

### Key Workflows
1.  **🪪 Smart Identity Extraction**: Automatic pre-filling of applicant data using vision-language models (Qwen2.5-VL) from ID cards and Passports.
2.  **🤖 Intelligent RAG Evaluation**: A three-input assessment engine that cross-references:
    -   **Applicant Form Data**: Self-reported numbers.
    -   **extracted Documents**: Fact-checked text from Income Proof/Utility PDFs.
    -   **Banking Policy**: Relevant rules retrieved from embedded bank manuals.
3.  **✍️ Signature Verification**: Automated signature detection (YOLOv11) and verification (Siamese Network) against stored reference benchmarks.
4.  **📊 Structured Reporting**: Professional HTML credit reports that aggregate decisions, confidence scores, and risk analysis.

---

## 🏗️ Architecture

The project follows a **Microservices Architecture** orchestrated via Docker Compose:

| Service | Technology | Description |
| :--- | :--- | :--- |
| **`frontend`** | Streamlit | The user interface for data entry and document uploads. |
| **`ai_service`** | FastAPI + Ollama | Handles the evaluation logic and generates reports. |
| **`id_extractor`** | FastAPI + Qwen2.5-VL | Extracts OCR data from identity documents with high precision. |
| **`signature_service`** | FastAPI + YOLO + CNN | Detects and verifies signatures in documents. |
| **`vector_service`** | FastAPI + Gemini | Handles PDF processing and vector search for banking policies. |
| **`pg_vector`** | PostgreSQL + pgvector | Stores document embeddings for RAG. |

---

## 🛠️ Tech Stack

-   **Frontend**: [Streamlit](https://streamlit.io/)
-   **API Framework**: [FastAPI](https://fastapi.tiangolo.com/)
-   **AI Engines**: 
    -   **Ollama**: Powering `gemma4:e4b` for reasoning and logic.
    -   **Qwen2.5-VL**: Vision-Language model for document extraction.
    -   **YOLOv11**: Real-time signature detection.
    -   **Siamese CNN**: Deep learning for signature similarity scoring.
-   **Database**: PostgreSQL with `pgvector`.
-   **Vectorization**: Google Gemini API via `langchain`.
-   **Infrastructure**: Docker & Docker Compose.

---

## 🚦 Getting Started

### Prerequisites
-   Docker and Docker Compose installed.
-   NVIDIA GPU (Recommended for `id_extractor`).
-   Google API Key (for `vector_service`).
-   Ollama instance running locally or accessible via network.

### Setup
1.  **Create an Environment File**: Create a `.env` file in the `Workflow/` directory:
    ```env
    GOOGLE_API_KEY=your_google_api_key_here
    OLLAMA_URL=http://your-ollama-ip:11434
    OLLAMA_MODEL=gemma4:e4b
    ```

2.  **Start the Services**:
    ```bash
    docker compose up --build
    ```

3.  **Access the Dashboard**:
    Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📂 Project Structure

```text
Workflow/
├── backend/
│   ├── ai_service/         # Evaluation logic & Report generation
│   ├── id_extractor/       # Qwen2.5-VL based ID processor
│   ├── signature/          # YOLO signature detector & Siamese verifier
│   └── vector_service/     # PDF vectorization & RAG support
├── frontend/
│   └── app.py              # Streamlit Web Interface
├── policy/                 # Banking policy PDFs for RAG
└── docker-compose.yml      # Service orchestration
```

## 🔐 Security & Validation
-   **Document Validation**: Automatic expiry date check for IDs.
-   **Signature Trust**: Multi-stage verification using a 0.85 similarity threshold.
-   **Audit Ready**: Generates detailed reports for manual review.

---
*Built for modern banking automation.*
