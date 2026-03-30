# Loan Originator AI Platform

## Overview
Loan Originator is an AI-powered document verification and loan assessment platform. It processes applicant documents through five sequential pipeline stages and produces an auditable lending verdict.

## Pipeline Stages
- **Stage 0: Intake & Pre-processing (OCR)**: Extract text from uploads (Qwen2-VL / LayoutLM).
- **Stage 1: Identity & Compliance**: Validate National IDs, Passports, and residency documents.
- **Stage 2: Employment & Income**: Verify payslips, Commercial Registers, and bank statements.
- **Stage 3: Application & Signature**: Isolate and verify signatures (SigNet / Siamese CNN).
- **Stage 4: Financial Strength & Scoring**: Compute DBR/DTI metrics and provide the final verdict.

## Tech Stack
- **AI/ML**: Qwen2-VL, LayoutLM, SigNet, BERT, GPT-4o.
- **Backend**: Python, FastAPI, Pydantic v2.
- **Database**: PostgreSQL 16 + pgcrypto.
- **Orchestration**: Apache Airflow / Prefect + RabbitMQ / AWS SQS.
- **Infrastructure**: Docker + Kubernetes + AWS S3 / MinIO.

## Structure
- `src/api/`: FastAPI endpoints.
- `src/ai/`: Model integrations (OCR, Signature, NLP).
- `src/services/`: Logic for each pipeline stage.
- `src/models/`: Pydantic & SQLAlchemy data models.
- `src/database/`: Persistence layer.
- `src/utils/`: PDF generation and audit logging.
- `infra/`: Docker, K8s, and orchestration configs.
