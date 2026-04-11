FROM python:3.12-slim
WORKDIR /app
COPY Workflow/backend/id_extractor/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy the entire project context to allow access to src
COPY . /app
CMD ["uvicorn", "Workflow.backend.id_extractor.main:app", "--host", "0.0.0.0", "--port", "8002", "--log-level", "info"]
