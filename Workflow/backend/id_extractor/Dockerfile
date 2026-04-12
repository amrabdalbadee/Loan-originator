# Use a multi-arch NVIDIA CUDA image (supports aarch64)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Link python3 to python if needed
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY Workflow/backend/id_extractor/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the entire project context
COPY . /app

CMD ["uvicorn", "Workflow.backend.id_extractor.main:app", "--host", "0.0.0.0", "--port", "8002", "--log-level", "info"]


