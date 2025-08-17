# Multi-stage Docker build for trading system
FROM python:3.9-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install DVC
RUN pip install dvc[s3]

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY dvc.yaml .
COPY params.yaml .

# Create directories
RUN mkdir -p data/processed models metrics plots

# Set environment variables
ENV PYTHONPATH=/app
ENV DVC_CACHE_DIR=/tmp/dvc-cache

# Training stage
FROM base as training
COPY train_*.py ./
COPY run_dvc_experiments.py .
COPY experiments/ ./experiments/

# Default command
CMD ["python", "train_multi_agent.py"]

# Airflow worker stage  
FROM base as airflow-worker
RUN pip install apache-airflow[celery,postgres,redis,s3]==2.5.0

# Copy Airflow DAGs
COPY airflow/dags/ /opt/airflow/dags/

# Set Airflow home
ENV AIRFLOW_HOME=/opt/airflow

# Airflow entrypoint
CMD ["airflow", "worker"]
