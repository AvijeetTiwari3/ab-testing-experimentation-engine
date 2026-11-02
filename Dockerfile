# Google-Standard Production Dockerfile for A/B Testing Engine API
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Expose API and Streamlit ports
EXPOSE 8000 8501

# Default environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Start FastAPI microservice by default
CMD uvicorn api.app:app --host 0.0.0.0 --port ${PORT}
