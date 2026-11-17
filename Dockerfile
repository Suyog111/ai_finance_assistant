# ==============================================================================
# Dockerfile — AI Finance Assistant
# Production-ready containerized deployment running Streamlit on port 8501
# ==============================================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, config, and data
COPY config.yaml .
COPY src/ ./src/
COPY data/ ./data/
COPY sample_data/ ./sample_data/

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck to monitor app status
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit app
ENTRYPOINT ["streamlit", "run", "src/web_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
