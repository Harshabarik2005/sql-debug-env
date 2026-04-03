# SQL Debug Environment — Docker image for Hugging Face Spaces
# Build:  docker build -t sql-debug-env .
# Run:    docker run -p 7860:7860 sql-debug-env

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        sqlite3 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Ensure the project root is always on PYTHONPATH
ENV PYTHONPATH=/app

# Hugging Face Spaces listens on port 7860
EXPOSE 7860

# Health check — used by the OpenEnv validator
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=5 \
    CMD curl -sf http://localhost:7860/health || exit 1

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
