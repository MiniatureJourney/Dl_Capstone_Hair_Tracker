# =====================================================================
# Production Dockerfile for Google Cloud Run
# Hair Progress Tracker & Deep Learning Pipeline Visualizer
# =====================================================================

FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing bytecode and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .

# Install optimized CPU PyTorch and Python dependencies
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files, model weights, and sample data
COPY . .

# Cloud Run listens on port 8080 by default
EXPOSE 8080

# Launch Gradio web server
CMD ["python", "app.py"]
