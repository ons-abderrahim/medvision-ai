# ──────────────────────────────────────────────────────────────
# MedVision AI — Docker Image
# ──────────────────────────────────────────────────────────────
# Multi-stage build: keeps the final image lean (~3 GB with CUDA)
#
# Build:
#   docker build -t medvision-ai .
#
# Run API:
#   docker run -p 8000:8000 -v $(pwd)/checkpoints:/app/checkpoints medvision-ai api
#
# Run Gradio UI:
#   docker run -p 7860:7860 -v $(pwd)/checkpoints:/app/checkpoints medvision-ai ui

FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source
COPY src/ src/
COPY app.py .
COPY configs/ configs/
COPY assets/ assets/

# Expose ports for API and Gradio
EXPOSE 8000 7860

# Default: launch Gradio UI
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["python app.py"]
