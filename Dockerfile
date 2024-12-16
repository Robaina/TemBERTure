# Use CUDA 11.3 base image which is compatible with PyTorch 1.10.2
FROM nvidia/cuda:11.3.1-base-ubuntu20.04

# Avoid timezone interactive prompt during installation
ENV DEBIAN_FRONTEND=noninteractive 

# Set working directory
WORKDIR /app

# Install system dependencies and Python 3.9
RUN apt update && apt install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY temBERTure/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the TemBERTure code and models
COPY temBERTure/temBERTure.py .
COPY temBERTure/temBERTure_CLS ./temBERTure_CLS
COPY temBERTure/temBERTure_TM ./temBERTure_TM

# Copy entrypoint script
COPY entrypoint.py .

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Set the entrypoint
ENTRYPOINT ["python", "/app/entrypoint.py"]