FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11 and set aliases
RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python3.11 get-pip.py \
    && rm get-pip.py \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Set up the workspace directory
WORKDIR /app

# Set application environment variables
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV HF_HOME="/app/.cache/huggingface"
ENV TRANSFORMERS_CACHE="/app/.cache/huggingface"

# Install PyTorch with CUDA support first (before other requirements)
RUN python3.11 -m pip install --no-cache-dir \
    torch==2.8.0+cu128 \
    torchvision==0.23.0+cu128 \
    torchaudio==2.8.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Copy requirements first for better caching
COPY requirements.txt /app/

# Install main requirements using Python 3.11
RUN python3.11 -m pip install --no-cache-dir -r /app/requirements.txt

# Setup cache and temp directories
RUN mkdir -p /app/.cache/huggingface && \
    mkdir -p /app/temp && \
    chmod 777 /app/temp && \
    chmod 777 /app/.cache/huggingface

# Copy application files
COPY handler.py /app/
COPY download_models.py /app/

# Set environment variables for model download
ENV BASE_MODEL="Qwen/Qwen-Image-Edit-2511"
ENV LORA_MODEL="fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA"
ENV USE_LORA="true"
ENV AUTO_DOWNLOAD_MODELS="true"

# Download models during build (base model and LoRA)
RUN python3.11 download_models.py --base-model "${BASE_MODEL}" --lora "${LORA_MODEL}"

# Set the entry point for RunPod serverless
ENTRYPOINT ["python", "-u", "handler.py"]

