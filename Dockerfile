# Use PyTorch CUDA base image (includes python, pip, torch, cuda)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
# git: for python dependencies pointing to git repos
# libgl1-mesa-glx, libglib2.0-0: for opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage caching
COPY requirements.txt .

# Install Python dependencies
# --no-deps for torch/torchvision to ensure we use the pre-installed CUDA versions
# OR simply remove them from requirements.txt dynamically if needed.
# Here we assume requirements.txt might list them but we prefer the image's version.
# A safe way is to install the rest.
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create outputs/checkpoints dirs
RUN mkdir -p checkpoints outputs

# Set environment variables for non-interactive python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["python", "train.py", "--config", "config/config.yaml"]
