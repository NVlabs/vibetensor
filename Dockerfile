# VibeTensor CI Docker Image
# Base: nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04
#
# Build:   docker build -t vibetensor-ci .
# Usage:   docker run --gpus all vibetensor-ci <command>

FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04

LABEL maintainer="VibeTensor Team"
LABEL description="CI runtime image for VibeTensor with CUDA 13.0.2 and cuDNN"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# ============================================================================
# System Dependencies
# ============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    ninja-build \
    # For adding PPAs and CMake repo and Node via nvm
    software-properties-common \
    wget \
    curl \
    ca-certificates \
    gnupg \
    # Git for checkout and submodules
    git \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Install Python 3.12 from deadsnakes PPA
# ============================================================================
RUN add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install pip for Python 3.12
RUN wget -qO- https://bootstrap.pypa.io/get-pip.py | python3.12

# ============================================================================
# Install CMake 3.26+ from Kitware APT repository
# ============================================================================
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
    | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
    && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' \
    | tee /etc/apt/sources.list.d/kitware.list >/dev/null \
    && apt-get update \
    && apt-get install -y --no-install-recommends cmake \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Install Node.js via nvm (for Node addon and JS tests)
# ============================================================================
ENV NVM_DIR=/usr/local/nvm

RUN mkdir -p "$NVM_DIR" \
    && curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh \
      | bash

RUN bash -lc 'set -euo pipefail; \
    . "$NVM_DIR/nvm.sh"; \
    nvm install 22; \
    nvm alias default 22; \
    NODE_PREFIX="$NVM_DIR/versions/node/$(nvm version 22)"; \
    ln -sf "$NODE_PREFIX/bin/node" /usr/local/bin/node; \
    ln -sf "$NODE_PREFIX/bin/npm" /usr/local/bin/npm; \
    ln -sf "$NODE_PREFIX/bin/npx" /usr/local/bin/npx; \
    echo "export NODE_INCLUDE_DIR=$NODE_PREFIX/include/node" > /etc/profile.d/vibetensor-node.sh; \
    echo "export PATH=$NODE_PREFIX/bin:\$PATH" >> /etc/profile.d/vibetensor-node.sh'


# ============================================================================
# Python Build Dependencies
# ============================================================================
RUN python -m pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel

RUN python -m pip install --no-cache-dir \
    build \
    "scikit-build-core>=0.9" \
    ninja

# Test dependencies
RUN python -m pip install --no-cache-dir \
    "pytest>=7" \
    numpy

# PyTorch and Triton (for full test coverage)
RUN python -m pip install --no-cache-dir \
    torch \
    triton

# ============================================================================
# Environment Configuration
# ============================================================================
# CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV CMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc"

# Enable torch parity check (torch is installed)
ENV VBT_SKIP_PARITY=0

# ============================================================================
# Working Directory
# ============================================================================
WORKDIR /workspace

# Default shell
CMD ["/bin/bash"]
