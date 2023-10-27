FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Setup basic system
RUN apt-get update -yq \
  && apt-get upgrade -yq \
  && apt-get install --no-install-recommends -qy \
    build-essential \
    gcc \
    g++ \
    git \
    inetutils-ping \
    wget \
    nano \
    vim \
    htop \
    procps \
    libjpeg-dev \
    gdb \
    libdw-dev \
    libelf-dev \
    cmake \
  && rm -rf /var/lib/apt/lists/*

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser
ENV PATH="${PATH}:/home/appuser/.local/bin"

# Install mamba
ENV CONDA_DIR /opt/mamba
ENV MAMBA_DIR /opt/mamba
RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh" -O ~/mamba.sh && \
    /bin/bash ~/mamba.sh -b -p /opt/mamba
ENV PATH=$CONDA_DIR/bin:$PATH
RUN mamba update -n base -c defaults mamba && mamba update --all && mamba init bash

# Install dependencies
COPY ./environment.yml /tmp/environment.yml
RUN mamba env create -f /tmp/environment.yml