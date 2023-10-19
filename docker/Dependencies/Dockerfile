FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

ARG DEBIAN_FRONTEND=noninteractive

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
    libpq-dev \
    gdb \
    libdw-dev \
    libelf-dev \
    cmake \
    ca-certificates \
    libpq-dev \
    postgresql-server-dev-all \
    libsqlite3-dev \
    software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/ppa -yq \
  && apt-get update -yq \
  && apt-get install -qy g++-13 gcc-13 \
  && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 --slave /usr/bin/g++ g++ /usr/bin/g++-13 --slave /usr/bin/gcov gcov /usr/bin/gcov-13 \
  && update-alternatives --set cc /usr/bin/gcc \
  && update-alternatives --set c++ /usr/bin/g++
RUN gcc --version && g++ --version && cmake --version
RUN rm -rf /var/lib/apt/lists/*

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