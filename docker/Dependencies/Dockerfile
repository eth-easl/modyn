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
    libpq-dev \
  && rm -rf /var/lib/apt/lists/*

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser
ENV PATH="${PATH}:/home/appuser/.local/bin"

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN if [ "$(dpkg --print-architecture)" = "arm64" ]; then ARCHITECTURE=aarch64; else ARCHITECTURE=x86_64; fi \
    && wget --quiet "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${ARCHITECTURE}.sh" -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda update -n base -c defaults conda && conda update --all && conda init bash

# Install CMake
ARG CMAKE_VERSION=3.26.4

RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh \
      -q -O /tmp/cmake-install.sh \
      && chmod u+x /tmp/cmake-install.sh \
      && mkdir /usr/bin/cmake \
      && /tmp/cmake-install.sh --skip-license --prefix=/usr/bin/cmake \
      && rm /tmp/cmake-install.sh \
      && chmod u+x /usr/bin/cmake

ENV PATH="/usr/local/cmake/bin:${PATH}"

# Install dependencies
COPY ./environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml