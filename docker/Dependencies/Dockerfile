FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Setup basic system
RUN apt-get update -yq \
  && apt-get upgrade -yq \
  && apt-get install --no-install-recommends -qy \
    autoconf \
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
    libtool \
    pkg-config \
    cmake \
    ca-certificates \
    libpq-dev \
    libsqlite3-dev \
    software-properties-common \
    curl \
    unzip \
  && rm -rf /var/lib/apt/lists/* \
  && gcc --version && g++ --version && cmake --version

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser
ENV PATH="${PATH}:/home/appuser/.local/bin"

RUN mkdir /src
ARG MODYN_BUILDTYPE=Release
ENV MODYN_BUILDTYPE=$MODYN_BUILDTYPE
ARG MODYN_DEP_BUILDTYPE=Release
ENV MODYN_DEP_BUILDTYPE=$MODYN_DEP_BUILDTYPE
RUN echo "Used buildtype is ${MODYN_BUILDTYPE}" >> /src/.modyn_buildtype
RUN echo "Used dependency buildtype is ${MODYN_DEP_BUILDTYPE}" >> /src/.modyn_dep_buildtype

# Install gRPC systemwide. When updating the version, make sure to also update the storage_dependencies.cmake file
RUN git clone --recurse-submodules -b v1.59.2 --depth 1 --shallow-submodules https://github.com/grpc/grpc && \
    cd grpc && mkdir -p cmake/build && cd cmake/build && \
    cmake -DgRPC_PROTOBUF_PROVIDER=module -DABSL_ENABLE_INSTALL=On -DgRPC_BUILD_CSHARP_EXT=Off -DABSL_BUILD_TESTING=Off -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=${MODYN_DEP_BUILDTYPE} ../.. && \
    make -j8 && make install && cd ../../

# Install mamba
ENV CONDA_DIR=/opt/mamba
ENV MAMBA_DIR=/opt/mamba
RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -O ~/mamba.sh && \
    /bin/bash ~/mamba.sh -b -p /opt/mamba
ENV PATH=$CONDA_DIR/bin:$PATH
RUN mamba update -n base -c defaults mamba && mamba update --all && mamba shell init bash

# Install dependencies
COPY ./environment.yml /tmp/environment.yml
RUN mamba env create -f /tmp/environment.yml
