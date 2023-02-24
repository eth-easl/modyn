FROM python:3.10-slim

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

# Mount modyn directory to container, make it working directory
ADD . /src
WORKDIR /src
RUN chown -R appuser /src

# Create modyn environment (delete old if exists)
RUN conda env remove -n modyn && conda env create -f ./environment.yml
ENV CONDA_DEFAULT_ENV modyn

# Initialize again for user shell
USER appuser
RUN /bin/bash -c "conda init"
RUN echo "conda activate modyn" >> /home/appuser/.bashrc

# The base container is just used as a base image for docker-compose and doesn't require anything else.
# We cannot quit here, as our test setup would break if the base container exits.
CMD tail -f /dev/null
