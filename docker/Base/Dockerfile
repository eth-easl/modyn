FROM python:3.10-slim

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Setup basic system
RUN apt-get update -q \
  && apt-get upgrade -q \
  && apt-get install --no-install-recommends -qy \
    build-essential \
    gcc \
    g++ \
    inetutils-ping \
    wget \
    nano \
    vim \
  && rm -rf /var/lib/apt/lists/*


# Install miniconda
ENV CONDA_DIR /opt/conda
RUN if [ "$(dpkg --print-architecture)" = "arm64" ]; then ARCHITECTURE=aarch64; else ARCHITECTURE=x86_64; fi \
    && wget --quiet "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${ARCHITECTURE}.sh" -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Mount modyn directory to container, make it working directory
ADD . /src
WORKDIR /src

# Create modyn environment
RUN conda update -n base -c defaults conda && conda update --all && conda env create -f ./environment.yml && conda init bash
ENV CONDA_DEFAULT_ENV modyn

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /src
USER appuser
ENV PATH="${PATH}:/home/appuser/.local/bin"

# Initialize again for user shell
RUN /bin/bash -c "conda init"
RUN echo "conda activate modyn" >> /home/appuser/.bashrc

# The base container is just used as a base image for docker-compose and doesn't require anything else.
RUN echo "This is the base container."