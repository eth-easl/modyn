# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN apt-get update -q \
  && apt-get install --no-install-recommends -qy \
    gcc \
    g++ \
    inetutils-ping \
  && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install build-essential -y
RUN apt-get install -y postgresql
RUN apt-get install -y python3-dev
RUN apt-get install -y libpq-dev

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /src
COPY . /src

RUN pip install -e .
RUN pip install --upgrade protobuf==3.20.0
RUN pip install torchvision

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /src
USER appuser

RUN chmod a+x run.sh

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["./run.sh"]
