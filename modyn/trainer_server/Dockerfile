# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10-slim

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN apt-get update -q \
  && apt-get install --no-install-recommends -qy \
    build-essential \
    postgresql \
    python3-dev \
    gcc \
    g++ \
    inetutils-ping \
    libpq-dev \
  && rm -rf /var/lib/apt/lists/*

# Install pip requirements
COPY ./modyn/trainer_server/requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /src
COPY . /src

RUN pip install -e .

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /src
USER appuser

RUN chmod a+x ./modyn/trainer_server/run.sh

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["./modyn/trainer_server/run.sh"]
