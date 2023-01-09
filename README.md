# Modyn: A Research Platform For ML Model Training On Dynamic Datasets

TODO(MaxiBoether): Improve the README.md

## Contributing

How to [contribute](CONTRIBUTING.md).

## Development 

### Requirements:
- [Docker](https://docs.docker.com/get-docker/) for the infrastructure, such as databases and storage
- For development, you might want to install the local dev requirements using `pip install -r dev-requirements.txt` when in the project root
- Furthermore, to install modyn as a development module, i.e., a module that can be imported but is synced to the current source code in the folder, run `pip install -e .` in the project root

### Running pytest and flake8 locally

Make sure that you installed modyn as a development module as stated above. Then, in the project root you can run `flake8 --statistics` and `pytest` to run unit tests and the flake8 linter.

### Running autopep8

autopep8 is an automatic linter for pep8 compliance that can fix many issues automatically. Make sure to commit/backup before running autopep8 inplace. Then, run
```
autopep8 modyn --recursive --in-place --pep8-passes 2000 --verbose
```
in the project root.

### Configuration:
- In `src/config/config.yaml` you will find a test configuration for the system, adapt as required

### Current architecture:

![Current architecture diagram](docs/images/Architecture.png)

### Conda and Docker Setup
We manage dependency required to run Modyn using conda. All dependencies are listed in the `environment.yml` file in the project root.
Development dependencies are managed in the `dev-requirements.txt` file in the project root.
There are two ways to develop modyn.
First, you can install all dependencies on your local machine via `conda env create -f ./environment.yml` and `pip install -r dev-dependencies.txt`. Modyn itself should be installed via conda, but if you run into problems try `pip install -e .` in the project root.

Second, you can use a Docker container. We provide a Modyn base container where the conda setup is already done. You can find the Dockerfile in `docker/Base/Dockerfile` and build the image using `docker build -t modyn -f docker/Base/Dockerfile .`. Then, you can run a container for example using `docker run modyn /bin/bash`.

### Docker-Compose Setup
We use docker-compose to manage the system setup.
The `docker-compose.yml` file describes our setup. 
Use `docker compose up --build` to start all containers and `docker compose up --build --abort-on-container-exit --exit-code-from tests` to run the integration tests.
The `tests` service runs integration tests, if started (e.g., in the Github Workflow).
Last, on macOS, you might be required to set the `DOCKER_BUILDKIT` environment variable to 0, if you run into problems during the build process.