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

### Automatic linting and import sorting

autopep8 and black are automatic linters for pep8 compliance that can fix many issues automatically.
isort is an automatic import sorting tool.
Make sure to commit/backup before running them inplace. Then, in order to automatically fix issues, run
```
isort .
autopep8 modyn --recursive --in-place --pep8-passes 2000 --verbose
black modyn --verbose --config black.toml
```
in the project root.

You can make sure that the code you have is compliant by running
```
flake8
isort . --check --diff
black --check modyn --verbose --config black.toml
pylint modyn
```

Remember to make sure that the tests still work after running linting.

For an automatic execution of automatic formatting and the linters, you can use the `compilance_check.sh` script in the project root. 
The script assumes that `conda` is available and if not, tries to activate it on `zsh` and `bash`.
Furthermore, you must have created a conda environment called `modyn` with the dependencies listed in `environment.yml` and `dev-requirements.txt`.

### Configuration:
- In `modyn/config/examples.yaml` you will find both an example pipeline configuration and system configuration

### Current architecture:

![Current architecture diagram](docs/images/Architecture.png)

### Conda and Docker Setup
We manage dependency required to run Modyn using conda. All dependencies are listed in the `environment.yml` file in the project root.
Development dependencies are managed in the `dev-requirements.txt` file in the project root.
There are two ways to develop modyn.
First, you can install all dependencies on your local machine via `conda env create -f ./environment.yml` and `pip install -r dev-requirements.txt`.
Modyn itself should be installed via conda, but if you run into problems try `pip install -e .` in the project root.

Second, you can use a Docker container. We provide a Modyn base container where the conda setup is already done. You can find the Dockerfile in `docker/Base/Dockerfile` and build the image using `docker build -t modyn -f docker/Base/Dockerfile .`. Then, you can run a container for example using `docker run modyn /bin/bash`.

### Docker-Compose Setup
We use docker-compose to manage the system setup.
The `docker-compose.yml` file describes our setup. 
Use `docker compose up --build` to start all containers and `docker compose up --build --abort-on-container-exit --exit-code-from tests` to run the integration tests.
The `tests` service runs integration tests, if started (e.g., in the Github Workflow).
Last, on macOS, you might be required to set the `DOCKER_BUILDKIT` environment variable to 0, if you run into problems during the build process.
