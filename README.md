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
mypy modyn
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
Use `docker compose up --build` to start all containers and `docker compose up --build --abort-on-container-exit --exit-code-from tests` to run the integration tests once and exit.
The `tests` service runs integration tests, if started (e.g., in the Github Workflow).
Currently, we are required to set the `DOCKER_BUILDKIT` environment variable to 0 by running `export DOCKER_BUILDKIT=0` before starting docker-compose.
This is because docker compose tries to fetch `modynbase` from Docker Hub in parallel while building it locally, which fails as this image only exists locally as an intermediate step.
We need to investigate why this is the case and how to fix this.
In case you encounter issues when running integration tests, you can try deleting the local postgres data folders.

### tmuxp Setup
For local deployment, you can use tmuxp, which enables to load a tmux session from a file.
After running `docker-compose up`, run `tmuxp load tmuxp.yaml` to start a tmux session that is attached to all containers.
You will have access to a supervisor container in which you can submit pipelines, to panes for administrating the databases, and to all gRPC components.
To end the session, run CTRL+B (or your tmux modifier), and enter `:kill-session`.

## Example: Running a Pipeline

In this section, we give an example on how to run an experiment on a Google Cloud VM with an NVIDIA GPU.
We assume you have NVIDIA drivers/CUDA, `docker`, and `nvidia-docker` installed.
Furthermore, we assume that a disk is mounted at `/mnt/datasets` where we can store datasets.

### [Optional] Creating an example dataset
If you do not have a dataset yet, you can create an example MNIST dataset.
For this, follow the instructions in `benchmarks/mnist/README.md` to install the necessary dependencies.
Then, run `python data_generation.py --timestamps INCREASING --dir /mnt/datasets/mnist` to download the dataset to `/mnst/datasets/mnist`.

### Updating the dependencies to use CUDA
Next, in the `environment.yml` file, you want to uncomment the two lines that install `pytorch-cuda` and `cudatoolkit`.
If necessary, you can adjust the CUDA version.
Furthermore, you need to comment out the line that installes `cpuonly` from the pytorch channel.
Until #104 is solved, all dependencies are managed there.

### Adjusting the docker-compose configurations
Next, we need to update the `docker-compose.yml` file to reflect the local setup.
First, for the `trainer_server` service, you should enable the `runtime` and `deploy` options such that we have access to the GPU in the trainer server container.
Next, for the `storage` service, you should uncomment the `volumes` option to mount `/mnt/datasets` to `/datasets` in the container.
Optionally, you can uncomment the `.:/modyn_host` mount for all services to enable faster development cycles.
This is not required if you do not iterate.

### Starting the containers and the pipeline
Next, run `export DOCKER_BUILDKIT=0 && docker compose up --build` to build the containers and start them. 
This may take several minutes.
After building the containers, run `tmuxp load tmuxp.yaml` to have access to all container shells and logs.
Switch to the supervisor pane (using regular tmux bindings).
There, you can now submit a pipeline using the `modyn-supervisor` command.
For example, you can run `modyn-supervisor --start-replay-at 0 benchmark/mnist/mnist.yaml modyn/config/examples/modyn_config.yaml`.

### Iterating
Since we copy the Modyn sources into the containers, if we change something locally outside of the containers, this does not get reflected in the containers.
To avoid rebuilding the containers every time you want to test a change, if you mounted `/modyn_host` into the containers, you can switch into the pane of the service you want to update, and run `cd /modyn_host && pip install -e .`.
Afterwards, run `docker compose restart <SERVICE>` on the host machine to reflect the changes on the running system.