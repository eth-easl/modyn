<div align="center">
<img src="docs/assets/logo.png" height=100 alt="Modyn logo"/>

---

[![GitHub Workflow Status](https://github.com/eth-easl/modyn/actions/workflows/workflow.yaml/badge.svg)](https://github.com/eth-easl/modyn/actions/workflows/workflow.yaml)
[![License](https://img.shields.io/github/license/eth-easl/modyn)](https://img.shields.io/github/license/eth-easl/modyn)

Modyn is an open-source platform for model training on dynamic datasets, i.e., datasets where points get added or removed over time.

</div>


## ⚡️ Quickstart

For deploying and running integration tests, you will need [Docker](https://docs.docker.com/get-docker/).
Furthermore, we use conda for local environments and tmuxp for easily managing components panes (optional).
For local development, run
```bash
conda env create -f ./environment.yml
pip install -e .
pip install -r dev-requirements.txt
```
and then `./compliance_check.sh` to check that your local installation of Modyn is functioning.

If you want to run all Modyn components, run
```bash
./run_modyn.sh
tmuxp load tmuxp.yaml
```

For running all integration tests, run
```bash
./run_integrationtests.sh
```

> **_macOS Installation:_**: If you develop/run on macOS, you need to modify the `environment.yml` file until we have conditional dependencies (#104). You need to remove the pytorch channel and all occurences of `pytorch::` from the file.

> **_GPU Installation:_**: If you want to use a GPU, you need to install `nvidia-docker` and adjust the `docker-compose.yml` file as explained in the file. Furthermore, you need to modify the `environment.yml` to use the CUDA version of Pytorch.

**Next Steps**.
Checkout our [Example Pipeline](docs/EXAMPLE.md) guide for an example on how to run a Modyn pipeline.
Checkout our [Technical Guidelines](docs/TECHNICAL.md) for some hints on developing Modyn.
Last, checkout our [vision paper on Modyn](https://anakli.inf.ethz.ch/papers/MLonDynamicData_EuroMLSys23.pdf) for an introduction to model training on dynamic datasets.

We are actively developing and designing Modyn, including more thorough documentation.
Please reach out via Github, Twitter, E-Mail, or any other channel of communication if you are interested in collaborating, have any questions, or have any problems running Modyn.

How to [contribute](docs/CONTRIBUTING.md).

## What are dynamic datasets and what is Modyn used for?
ML is is often applied in use cases where training data evolves and/or grows over time, i.e., datasets are _dynamic_ instead
Training must incorporate data changes for high model quality, however this is often challenging and expensive due to large datasets and models.
With Modyn, we are actively developing an an open-source platform that manages dynamic datasets at scale and supports pluggable policies for when and what data to train on.
Furthermore, we ship a representative open-source benchmarking suite for ML training on dynamic datasets.

Modyn allows researchers to explore training and data selection policies, while alleviating the burdens of managing large dynamic datasets and orchestrating recurring training jobs.
However, we strive towards usage of Modyn in practical environments as well.
We welcome input from both research and practice.

## ✉️ About
Modyn is being developed at the [Efficient Architectures and Systems Lab (EASL)](https://anakli.inf.ethz.ch/#Group) at the [ETH Zurich Systems Group](https://systems.ethz.ch/).
Please reach out to `mboether [at] inf [­dot] ethz [dot] ch` if you have any questions or inquiry related to Modyn and its usage.


## Development 

### Requirements:
- [Docker](https://docs.docker.com/get-docker/) for deployment and running integration tests
- For development, you might want to install the local dev requirements using `pip install -r dev-requirements.txt` when in the project root
- Furthermore, to install modyn as a development module, run `pip install -e .` in the project root

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
We manage dependency required to run Modyn using conda.
All dependencies are listed in the `environment.yml` file in the project root.
Development dependencies are managed in the `dev-requirements.txt` file in the project root.
There are two ways to develop modyn locally.
First, if not using Docker, you can install all dependencies and the Modyn module itself on your local machine via `conda env create -f ./environment.yml`, `pip install -e .`, and `pip install -r dev-requirements.txt`.

Second, you can use a Docker container.
We provide a Modyn base container where the conda setup is already done. 
You can find the Dockerfile in `docker/Base/Dockerfile` and build the image using `docker build -t modyn -f docker/Base/Dockerfile .`.
Then, you can run a container for example using `docker run modyn /bin/bash`.

### Docker-Compose Setup
We use docker-compose to manage the system setup.
The `docker-compose.yml` file describes our setup. 
The setup expects the base image to be built already; if you use the scripts, these take care of that for you.
The `tests` service runs integration tests, if started (e.g., in the Github Workflow).
You can run `run_integrationtests.sh` to run the integration tests, and `run_modyn.sh` to run all containers required for end-to-end workflows.
In case you encounter issues when running integration tests, you can try deleting the local postgres data folders.
Note that you might want to update the `metadata_postgres.conf` and `storage_postgresql.conf` according to your machine.

### tmuxp Setup
For local deployment, you can use tmuxp, which enables to load a tmux session from a file.
After running `./run_modyn.sh`, run `tmuxp load tmuxp.yaml` to start a tmux session that is attached to all containers.
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
Next, run `./run_modyn.sh` to build the containers and start them. 
This may take several minutes for the first time.
After building the containers, run `tmuxp load tmuxp.yaml` to have access to all container shells and logs.
Switch to the supervisor pane (using regular tmux bindings).
There, you can now submit a pipeline using the `modyn-supervisor` command.
For example, you can run `modyn-supervisor --start-replay-at 0 benchmark/mnist/mnist.yaml modyn/config/examples/modyn_config.yaml`.

### Iterating
Since we copy the Modyn sources into the containers, if we change something locally outside of the containers, this does not get reflected in the containers.
To avoid rebuilding the containers every time you want to test a change, if you mounted `/modyn_host` into the containers, you can switch into the pane of the service you want to update, and run `cd /modyn_host && pip install -e .`.
Afterwards, run `docker compose restart <SERVICE>` on the host machine to reflect the changes on the running system.