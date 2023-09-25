# Technical Nuts and Bolts

Please follow the Quickstart Section in the [README](../README.md) for a list of required software and how to install Modyn.
Please refer to the ­[Architecture Documentation](ARCHITECTURE.md) for an overview of Modyn's components.
This document will deal with some more details of development and how to extend Modyn.

### Linting and Testing
For an automatic execution of automatic formatting, linting, and testing, you can use the `compliance_check.sh` script in the project root. 
This script runs isort, autopep8, black, mypy, pylint, and pytest.
The script assumes that `mamba` is available and if not, tries to activate it on `zsh` and `bash`.
Furthermore, you must have created a mamba environment called `modyn` with the dependencies listed in `environment.yml` and `dev-requirements.txt`.

To run linters/formatters/pytest manually, make sure to enable the mamba environment and then run the tools in your command line.
To run integration tests, run the `./run_integrationtests.sh` script.
This will take care of setting up all containers and running the tests.

### Mamba and Docker Setup
We manage dependency required to run Modyn using mamba.
Mamba is a fast implementation of conda.
All dependencies are listed in the `environment.yml` file in the project root.
Development dependencies are managed in the `dev-requirements.txt` file in the project root.
There are two ways to develop modyn locally.
First, if not using Docker, you can install all dependencies and the Modyn module itself on your local machine via `mamba env create -f ./environment.yml`, `pip install -e .`, and `pip install -r dev-requirements.txt`, as outlined in the README.
Note that the `initial_setup.sh` scripts performs some adjustments to your mamba and docker settings, depending on your local system (e.g., choose correct Pytorch channel when on macOS, or enable CUDA).

Second, you can use a Docker container.
We provide a Modyn base container where the mamba setup is already done. 
You can find the Dockerfile in `docker/Base/Dockerfile` and build the image using `docker build -t modyn -f docker/Base/Dockerfile .`.
Then, you can run a container for example using `docker run modyn /bin/bash`.

### Docker-Compose Setup
We use docker-compose to manage the system setup.
The `docker-compose.yml` file describes our setup and includes comments explaining it.
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

## Adding new data selection and triggering policies

Data selection policies are managed by the Selector.
All policies are implemented in the `modyn/selector/internal/selector_strategies` directory.
The `AbstractSelectionStrategy` class is the parent class all selection strategies inherit from.
A new strategy needs to overwrite the `_on_trigger`, `_reset_state`, and `inform_data` functions according to the specification outlined in the `AbstractSelectionStrategy` class.
When implementing a new strategy, make sure to respect the `limit`, `reset_after_trigger`, and `_maximum_keys_in_memory` parameters.
Check out the `NewDataStrategy` for an example implementation.

Triggering policies are managed by the Supervisor.
All policies are implemented in the `modyn/supervisor/internal/triggers` directory.
The `Trigger` class is the parent class all triggers inherit from.
A new trigger needs to overwrite the `inform` function according to the specification outlined in the `Trigger` class.
Checkout the `DataAmountTrigger` for an example implementation.
