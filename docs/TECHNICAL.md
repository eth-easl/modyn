# Technical Nuts and Bolts

Please follow the Quickstart Section in the [README](../README.md) for a list of required software and how to install Modyn.
Please refer to the ­[Architecture Documentation](ARCHITECTURE.md) for an overview of Modyn's components.
This document will deal with some more details of development and how to extend Modyn.

## Coding

### Pipeline Orchestration

The pipeline is orchestrated by the `Supervisor` component. We use the `PipelineExecutor` class in the `modyn/supervisor/internal/pipeline_executor/pipeline_executor.py` file to coordinate the different stages of the pipeline. The pipeline stages correspond to the `PipelineStage` enum values from `modyn/supervisor/internal/grpc/enums.py` and are reflected in dedicated member functions of the `PipelineExecutor` class. State transitions are managed by the `PipelineExecutor's` `run` method together with the `@pipeline_stage` decorator which adds logging to every pipeline stage function. The flow of stages can be found in [PIPELINE.md](pipeline/PIPELINE.md).

### Data selection policies

Data selection policies are managed by the Selector.
All policies are implemented in the `modyn/selector/internal/selector_strategies` directory.
The `AbstractSelectionStrategy` class is the parent class all selection strategies inherit from.
A new strategy needs to overwrite the `_on_trigger`, `_reset_state`, and `inform_data` functions according to the specification outlined in the `AbstractSelectionStrategy` class.

#### Adding new data selection policies

When implementing a new strategy, make sure to respect the `limit`, `reset_after_trigger`, and `_maximum_keys_in_memory` parameters.
Check out the `NewDataStrategy` for an example implementation.

### Triggering policies

Triggering policies are managed by the Supervisor.
All policies are implemented in the `modyn/supervisor/internal/triggers` directory.
The `Trigger` class is the parent class all triggers inherit from.

#### Adding new triggering policies

A new trigger needs to overwrite the `inform` function according to the specification outlined in the `Trigger` class.
Checkout the `DataAmountTrigger` for an example implementation.

### Evaluation logic

As depicted in [PIPELINE.md](pipeline/PIPELINE.md) we support running evaluations on model right after their training (during their pipeline) as well as after the full pipeline has been executed.
Evaluations can be conducted with different strategies (e.g. Slicing, Periodic Windows, etc.). To measure the performance of a full pipeline run we use `composite models`, a post factum evaluation concept condensing multiple models that were trained during a pipeline run.
More details can be found in the [EVALUATION.md](EVALUATION.md) document.

## Tooling

### Linting and Testing

For an automatic execution of automatic formatting, linting, and testing of the Python components, you can use the `scripts/python_compliance.sh` script in the project root.
This script runs isort, autopep8, black, mypy, pylint, and pytest.
The script assumes that `micromamba` is available and if not, tries to activate it on `zsh` and `bash`.
Furthermore, you must have created a conda environment called `modyn` with the dependencies listed in `environment.yml` and `dev-requirements.txt`.
To run linters/formatters/pytest manually, make sure to enable the conda environment and then run the tools in your command line.

To run clang-format and clang-tidy in order to lint the C++ part of the codebase, use the `scripts/clang-format.sh` and `scripts/clang-tidy.sh` scripts.
Clang-format automatically fixes everything, while clang-tidy gives hints on what needs to be fixed.
To run C++ unit tests, create a `build` directory in the project root, run `cmake ..`, and then `make -j8 modyn-test` to build the test application.
After building, run `modyn/tests/modyn-test` to execute the tests.
Note you might want to enable `-DCMAKE_BUILD_TYPE=Debug` (or `asan` or `tsan`) in the `cmake` command to switch the build mode to debug, address sanitization, or thread sanitization, accordingly.

To run integration tests, run the `./scripts/run_integrationtests.sh` script.
This will take care of setting up all containers and running the tests.

### Mamba and Docker Setup

We manage dependency required to run Modyn using micromamba.
Micromamba is a fast single-executable implementation of conda.
All dependencies are listed in the `environment.yml` file in the project root.
Development dependencies are managed in the `dev-requirements.txt` file in the project root.
There are two ways to develop modyn locally.
First, if not using Docker, you can install all dependencies and the Modyn module itself on your local machine via `micromamba env create -f ./environment.yml`, `pip install -e .`, and `pip install -r dev-requirements.txt`, as outlined in the README.
Note that the `scripts/initial_setup.sh` scripts performs some adjustments to your micromamba and docker settings, depending on your local system (e.g., choose correct Pytorch channel when on macOS, or enable CUDA).

Second, you can use a Docker container.
We provide a Modyn base container where the micromamba setup is already done.
You can find the Dockerfile in `docker/Base/Dockerfile` and build the image using `docker build -t modyn -f docker/Base/Dockerfile .`.
Then, you can run a container for example using `docker run modyn /bin/bash`.

### C++ Setup

Modyn uses pure Python components, C++ extensions for Python, and also pure C++ components.
We use CMake to manage the installation of extensions and components.
The `setup.py` file builds the C++ extensions when Modyn is installed via CMake.
When running Modyn using its Docker setup, the Dockerfile takes care of building the pure C++ components.
In case you want to build extensions or components on your own, you need to create a `build` directory and use CMake to create the Makefiles.
By default, we only build the extensions to avoid downloading the huge gRPC library.
In case you want to build the storage C++ component, enable `-DMODYN_BUILD_STORAGE=On` when running CMake.

Furthermore, by default, we enable the `-DMODYN_TRY_LOCAL_GRPC` flag.
This flag checks whether gRPC is available locally on your system and uses this installation for rapid development, instead of rebuilding gRPC from source everytime like in CI.
In order to install gRPC on your system, you can either use your system's package manager or run the following instructions:

```
git clone --recurse-submodules -b v1.59.2 --depth 1 --shallow-submodules https://github.com/grpc/grpc && \
    cd grpc && mkdir -p cmake/build && cd cmake/build && \
    cmake -DgRPC_PROTOBUF_PROVIDER=module -DABSL_ENABLE_INSTALL=On -DgRPC_BUILD_CSHARP_EXT=Off -DABSL_BUILD_TESTING=Off -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=${MODYN_DEP_BUILDTYPE} ../.. && \
    make -j8 && make install && cd ../../
```

Please adjust the version as required.
If you run into problems with the system gRPC installation, set `-DMODYN_TRY_LOCAL_GRPC=Off`.

### Docker-Compose Setup

We use docker-compose to manage the system setup.
The `docker-compose.yml` file describes our setup and includes comments explaining it.
The setup expects the base image to be built already; if you use the scripts, these take care of that for you.
The `tests` service runs integration tests, if started (e.g., in the Github Workflow).
You can run `scripts/run_integrationtests.sh` to run the integration tests, and `scripts/run_modyn.sh` to run all containers required for end-to-end workflows.
In case you encounter issues when running integration tests, you can try deleting the local postgres data folders.
Note that you might want to update the `conf/metadata_postgres.conf` and `conf/storage_postgresql.conf` according to your machine.

### tmuxp Setup

For local deployment, you can use tmuxp, which enables to load a tmux session from a file.
After running `./scripts/run_modyn.sh`, run `tmuxp load tmuxp.yaml` to start a tmux session that is attached to all containers.
You will have access to a supervisor container in which you can submit pipelines, to panes for administrating the databases, and to all gRPC components.
To end the session, run CTRL+B (or your tmux modifier), and enter `:kill-session`.
