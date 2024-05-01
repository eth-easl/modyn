# Technical Nuts and Bolts

Please follow the Quickstart Section in the [README](../README.md) for a list of required software and how to install Modyn.
Please refer to the ­[Architecture Documentation](ARCHITECTURE.md) for an overview of Modyn's components.
This document will deal with some more details of development and how to extend Modyn.

### Linting and Testing
For an automatic execution of automatic formatting, linting, and testing of the Python components, you can use the `scripts/python_compliance.sh` script in the project root. 
This script runs isort, autopep8, black, mypy, pylint, and pytest.
The script assumes that `mamba` is available and if not, tries to activate it on `zsh` and `bash`.
Furthermore, you must have created a mamba environment called `modyn` with the dependencies listed in `environment.yml` and `dev-requirements.txt`.
To run linters/formatters/pytest manually, make sure to enable the mamba environment and then run the tools in your command line.

To run clang-format and clang-tidy in order to lint the C++ part of the codebase, use the `scripts/clang-format.sh` and `scripts/clang-tidy.sh` scripts. 
Clang-format automatically fixes everything, while clang-tidy gives hints on what needs to be fixed.
To run C++ unit tests, create a `build` directory in the project root, run `cmake ..`, and then `make -j8 modyn-test` to build the test application.
After building, run `modyn/tests/modyn-test` to execute the tests.
Note you might want to enable `-DCMAKE_BUILD_TYPE=Debug` (or `asan` or `tsan`) in the `cmake` command to switch the build mode to debug, address sanitization, or thread sanitization, accordingly.

To run integration tests, run the `./scripts/run_integrationtests.sh` script.
This will take care of setting up all containers and running the tests.

### Mamba and Docker Setup
We manage dependency required to run Modyn using mamba.
Mamba is a fast implementation of conda.
All dependencies are listed in the `environment.yml` file in the project root.
Development dependencies are managed in the `dev-requirements.txt` file in the project root.
There are two ways to develop modyn locally.
First, if not using Docker, you can install all dependencies and the Modyn module itself on your local machine via `mamba env create -f ./environment.yml`, `pip install -e .`, and `pip install -r dev-requirements.txt`, as outlined in the README.
Note that the `scripts/initial_setup.sh` scripts performs some adjustments to your mamba and docker settings, depending on your local system (e.g., choose correct Pytorch channel when on macOS, or enable CUDA).

Second, you can use a Docker container.
We provide a Modyn base container where the mamba setup is already done. 
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
This flag checks whether gRPC is available locally on your system and uses this installation for rapid development, instead of rebuilding gRPC from source everytime. Using pre-compiled gRPC can speed up the build process significantly.
Furthermore, this avoids cloning the gRPC repository during the cmake build process.
In order to install gRPC on your system, you can either use your system's package manager or run the following instructions:


#### `DMODYN_TRY_LOCAL_GRPC="ON"`

If you installed gPRC on your system with a package manager, there is nothing to do.

##### Build grpc from source once

> **Note**: CLion provides nice CMake integration, which can be used to build both modyn and grpc. We won't show how to specify the CLion equivalents of the commandline variants here though.

> **Warning**: Adjust the following variables:

- **`<grpc_src_dir>`**: the directory where you want to clone the grpc repository: e.g. `~/vendor/grpc`
- **`<build_type>`**: "Debug" or "Release" ("Release" is recommended)
- **`<grpc_install_prefix>`**: the directory where grpc binaries should be installed
  We recommend using a build specific directory: e.g.
  - `~/vendor/lib/dev/`
  - `~/vendor/lib/release/`

```bash
# Clone the grpc repository into a directory of your choice (e.g., `~/vendor/grpc`).:
git clone --recurse-submodules -b v1.59.2 --depth 1 --shallow-submodules https://github.com/grpc/grpc <grpc_src_dir>

#  Create a build directory
cd <grpc_src_dir> && mkdir -p cmake/build && cd cmake/build

# Run cmake to configure the build
cmake \
    -DCMAKE_INSTALL_PREFIX="<grpc_install_prefix>" \
    -DCMAKE_BUILD_TYPE=<build_type> \
    -DgRPC_PROTOBUF_PROVIDER=module \
    -DABSL_ENABLE_INSTALL=On \
    -DgRPC_BUILD_CSHARP_EXT=Off \
    -DABSL_BUILD_TESTING=Off \
    -DgRPC_INSTALL=ON \
    -DgRPC_BUILD_TESTS=Off \
    ../..

# Build and install grpc (this will take a while)
make -j8

# Install grpc into the install prefix directory
make install

# Inspect the linked libraries
ls <grpc_install_prefix>
```

Even though `Release` should be enough for most cases, you might want to build `Debug` as well (using a different install prefix).

##### Build modyn using those grpc binaries

> **Note**: CLion can again be useful here for automatic CMake configuration.

> **Warning**: Adjust the following variables:

- **`<build_type>`**: "Debug" or "Release"
- **`<modyn_src_dir>`**: the directory where you cloned the modyn repository
- **`<grpc_install_prefix>`**: the directory where grpc binaries are installed

```bash
# Enter the modyn source directory
cd <modyn_src_dir>

# Create a build directory
mkdir -p build && cd build

# Run cmake to configure the build
cmake -G "Unix Makefiles" \
    -DCMAKE_PREFIX_PATH="<grpc_install_prefix>" \
    -DCMAKE_BUILD_TYPE=<build_type> \
    -DMODYN_BUILD_STORAGE="ON" \
    -DMODYN_BUILD_PLAYGROUND="ON" \
    -DMODYN_TRY_LOCAL_GRPC="ON" \
    -DMODYN_BUILD_TESTS="ON" \
    -DMODYN_TEST_COVERAGE="OFF" \
    ..

# Build modyn
make -j8

# Run the tests
./build/modyn/tests/modyn-test
```

If you run into problems with the system gRPC installation, use the `-DMODYN_TRY_LOCAL_GRPC=OFF` option described below.

#### `DMODYN_TRY_LOCAL_GRPC="OFF"` (clones grpc repo into modyn directory)

You can invoke CMake with the `-DMODYN_TRY_LOCAL_GRPC=OFF` and let Cmake download and build gRPC for you.

This option will try to rebuild gRPC from source every time you compile modyn which can be time-consuming.

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
