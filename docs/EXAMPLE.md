# Example: Running a Pipeline

Here, we give an example on how to run an experiment on a Google Cloud VM with an NVIDIA GPU.
We assume you have NVIDIA drivers/CUDA, `docker`, and `nvidia-docker` installed.
Furthermore, we assume that a disk is mounted at `/mnt/datasets` where we can store datasets.

Generally, in `modyn/config/examples`, you will find both an example pipeline configuration and system configuration.
Further pipelines can be found in the `benchmarks` directory.

### [Optional] Creating an example dataset
If you do not have a dataset yet, you can create an example MNIST dataset.
In the `benchmark/mnist` subdirectory, run `python data_generation.py --timestamps INCREASING --dir /mnt/datasets/mnist` to download the dataset to `/mnst/datasets/mnist`.

### Running the initial setup
Please run the `initial_setup.sh` file, if you have not configured Modyn already.
This is necessary if you want to run model training on a GPU.
This will ensure that we do not use the CPU, but CUDA version of PyTorch.
If necessary, you can adjust the CUDA version in this file.

### Adjusting the docker-compose configurations
Next, we need to update the `docker-compose.yml` file to reflect the local setup.
First, for the `trainer_server` service, you should verify that the `runtime` and `deploy` options are enabled such that we have access to the GPU in the trainer server container (the script should have taken care of that),
Next, for the `storage` service, you should uncomment the `volumes` option to mount `/mnt/datasets` to `/datasets` in the container.
Optionally, you can uncomment all lines increasing the `shm_size` of the containers.
This is required for some trainings, e.g., for Criteo training, as you run into OOM errors otherwise.
Optionally, you can uncomment the `.:/modyn_host` mount for all services to enable faster development cycles.
This is not required if you do not iterate.

### Starting the containers and the pipeline
Next, run `./run_modyn.sh` to build the containers and start them. 
This may take several minutes for the first time.
After building the containers, run `tmuxp load tmuxp.yaml` to have access to all container shells and logs.
Switch to the supervisor pane (using regular tmux bindings).
There, you can now submit a pipeline using the `modyn-supervisor` command.
For example, you can run `modyn-supervisor --start-replay-at 0 benchmark/mnist/mnist.yaml modyn/config/examples/modyn_config.yaml`.

### Iterating (for development)
Since we copy the Modyn sources into the containers, if we change something locally outside of the containers, this does not get reflected in the containers.
To avoid rebuilding the containers every time you want to test a change, if you mounted `/modyn_host` into the containers, you can switch into the pane of the service you want to update, and run `cd /modyn_host && pip install -e .`.
Afterwards, run `docker compose restart <SERVICE>` on the host machine to reflect the changes on the running system.
