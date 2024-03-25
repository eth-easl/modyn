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
Please run the `scripts/initial_setup.sh` file, if you have not configured Modyn already.
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

### Starting the containers
Next, run `./scripts/run_modyn.sh` to build the containers and start them. 
This may take several minutes for the first time.
After building the containers, run `tmuxp load tmuxp.yaml` to have access to all container shells and logs.
 
### Starting the pipeline
You can now submit a pipeline to the supervisor container using the `modyn-client` command in `modynclient/client/`.\
For example, you can run `modyn-client --start-replay-at 0 --maximum-triggers 1 <pipeline config file> <modyn client config file>`\
pipeline config file example: modynclient/config/examples/mnist.yaml\
modyn client config file example:
- on your local machine: modynclient/config/examples/modyn_client_config.yaml
- in one of the containers: modynclient/config/examples/modyn_client_config_container.yaml

The evaluation logs will be stored in supervisor's container, in path `modyn_config['supervisor']['eval_directory']`, see [the modyn config file](modyn/config/examples/modyn_config.yaml).
Note that this is only a temporary solution. In future the logs should be lively transferred to client via, e.g., FTP.

### Iterating (for development)
Since we copy the Modyn sources into the containers, if we change something locally outside of the containers, this does not get reflected in the containers.
To avoid rebuilding the containers every time you want to test a change, if you mounted `/modyn_host` into the containers, you can switch into the pane of the service you want to update, and run `cd /modyn_host && pip install -e .`.
Afterwards, run `docker compose restart <SERVICE>` on the host machine to reflect the changes on the running system.
