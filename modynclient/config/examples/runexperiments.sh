#!/bin/bash

# Function to kill GPU processes


# Run experiments
#modyn-client --start-replay-at 0 --maximum-triggers 1 /scratch/sjohn/modyn/modynclient/config/examples/lora.yaml /scratch/sjohn/modyn/modynclient/config/examples/modyn_client_config.yaml
#kill_gpu_processes

#modyn-client --start-replay-at 0 --maximum-triggers 1 /scratch/sjohn/modyn/modynclient/config/examples/highlr.yaml /scratch/sjohn/modyn/modynclient/config/examples/modyn_client_config.yaml
#kill_gpu_processes

modyn-client --start-replay-at 0 --maximum-triggers 1 /scratch/sjohn/modyn/modynclient/config/examples/claridencleaned.yaml /scratch/sjohn/modyn/modynclient/config/examples/modyn_client_config.yaml
#kill_gpu_processes
sleep 900
/scratch/sjohn/modyn/scripts/run_modyn.sh
sleep 900
modyn-client --start-replay-at 0 --maximum-triggers 1 /scratch/sjohn/modyn/modynclient/config/examples/unfiltered.yaml /scratch/sjohn/modyn/modynclient/config/examples/modyn_client_config.yaml
