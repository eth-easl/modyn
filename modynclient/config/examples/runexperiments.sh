#!/bin/bash

# Function to kill GPU processes
kill_gpu_processes() {
    echo "Killing all GPU processes..."
    sudo nvidia-smi --gpu-reset  # Reset the GPU
    sudo nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I {} sudo kill -9 {} 2>/dev/null
    echo "GPU processes killed."
    sleep 5  # Allow some time before the next run
}

# Run experiments
#modyn-client --start-replay-at 0 --maximum-triggers 1 /scratch/sjohn/modyn/modynclient/config/examples/lora.yaml /scratch/sjohn/modyn/modynclient/config/examples/modyn_client_config.yaml
#kill_gpu_processes

#modyn-client --start-replay-at 0 --maximum-triggers 1 /scratch/sjohn/modyn/modynclient/config/examples/highlr.yaml /scratch/sjohn/modyn/modynclient/config/examples/modyn_client_config.yaml
#kill_gpu_processes

modyn-client --start-replay-at 0 --maximum-triggers 1 /scratch/sjohn/modyn/modynclient/config/examples/unfiltered.yaml /scratch/sjohn/modyn/modynclient/config/examples/modyn_client_config.yaml
#kill_gpu_processes
sleep 900
/scratch/sjohn/modyn/scripts/run_modyn.sh
sleep 900
modyn-client --start-replay-at 0 --maximum-triggers 1 /scratch/sjohn/modyn/modynclient/config/examples/claridencleaned.yaml /scratch/sjohn/modyn/modynclient/config/examples/modyn_client_config.yaml
#kill_gpu_processes
