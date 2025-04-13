#!/bin/bash
/scratch/sjohn/modyn/scripts/run_modyn.sh
sleep 60
modyn-client --start-replay-at 0  /scratch/sjohn/modyn/modynclient/config/examples/abstractscoreset.yaml /scratch/sjohn/modyn/modynclient/config/examples/modyn_client_config.yaml