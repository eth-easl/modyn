# !/bin/bash

PIPELINE_CONFIG_DIR=exp_infra/pipeline_configs/huffpost
MODYN_CONFIG=modynclient/config/examples/modyn_client_config.yaml
REMOTE_LOG_DIR=/tmp/eval
LOCAL_LOG_DIR=~/exp_results

# ./modynclient/client/modyn-client --start-replay-at 0 $PIPELINE_CONFIG_DIR/huffpost_time.yaml $MODYN_CONFIG $REMOTE_LOG_DIR --log-directory $LOCAL_LOG_DIR
# ./modynclient/client/modyn-client --start-replay-at 0 $PIPELINE_CONFIG_DIR/huffpost_amount.yaml $MODYN_CONFIG $REMOTE_LOG_DIR --log-directory $LOCAL_LOG_DIR
./modynclient/client/modyn-client --start-replay-at 0 $PIPELINE_CONFIG_DIR/huffpost_datadrift.yaml $MODYN_CONFIG $REMOTE_LOG_DIR --log-directory $LOCAL_LOG_DIR