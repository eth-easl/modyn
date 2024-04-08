# !/bin/bash

set -x

PIPELINE_CONFIG_DIR=exp_infra/pipeline_configs/yearbook
MODYN_CONFIG=modynclient/config/examples/modyn_client_config.yaml
REMOTE_LOG_DIR=/tmp/eval
LOCAL_LOG_DIR=~/modyn_exp_results/modyn_results

BATCH_SIZE=10

START_YEAR=1930
END_YEAR=2014

for ((i=$START_YEAR; i<=$END_YEAR; i+=$BATCH_SIZE))
do
    diff_start=$(($i - $START_YEAR))
    tmp_end_year=$(($i + $BATCH_SIZE))
    end_year=$(( $tmp_end_year < $END_YEAR ? $tmp_end_year : $END_YEAR ))
    diff_end=$(($tmp_end_year - $START_YEAR))

    ts_start=$(TZ=UTC date -d "19700101 + $diff_start days" +%s)
    ts_end=$(TZ=UTC date -d "19700101 + $diff_end days" +%s)

    ./modynclient/client/modyn-client --start-replay-at $ts_start --stop-replay-at $ts_end $PIPELINE_CONFIG_DIR/yearbook_time.yaml $MODYN_CONFIG $REMOTE_LOG_DIR --log-directory $LOCAL_LOG_DIR
done

# ./modynclient/client/modyn-client --start-replay-at 0 $PIPELINE_CONFIG_DIR/yearbook_amount.yaml $MODYN_CONFIG $REMOTE_LOG_DIR --log-directory $LOCAL_LOG_DIR