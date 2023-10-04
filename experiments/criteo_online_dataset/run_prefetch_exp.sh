#!/usr/bin/env bash

BASEDIR="/modyn_host/eval/criteo_dataset_$(date +%s)"


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
MODYN_CONFIG_PATH="$SCRIPT_DIR/../../modyn/config/examples/modyn_config.yml"

for filename in $SCRIPT_DIR/pipelines/*.yml; do
    BASE=$(basename "$filename" | cut -d. -f1)
    EVAL_DIR="$BASEDIR/$BASE"
    mkdir -p $EVAL_DIR
    modyn-supervisor --start-replay-at 0 --maximum-triggers 1 $filename $MODYN_CONFIG_PATH $EVAL_DIR
done
