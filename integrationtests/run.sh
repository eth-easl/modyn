#!/bin/bash
set -e # stops execution on non zero exit code

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
MODYN_CONFIG_PATH=${MODYN_CONFIG_PATH:-$SCRIPT_DIR/../../modyn/config/examples}
MODYNCLIENT_CONFIG_PATH=${MODYNCLIENT_CONFIG_PATH:-$SCRIPT_DIR/../../modynclient/config/examples}
MODYN_INTEGRATIONTESTS_CONFIG_PATH=${MODYN_INTEGRATIONTESTS_CONFIG_PATH:-$SCRIPT_DIR/../config}

echo "Integration tests are located in $SCRIPT_DIR"
echo "Running as user $USER"

echo "Running basic availability tests"
python $SCRIPT_DIR/test_docker_compose.py
echo "Running FTP availability tests"
python $SCRIPT_DIR/test_ftp_connections.py
echo "Running storage integration tests"
python $SCRIPT_DIR/storage/integrationtest_storage.py
python $SCRIPT_DIR/storage/integrationtest_storage_csv.py
python $SCRIPT_DIR/storage/integrationtest_storage_binary.py
echo "Running selector integration tests"
python $SCRIPT_DIR/selector/integrationtest_selector.py
echo "Running online datasets integration tests"
python $SCRIPT_DIR/online_dataset/test_online_dataset.py
echo "Running model storage integration tests"
python $SCRIPT_DIR/model_storage/integrationtest_model_storage.py
echo "Running evaluator integration tests"
python $SCRIPT_DIR/evaluator/integrationtest_evaluator.py
echo "Running supervisor integration tests"
python $SCRIPT_DIR/supervisor/integrationtest_supervisor.py
# metadata processor is not used in currently
#echo "Running metadata processor integration tests"
#python $SCRIPT_DIR/metadata_processor/integrationtest_metadata_processor.py
echo "Running modynclient integration tests"
python $SCRIPT_DIR/modynclient/integrationtest_modynclient.py
echo "Successfully ran all integration tests."
