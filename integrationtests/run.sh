#!/bin/bash
set -e # stops execution on non zero exit code

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "Integration tests are located in $SCRIPT_DIR"

echo "Running basic availability tests"
python $SCRIPT_DIR/test_docker_compose.py
echo "Running storage integration tests"
python $SCRIPT_DIR/storage/integrationtest_storage.py
echo "Running selector integration tests"
python $SCRIPT_DIR/selector/integrationtest_selector.py
echo "Successfuly ran all integration tests."