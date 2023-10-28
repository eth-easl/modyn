#!/usr/bin/env bash

PARENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"

function run_format() {
    local subdir=$1
    find "${PARENT_DIR}"/"${subdir}" \( -iname '*.hpp' -o -iname '*.cpp' \) -print | awk '!/build/' | xargs clang-format -i
}

echo "Running on storage directory"
run_format "modyn/storage"
echo "Running on tests directory"
run_format "modyn/tests"
echo "Running on common directory"
run_format "modyn/common"