#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"

function run_format() {
    local subdir=$1
    find "${DIR}"/"${subdir}" \( -iname '*.hpp' -o -iname '*.cpp' \) -print0 | xargs -0 clang-format -i
}

run_format "include"
run_format "src"
run_format "test"
