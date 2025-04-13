#!/bin/bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BUILD_DIR_REL="${SCRIPT_DIR}/../clang-tidy-build"
mkdir -p "${BUILD_DIR_REL}"
BUILD_DIR=$(realpath "${BUILD_DIR_REL}")

function run_build() {
    echo "Running cmake build..."
    set -x


    # Second config (Unity build + tests enabled)
    cmake -S "${SCRIPT_DIR}/.." -B "${BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_UNITY_BUILD=ON \
        -DCMAKE_UNITY_BUILD_BATCH_SIZE=0 \
        -DMODYN_BUILD_STORAGE=ON \
        -DMODYN_BUILD_TESTS=ON

    # Build again so the tests get compiled under this updated config
    pushd "${BUILD_DIR}"
    make -j8
    popd

    # Make sure clang-tidy finds the config if needed
    ln -fs "${SCRIPT_DIR}/../modyn/tests/.clang-tidy" "${BUILD_DIR}/modyn/tests/"

    set +x
}

function run_tests() {
    echo "Running tests via CTest..."
    set -x
    pushd "${BUILD_DIR}"
    # Optional: "ctest -N" to list test names before running
    ctest -N
    ctest --output-on-failure
    popd
    set +x
}

# Check if we're in a "problematic" directory
if [[ $PWD =~ "modyn/storage" ]] || [[ $PWD =~ "modyn/selector" ]] || [[ $PWD =~ "modyn/common" ]]; then
    echo "Please do not run this script from a directory containing modyn/storage, modyn/selector, or modyn/common in its path. Current path is ${PWD}."
    exit 1
fi

case $1 in
    "build")
        run_build
        ;;
    "test")
        run_tests
        ;;
    *)
        run_build
        run_tests
        ;;
esac
