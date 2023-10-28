#!/bin/bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
RUN_CLANG_TIDY=${RUN_CLANG_TIDY:-run-clang-tidy}
CLANG_TIDY=${CLANG_TIDY:-clang-tidy}
BUILD_DIR_REL="${SCRIPT_DIR}/../clang-tidy-build"
mkdir -p "${BUILD_DIR_REL}"
BUILD_DIR=$(realpath ${BUILD_DIR_REL})
APPLY_REPLACEMENTS_BINARY=${APPLY_REPLACEMENTS_BINARY:-clang-apply-replacements}

function run_build() {
    echo "Running cmake build..."
    set -x

    mkdir -p "${BUILD_DIR}"
    cmake -S ${SCRIPT_DIR}/.. -B "${BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_UNITY_BUILD=OFF \
        -DMODYN_BUILD_STORAGE=ON

    pushd ${BUILD_DIR}
    make -j8 modyn-storage-proto
    popd

    cmake -S ${SCRIPT_DIR}/.. -B "${BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_UNITY_BUILD=ON \
        -DCMAKE_UNITY_BUILD_BATCH_SIZE=0 \
        -DMODYN_BUILD_STORAGE=ON

    # Due to the include-based nature of the unity build, clang-tidy will not find this configuration file otherwise:
    ln -fs "${SCRIPT_DIR}"/../modyn/tests/.clang-tidy "${BUILD_DIR}"/modyn/tests/

    set +x
}

function run_tidy() {
    echo "Running clang-tidy using run-clang-tidy..."
    set -x

    fix=$1
    additional_args=""
    if [ "${fix}" == true ]
    then
      additional_args="${additional_args} -fix -clang-apply-replacements-binary ${APPLY_REPLACEMENTS_BINARY}"
      echo "Will also automatically fix everything that we can..."
    fi

    ${RUN_CLANG_TIDY} -p "${BUILD_DIR}" \
        -clang-tidy-binary="${CLANG_TIDY}" \
        -config-file="${SCRIPT_DIR}/../.clang-tidy" \
        -quiet \
        -checks='-bugprone-suspicious-include,-google-global-names-in-headers' \
        -header-filter='(.*modyn/storage/src/.*)|(.*modyn/storage/include/.*)|(.*modyn/common/.*)|(.*modyn/playground/.*)|(.*modyn/selector/.*)|(.*modyn/tests.*)' \
        ${additional_args} \
        "${BUILD_DIR}"/modyn/*/Unity/*.cxx \
        "${BUILD_DIR}"/modyn/*/*/Unity/*.cxx \
        "${BUILD_DIR}"/modyn/*/*/*/Unity/*.cxx \
        "${BUILD_DIR}"/modyn/*/*/*/*/Unity/*.cxx \
        "${BUILD_DIR}"/modyn/*/*/*/*/*/Unity/*.cxx \
        "${BUILD_DIR}"/modyn/*/*/*/*/*/*/Unity/*.cxx \
        "${BUILD_DIR}"/modyn/*/*/*/*/*/*/*/Unity/*.cxx \
        "${BUILD_DIR}"/modyn/*/*/*/*/*/*/*/*/Unity/*.cxx \
        "${BUILD_DIR}"/modyn/*/*/*/*/*/*/*/*/*/Unity/*.cxx \
        "${BUILD_DIR}"/modyn/*/*/*/*/*/*/*/*/*/*/Unity/*.cxx \
        "${BUILD_DIR}"/modyn/*/*/*/*/*/*/*/*/*/*/*/Unity/*.cxx \
        "${BUILD_DIR}"/modyn/*/*/*/*/*/*/*/*/*/*/*/*/Unity/*.cxx \
        "${BUILD_DIR}"/modyn/tests/CMakeFiles/modyn-all-test-sources-for-tidy.dir/Unity/*.cxx \
        "${BUILD_DIR}"/CMakeFiles/modyn.dir/Unity/*.cxx

    set +x
}

# The problem is in the --header-filter option above in RUN_CLANG_TIDY: otherwise, we will match dependency headers as well.
if [[ $PWD =~ "modyn/storage" ]]; then
    echo "Please do not run this script from a directory that has modyn/storage in its path. Current path is ${PWD}."
    exit -1
fi

if [[ $PWD =~ "modyn/selector" ]]; then
    echo "Please do not run this script from a directory that has modyn/selector in its path. Current path is ${PWD}."
    exit -1
fi

if [[ $PWD =~ "modyn/common" ]]; then
    echo "Please do not run this script from a directory that has modyn/common in its path. Current path is ${PWD}."
    exit -1
fi

case $1 in
    "build")
        run_build
        ;;
    "run_tidy")
        run_tidy false
        ;;
    "fix")
        run_tidy true
        ;;
    *)
        run_build
        run_tidy false
        ;;
esac
