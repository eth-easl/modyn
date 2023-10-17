#!/bin/bash
set -e

RUN_CLANG_TIDY=${RUN_CLANG_TIDY:-run-clang-tidy}
CLANG_TIDY=${CLANG_TIDY:-clang-tidy}
BUILD_DIR=${BUILD_DIR:-cmake-build-debug/clang-tidy-build}
APPLY_REPLACEMENTS_BINARY=${APPLY_REPLACEMENTS_BINARY:-clang-apply-replacements}

function run_build() {
    echo "Running cmake build..."
    set -x

    mkdir -p "${BUILD_DIR}"
    cmake -B "${BUILD_DIR}"
    cmake -S . -B "${BUILD_DIR}"

    # Due to the include-based nature of the unity build, clang-tidy will not find this configuration file otherwise:
    ln -fs "${PWD}"/tests/.clang-tidy "${BUILD_DIR}"/tests/

    # TODO(MaxiBoether): add when merging storage
    #make -j8 -C "${BUILD_DIR}" modynstorage-proto

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
        -header-filter='(.*modyn/storage/.*)|(.*modyn/common-cpp/.*)|(.*modyn/playground/.*)|(.*modyn/selector/.*)' \
        -checks='-bugprone-suspicious-include,-google-global-names-in-headers' \
        -quiet \
        ${additional_args} \
        "${BUILD_DIR}"/CMakeFiles/modyn.dir/Unity/*.cxx \
        "${BUILD_DIR}"/tests/CMakeFiles/modyn-all-test-sources-for-tidy.dir/Unity/*.cxx
    set +x
}

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
