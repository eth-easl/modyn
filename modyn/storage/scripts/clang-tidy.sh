#!/bin/bash
set -e

RUN_CLANG_TIDY=${RUN_CLANG_TIDY:-run-clang-tidy}
CLANG_TIDY=${CLANG_TIDY:-clang-tidy}
BUILD_DIR=${BUILD_DIR:-cmake-build-debug/clang-tidy-build}
APPLY_REPLACEMENTS_BINARY=${APPLY_REPLACEMENTS_BINARY:-clang-apply-replacements}
PROTO_OUT_DIR=${PROTO_OUT_DIR:-${BUILD_DIR}/src/generated}
PROTO_IN_DIR=${PROTO_IN_DIR:-../protos}

function generate_proto() {
    echo "Generating proto files..."
    mkdir -p ${PROTO_OUT_DIR}

    PROTO_FILE=storage.proto
    GRPC_CPP_PLUGIN_PATH=$(which grpc_cpp_plugin)

    protoc \
    -I=${PROTO_IN_DIR} \
    --grpc_out=${PROTO_OUT_DIR} \
    --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN_PATH} \
    --cpp_out=${PROTO_OUT_DIR} \
    ${PROTO_IN_DIR}/${PROTO_FILE}

    echo "Generating proto files...done"
    ls -l ${PROTO_OUT_DIR}
}

function run_build() {
    echo "Running cmake build..."
    set -x

    cmake --build "${BUILD_DIR}" --target modynstorage-proto

    cmake -S . -B "${BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_UNITY_BUILD=ON \
        -DCMAKE_UNITY_BUILD_BATCH_SIZE=0
    # Due to the include-based nature of the unity build, clang-tidy will not find this configuration file otherwise:
    ln -fs "${PWD}"/test/.clang-tidy "${BUILD_DIR}"/test/
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
        -header-filter='(.*modyn/modyn/storage/src/.*)|(.*modyn/modyn/storage/include/.*)|(.*modyn/modyn/storage/test/.*)' \
        -checks='-bugprone-suspicious-include,-google-global-names-in-headers' \
        -quiet \
        ${additional_args} \
        "${BUILD_DIR}"/CMakeFiles/modynstorage.dir/Unity/*.cxx \
        "${BUILD_DIR}"/test/CMakeFiles/modynstorage-all-test-sources-for-tidy.dir/Unity/*.cxx
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
