set(MODYNSTORAGE_SOURCES
    storage.cpp
    internal/database/storage_database_connection.cpp
    internal/file_watcher/file_watchdog.cpp
    internal/file_watcher/file_watcher.cpp
    internal/file_wrapper/binary_file_wrapper.cpp
    internal/file_wrapper/single_sample_file_wrapper.cpp
    internal/filesystem_wrapper/local_filesystem_wrapper.cpp
    internal/grpc/storage_service_impl.cpp
)

# Explicitly set all header files so that IDEs will recognize them as part of the project
set(MODYNSTORAGE_HEADERS
    ../include/storage.hpp
    ../include/internal/database/storage_database_connection.hpp
    ../include/internal/file_watcher/file_watchdog.hpp
    ../include/internal/file_watcher/file_watcher.hpp
    ../include/internal/file_wrapper/file_wrapper.hpp
    ../include/internal/file_wrapper/binary_file_wrapper.hpp
    ../include/internal/file_wrapper/single_sample_file_wrapper.hpp
    ../include/internal/filesystem_wrapper/filesystem_wrapper.hpp
    ../include/internal/filesystem_wrapper/local_filesystem_wrapper.hpp
    ../include/internal/grpc/storage_grpc_server.hpp
    ../include/internal/utils/utils.hpp
    )

set(MODYNSTORAGE_PROTOS
    ../../protos/storage.proto
)

add_library(modynstorage-proto ${MODYNSTORAGE_PROTOS})
target_link_libraries(modynstorage-proto
    PUBLIC
        libprotobuf
        grpc++
)

set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY ${PROTO_BINARY_DIR})

protobuf_generate(
    TARGET modynstorage-proto
    OUT_VAR PROTO_GENERATED_FILES
    IMPORT_DIRS ../../protos
    PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")
set_source_files_properties(${PROTO_GENERATED_FILES} PROPERTIES SKIP_UNITY_BUILD_INCLUSION on)
protobuf_generate(
    TARGET modynstorage-proto
    OUT_VAR PROTO_GENERATED_FILES
    LANGUAGE grpc
    GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
    PLUGIN "protoc-gen-grpc=\$<TARGET_FILE:grpc_cpp_plugin>"
    # PLUGIN_OPTIONS "generate_mock_code=true"
    IMPORT_DIRS ../../protos
    PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")
set_source_files_properties(${PROTO_GENERATED_FILES} PROPERTIES SKIP_UNITY_BUILD_INCLUSION on)

target_include_directories(modynstorage-proto PUBLIC "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}>")
target_compile_options(modynstorage-proto INTERFACE -Wno-unused-parameter -Wno-c++98-compat-extra-semi -Wno-conditional-uninitialized -Wno-documentation)

target_sources(modynstorage PRIVATE ${MODYNSTORAGE_HEADERS} ${MODYNSTORAGE_SOURCES})
target_include_directories(modynstorage PUBLIC ../include ${CMAKE_CURRENT_BINARY_DIR}/../cmake-build-debug/clang-tidy-build/_deps/soci-src/include ${CMAKE_CURRENT_BINARY_DIR}/../build/_deps/soci-src/include ${soci_SOURCE_DIR}/build/include ${PostgreSQL_INCLUDE_DIRS})
target_compile_options(modynstorage PRIVATE ${MODYNSTORAGE_COMPILE_OPTIONS})

target_link_libraries(modynstorage PUBLIC spdlog argparse yaml-cpp ${PostgreSQL_LIBRARIES} soci_postgresql soci_sqlite3 soci_core grpc++ modynstorage-proto)

message(STATUS "Current dir: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Current binary dir: ${CMAKE_CURRENT_BINARY_DIR}")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
target_compile_definitions(modynstorage PRIVATE MODYNSTORAGE_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\")

# AppleClang needs the SDKROOT setup in order to be used directly. This is not the case for clang++ or g++.
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang")
  message(STATUS "Detected AppleClang compiler. Setting up AppleClang SDKROOT.")
  message(STATUS "  SDKROOT: ${CMAKE_OSX_SYSROOT}")
  set(MODYNSTORAGE_COMPILER_ENV "SDKROOT=${CMAKE_OSX_SYSROOT}")
endif ()

target_compile_definitions(modynstorage PRIVATE "MODYNSTORAGE_CMAKE_COMPILER=\"${MODYNSTORAGE_COMPILER_ENV} ${CMAKE_CXX_COMPILER}\"")

target_compile_definitions(modynstorage PUBLIC ${MODYNSTORAGE_COMPILE_DEFINITIONS})
message(STATUS "System-specific compile definitions: ${MODYNSTORAGE_COMPILE_DEFINITIONS}")

# This adds a `INCLUDE_DIRECTORIES` definition containing all include directories, separate by comma.
# The definition is set to PRIVATE, so it will not be exposed if the target is itself a dependency.
set(INCLUDE_EXPR "$<TARGET_PROPERTY:modynstorage,INCLUDE_DIRECTORIES>")
set(INCLUDE_FILTER "$<FILTER:${INCLUDE_EXPR},INCLUDE,.*/.*>")
set(INCLUDE_JOINED "$<JOIN:${INCLUDE_FILTER},,>")
target_compile_definitions(modynstorage PRIVATE "INCLUDE_DIRECTORIES=\"${INCLUDE_JOINED}\"")