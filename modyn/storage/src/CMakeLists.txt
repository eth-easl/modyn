set(MODYN_STORAGE_SOURCES
    storage_server.cpp
    internal/database/storage_database_connection.cpp
    internal/database/cursor_handler.cpp
    internal/file_watcher/file_watcher_watchdog.cpp
    internal/file_watcher/file_watcher.cpp
    internal/file_wrapper/binary_file_wrapper.cpp
    internal/file_wrapper/csv_file_wrapper.cpp
    internal/file_wrapper/single_sample_file_wrapper.cpp
    internal/filesystem_wrapper/local_filesystem_wrapper.cpp
    internal/grpc/storage_grpc_server.cpp
    internal/grpc/storage_service_impl.cpp
)

# Explicitly set all header files so that IDEs will recognize them as part of the project
set(MODYN_STORAGE_HEADERS
    ../include/storage_server.hpp
    ../include/internal/database/storage_database_connection.hpp
    ../include/internal/database/cursor_handler.hpp
    ../include/internal/file_watcher/file_watcher_watchdog.hpp
    ../include/internal/file_watcher/file_watcher.hpp
    ../include/internal/file_wrapper/file_wrapper.hpp
    ../include/internal/file_wrapper/binary_file_wrapper.hpp
    ../include/internal/file_wrapper/single_sample_file_wrapper.hpp
    ../include/internal/file_wrapper/csv_file_wrapper.hpp
    ../include/internal/file_wrapper/file_wrapper_utils.hpp
    ../include/internal/filesystem_wrapper/filesystem_wrapper.hpp
    ../include/internal/filesystem_wrapper/local_filesystem_wrapper.hpp
    ../include/internal/filesystem_wrapper/filesystem_wrapper_utils.hpp
    ../include/internal/grpc/storage_grpc_server.hpp
    ../include/internal/grpc/storage_service_impl.hpp
    )

set(MODYN-STORAGE_PROTOS
    ../../protos/storage.proto
)

add_library(modyn-storage-proto ${MODYN-STORAGE_PROTOS})

# We output the proto generated headers into the generated directory
# However, CMAKE_CURRENT_BINARY_DIR includes "src", such that the directory is [...]/src/../generated
# This is fine here, but then clang-tidy starts to match the auto-generated files, which we do not want
# Hence, we have to take the realpath of this directory.
# We have to generate the directory first to make realpath work.
set(PROTO_BINARY_DIR_REL "${CMAKE_CURRENT_BINARY_DIR}/../generated")
file(MAKE_DIRECTORY ${PROTO_BINARY_DIR_REL})
execute_process(COMMAND realpath ${PROTO_BINARY_DIR_REL} OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PROTO_BINARY_DIR)

protobuf_generate(
    TARGET modyn-storage-proto
    OUT_VAR PROTO_GENERATED_FILES
    IMPORT_DIRS ../../protos
    PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")

protobuf_generate(
    TARGET modyn-storage-proto
    OUT_VAR PROTO_GENERATED_FILES
    LANGUAGE grpc
    GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
    PLUGIN "protoc-gen-grpc=\$<TARGET_FILE:grpc_cpp_plugin>"
    # PLUGIN_OPTIONS "generate_mock_code=true"
    IMPORT_DIRS ../../protos
    PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")

message(STATUS "Proto generated files in ${PROTO_BINARY_DIR}: ${PROTO_GENERATED_FILES}")

target_include_directories(modyn-storage-proto PUBLIC "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}>")

target_link_libraries(modyn-storage-proto PUBLIC libprotobuf grpc++)
target_compile_options(modyn-storage-proto INTERFACE -Wno-unused-parameter -Wno-c++98-compat-extra-semi -Wno-conditional-uninitialized -Wno-documentation)

target_sources(modyn-storage-library PRIVATE ${MODYN_STORAGE_HEADERS} ${MODYN_STORAGE_SOURCES})
target_include_directories(modyn-storage-library PUBLIC ../include ${CMAKE_CURRENT_BINARY_DIR}/../clang-tidy-build/_deps/soci-src/include ${CMAKE_CURRENT_BINARY_DIR}/../build/_deps/soci-src/include ${CMAKE_CURRENT_BINARY_DIR}/_deps/include ${CMAKE_CURRENT_BINARY_DIR}/../_deps/include ${FETCHCONTENT_BASE_DIR}/include ${soci_SOURCE_DIR}/build/include ${PostgreSQL_INCLUDE_DIRS})
target_compile_options(modyn-storage-library PRIVATE ${MODYN_COMPILE_OPTIONS})

target_link_libraries(modyn-storage-library PUBLIC modyn yaml-cpp ${PostgreSQL_LIBRARIES} soci_postgresql_static soci_sqlite3_static soci_core_static grpc++ grpc++_reflection modyn-storage-proto rapidcsv)

message(STATUS "Current dir: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Current binary dir: ${CMAKE_CURRENT_BINARY_DIR}")

target_compile_definitions(modyn-storage-library PRIVATE MODYN_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\")
target_compile_definitions(modyn-storage-library PRIVATE "MODYN_CMAKE_COMPILER=\"${MODYN_COMPILER_ENV} ${CMAKE_CXX_COMPILER}\"")
target_compile_definitions(modyn-storage-library PUBLIC ${MODYN_COMPILE_DEFINITIONS})

# This adds a `INCLUDE_DIRECTORIES` definition containing all include directories, separate by comma.
# The definition is set to PRIVATE, so it will not be exposed if the target is itself a dependency.
set(INCLUDE_EXPR "$<TARGET_PROPERTY:modyn-storage-library,INCLUDE_DIRECTORIES>")
set(INCLUDE_FILTER "$<FILTER:${INCLUDE_EXPR},INCLUDE,.*/.*>")
set(INCLUDE_JOINED "$<JOIN:${INCLUDE_FILTER},,>")
target_compile_definitions(modyn-storage-library PRIVATE "INCLUDE_DIRECTORIES=\"${INCLUDE_JOINED}\"")