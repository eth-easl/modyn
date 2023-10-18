include(FetchContent)
list(APPEND CMAKE_PREFIX_PATH /opt/homebrew/opt/libpq) # for macOS builds

# Configure path to modules (for find_package)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/modules/")

################### spdlog ####################
message(STATUS "Making spdlog available.")
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.12.0
)
FetchContent_MakeAvailable(spdlog)

################### fmt ####################
message(STATUS "Making fmt available.")
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 10.1.1
)
FetchContent_MakeAvailable(fmt)

################### argparse ####################
message(STATUS "Making argparse available.")
FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
  GIT_TAG v2.9
)
FetchContent_MakeAvailable(argparse)

################### googletest ####################
message(STATUS "Making googletest available.")

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

################### rapidcsv ####################
message(STATUS "Making rapidcsv available.")

FetchContent_Declare(
  rapidcsv
  GIT_REPOSITORY https://github.com/d99kris/rapidcsv.git
  GIT_TAG v8.80
)
FetchContent_MakeAvailable(rapidcsv)

################### libpq++ ####################
find_package(PostgreSQL REQUIRED) # This needs to be installed on the system - cannot do a lightweight CMake install

################### soci ####################
message(STATUS "Making soci available.")

FetchContent_Declare(
  soci
  GIT_REPOSITORY https://github.com/SOCI/soci.git
  GIT_TAG v4.0.3
)
set(SOCI_TESTS OFF CACHE BOOL "soci configuration")
set(SOCI_CXX11 ON CACHE BOOL "soci configuration")
set(SOCI_STATIC ON CACHE BOOL "soci configuration")
set(SOCI_SHARED ON CACHE BOOL "soci configuration")
set(SOCI_EMPTY OFF CACHE BOOL "soci configuration")
set(SOCI_HAVE_BOOST OFF CACHE BOOL "configuration" FORCE)

FetchContent_GetProperties(soci)
if(NOT soci_POPULATED)
    FetchContent_Populate(soci)
    add_subdirectory(${soci_SOURCE_DIR} _deps)
endif()

# Function to help us fix compiler warnings for all soci targets
function(get_all_targets src_dir var)
    set(targets)
    get_all_targets_recursive(targets ${src_dir})
    set(${var} ${targets} PARENT_SCOPE)
endfunction()

macro(get_all_targets_recursive targets dir)
    get_property(subdirectories DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
    foreach(subdir ${subdirectories})
        get_all_targets_recursive(${targets} ${subdir})
    endforeach()

    get_property(current_targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
    list(APPEND ${targets} ${current_targets})
endmacro()

get_all_targets(${soci_SOURCE_DIR} all_soci_targets)
foreach(_soci_target IN LISTS all_soci_targets)
    target_compile_options(${_soci_target} INTERFACE -Wno-shadow -Wno-zero-as-null-pointer-constant -Wno-pedantic -Wno-undef)
endforeach()


################### yaml-cpp ####################
message(STATUS "Making yaml-cpp available.")

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG yaml-cpp-0.7.0
)
FetchContent_MakeAvailable(yaml-cpp)

target_compile_options(yaml-cpp INTERFACE -Wno-shadow -Wno-pedantic -Wno-deprecated-declarations)

################### gRPC ####################
message(STATUS "Making gRPC available (this may take a while).")

set(gRPC_PROTOBUF_PROVIDER "module" CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL ON)  # https://github.com/protocolbuffers/protobuf/issues/12185
FetchContent_Declare(
  gRPC
  GIT_REPOSITORY https://github.com/grpc/grpc
  GIT_TAG        v1.54.0
  GIT_SHALLOW TRUE
)
set(gRPC_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(gRPC_BUILD_CSHARP_EXT OFF CACHE BOOL "" FORCE)
set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)

set(FETCHCONTENT_QUIET OFF)
FetchContent_MakeAvailable(gRPC)
set(FETCHCONTENT_QUIET ON)

file(DOWNLOAD
    https://raw.githubusercontent.com/protocolbuffers/protobuf/v23.1/cmake/protobuf-generate.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/protobuf-generate.cmake)
include(${CMAKE_CURRENT_BINARY_DIR}/protobuf-generate.cmake)

message(STATUS "Processed gRPC.")
