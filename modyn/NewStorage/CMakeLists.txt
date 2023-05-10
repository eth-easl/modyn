cmake_minimum_required(VERSION 3.10)
project(NewStorage)

set(CMAKE_CXX_STANDARD 23)

find_package(PostgreSQL REQUIRED)

# Install Dependencies:
include(FetchContent)

message(STATUS "Making dependencies available.")

################### spdlog ####################
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.11.0
)
FetchContent_MakeAvailable(spdlog)

################### argparse ####################
FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
  GIT_TAG v2.9
)
FetchContent_MakeAvailable(argparse)

################### googletest ####################
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.13.0
)
FetchContent_MakeAvailable(googletest)

################### soci ####################
FetchContent_Declare(
  soci
  GIT_REPOSITORY https://github.com/SOCI/soci.git
  GIT_TAG v4.0.3
)
set(SOCI_TESTS OFF)
set(SOCI_CXX11 ON)
FetchContent_MakeAvailable(soci)

################### yaml-cpp ####################
FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG yaml-cpp-0.7.0
)
FetchContent_MakeAvailable(yaml-cpp)

################### grpc ####################
FetchContent_Declare(
  grpc
  GIT_REPOSITORY https://github.com/grpc/grpc.git
  GIT_TAG v1.54.1
)
FetchContent_MakeAvailable(grpc)

include_directories(
    src
    cmake
    ${PostgreSQL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}/_deps/soci-build/include  # soci is generating the soci-config.h file here
)

add_subdirectory(src)
add_subdirectory(test)
