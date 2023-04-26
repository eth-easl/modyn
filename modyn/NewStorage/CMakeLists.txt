cmake_minimum_required(VERSION 3.10)
project(NewStorage)

set(CMAKE_CXX_STANDARD 23)

include_directories(
    src
    lib/yaml-cpp/include
    lib/googletest/googletest/include
    lib/argparse/include
    lib/spdlog/include
)

add_subdirectory(src)
add_subdirectory(tst)
add_subdirectory(lib/yaml-cpp)
add_subdirectory(lib/googletest)
add_subdirectory(lib/argparse)
add_subdirectory(lib/spdlog)
