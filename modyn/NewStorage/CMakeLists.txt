cmake_minimum_required(VERSION 3.10)
project(NewStorage)

set(CMAKE_CXX_STANDARD 23)

find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(PostgreSQL REQUIRED)

include_directories(
    src
    lib/yaml-cpp/include
    lib/googletest/googletest/include
    lib/argparse/include
    lib/spdlog/include
    ${Boost_INCLUDE_DIRS}
    lib/soci/include
    ${CMAKE_CURRENT_BINARY_DIR}/lib/soci/include
    ${PostgreSQL_INCLUDE_DIRS}
)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(lib/yaml-cpp)
add_subdirectory(lib/googletest)
add_subdirectory(lib/argparse)
add_subdirectory(lib/spdlog)
add_subdirectory(lib/soci)
add_subdirectory(executables/file_watchdog)
add_subdirectory(executables/file_watcher)