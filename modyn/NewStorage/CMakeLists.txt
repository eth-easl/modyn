cmake_minimum_required(VERSION 3.10)
project(NewStorage)

set(CMAKE_CXX_STANDARD 23)

find_package(Boost REQUIRED)

include_directories(
    src
    lib/yaml-cpp/include
    lib/googletest/googletest/include
    lib/argparse/include
    lib/spdlog/include
    #lib/grpc/include
    ${Boost_INCLUDE_DIRS}
    lib/soci/include
)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(lib/yaml-cpp)
add_subdirectory(lib/googletest)
add_subdirectory(lib/argparse)
add_subdirectory(lib/spdlog)
#add_subdirectory(lib/grpc)
add_subdirectory(lib/soci)
