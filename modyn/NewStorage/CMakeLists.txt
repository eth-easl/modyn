cmake_minimum_required(VERSION 3.10)
project(NewStorage)

set(CMAKE_CXX_STANDARD 23)

find_package(Boost REQUIRED)

# Set BUILD_POSTGRESQL_CONNECTOR to ON to build the PostgreSQL connector
set(BUILD_POSTGRESQL_CONNECTOR ON CACHE BOOL "Build PostgreSQL connector" FORCE)
set(BUILD_SQLITE3_CONNECTOR ON CACHE BOOL "Build SQLite3 connector" FORCE)

include_directories(
    src
    lib/yaml-cpp/include
    lib/googletest/googletest/include
    lib/argparse/include
    lib/spdlog/include
    lib/sqlpp11/include
    #lib/grpc/include
    ${Boost_INCLUDE_DIRS}
)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(lib/yaml-cpp)
add_subdirectory(lib/googletest)
add_subdirectory(lib/argparse)
add_subdirectory(lib/spdlog)
add_subdirectory(lib/sqlpp11)
#add_subdirectory(lib/grpc)
