set(BINARY ${CMAKE_PROJECT_NAME})

file(GLOB_RECURSE SOURCES LIST_DIRECTORIES true *.hpp *.cpp)

set(SOURCES ${SOURCES})
set(SOCI_HAVE_BOOST OFF CACHE BOOL "SOCI_HAVE_BOOST" FORCE)

add_executable(${BINARY}_run ${SOURCES})

add_library(${BINARY}_lib STATIC ${SOURCES})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(SOCI_SHARED ON)

target_link_libraries(${BINARY}_run PUBLIC spdlog argparse yaml-cpp ${PostgreSQL_LIBRARIES} soci_core soci_postgresql soci_sqlite3 ${BINARY}_lib)
