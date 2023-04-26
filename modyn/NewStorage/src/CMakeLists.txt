set(BINARY ${CMAKE_PROJECT_NAME})

file(GLOB_RECURSE SOURCES LIST_DIRECTORIES true *.h *.cpp)

set(SOURCES ${SOURCES})

add_executable(${BINARY}_run ${SOURCES})

add_library(${BINARY}_lib STATIC ${SOURCES})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

target_link_libraries(${BINARY}_run PUBLIC spdlog argparse ${BINARY}_lib yaml-cpp)
