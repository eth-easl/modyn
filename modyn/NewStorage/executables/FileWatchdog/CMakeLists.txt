set(BINARY FileWatchdog)

file(GLOB_RECURSE SOURCES LIST_DIRECTORIES true *.hpp *.cpp)

set(SOURCES ${SOURCES})

add_executable(${BINARY} ${SOURCES})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(SOCI_SHARED ON)

target_link_libraries(${BINARY} PUBLIC ${Boost_LIBRARIES} spdlog argparse yaml-cpp ${PostgreSQL_LIBRARIES} soci_core soci_postgresql soci_sqlite3 ${CMAKE_PROJECT_NAME}_lib)
