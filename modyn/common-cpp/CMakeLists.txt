set(MODYN_SOURCES
    src/utils/utils.cpp
)

# Explicitly set all header files so that IDEs will recognize them as part of the project
set(MODYN_HEADERS
    ./include/modyn/utils/utils.hpp
)

target_sources(modyn PRIVATE ${MODYN_HEADERS} ${MODYN_SOURCES})
target_include_directories(modyn PUBLIC include)
target_compile_options(modyn PRIVATE ${MODYN_COMPILE_OPTIONS})

target_link_libraries(modyn PUBLIC spdlog fmt)

message(STATUS "Current dir: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Current binary dir: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "System-specific compile definitions: ${MODYN_COMPILE_DEFINITIONS}")

target_compile_definitions(modyn PRIVATE MODYN_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\")
target_compile_definitions(modyn PRIVATE "MODYN_CMAKE_COMPILER=\"${MODYN_COMPILER_ENV} ${CMAKE_CXX_COMPILER}\"")
target_compile_definitions(modyn PUBLIC ${MODYN_COMPILE_DEFINITIONS})

# This adds a `INCLUDE_DIRECTORIES` definition containing all include directories, separate by comma.
# The definition is set to PRIVATE, so it will not be exposed if the target is itself a dependency.
set(INCLUDE_EXPR "$<TARGET_PROPERTY:modyn,INCLUDE_DIRECTORIES>")
set(INCLUDE_FILTER "$<FILTER:${INCLUDE_EXPR},INCLUDE,.*/.*>")
set(INCLUDE_JOINED "$<JOIN:${INCLUDE_FILTER},,>")
target_compile_definitions(modyn PRIVATE "INCLUDE_DIRECTORIES=\"${INCLUDE_JOINED}\"")
