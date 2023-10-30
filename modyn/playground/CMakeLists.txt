add_executable(playground playground.cpp)
target_compile_options(playground PRIVATE ${MODYN_COMPILE_OPTIONS})
target_link_libraries(playground PRIVATE modyn)

set_target_properties(playground
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
