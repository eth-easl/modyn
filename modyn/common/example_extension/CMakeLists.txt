set(MODYN_EXAMPLEEXTENSION_SOURCES
    ./src/example_extension.cpp
    ./src/example_extension_wrapper.cpp
)

# Explicitly set all header files so that IDEs will recognize them as part of the project
set(MODYN_EXAMPLEEXTENSION_HEADERS
    ./include/example_extension.hpp
)
add_library(example_extension SHARED)

set_target_properties(example_extension
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_sources(example_extension PRIVATE ${MODYN_EXAMPLEEXTENSION_HEADERS} ${MODYN_EXAMPLEEXTENSION_SOURCES})
target_include_directories(example_extension PUBLIC include)
target_compile_options(example_extension PRIVATE ${MODYN_COMPILE_OPTIONS} -Wno-unused-function)
target_link_libraries(example_extension PUBLIC modyn)


