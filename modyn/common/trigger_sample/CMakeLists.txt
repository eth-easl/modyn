set(MODYN_TRIGGERSAMPLESTORAGE_SOURCES
    ./src/trigger_sample_storage.cpp
    ./src/trigger_sample_storage_wrapper.cpp
)

# Explicitly set all header files so that IDEs will recognize them as part of the project
set(MODYN_TRIGGERSAMPLESTORAGE_HEADERS
    ./include/trigger_sample_storage.hpp
)
add_library(trigger_sample_storage SHARED)

set_target_properties(trigger_sample_storage
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)



target_sources(trigger_sample_storage PRIVATE ${MODYN_TRIGGERSAMPLESTORAGE_HEADERS} ${MODYN_TRIGGERSAMPLESTORAGE_SOURCES})
target_include_directories(trigger_sample_storage PUBLIC include)
target_compile_options(trigger_sample_storage PRIVATE ${MODYN_COMPILE_OPTIONS} -Wno-unused-function)
target_link_libraries(trigger_sample_storage PUBLIC modyn)
