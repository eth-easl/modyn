set(MODYN_TRIGGERSTORAGECPP_SOURCES
    ./src/trigger_storage_cpp.cpp
    ./src/trigger_storage_cpp_wrapper.cpp
)

# Explicitly set all header files so that IDEs will recognize them as part of the project
set(MODYN_TRIGGERSTORAGECPP_HEADERS
    ./include/trigger_storage_cpp.hpp
)
add_library(trigger_storage_cpp SHARED)
    
set_target_properties(trigger_storage_cpp
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)



target_sources(trigger_storage_cpp PRIVATE ${MODYN_TRIGGERSTORAGECPP_HEADERS} ${MODYN_TRIGGERSTORAGECPP_SOURCES})
target_include_directories(trigger_storage_cpp PUBLIC include)
target_compile_options(trigger_storage_cpp PRIVATE ${MODYN_COMPILE_OPTIONS} -Wno-unused-function)
target_link_libraries(trigger_storage_cpp PUBLIC modyn)


