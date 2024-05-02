set(MODYN_LOCALSTORAGEBACKEND_SOURCES
    ./src/local_storage_backend.cpp
    ./src/local_storage_backend_wrapper.cpp
)

# Explicitly set all header files so that IDEs will recognize them as part of the project
set(MODYN_LOCALSTORAGEBACKEND_HEADERS
    ./include/local_storage_backend.hpp
)
add_library(local_storage_backend SHARED)
    
set_target_properties(local_storage_backend
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)



target_sources(local_storage_backend PRIVATE ${MODYN_LOCALSTORAGEBACKEND_HEADERS} ${MODYN_LOCALSTORAGEBACKEND_SOURCES})
target_include_directories(local_storage_backend PUBLIC include)
target_compile_options(local_storage_backend PRIVATE ${MODYN_COMPILE_OPTIONS} -Wno-unused-function)
target_link_libraries(local_storage_backend PUBLIC modyn)


