### Make modyn-storage-library lib available as target in next steps ###
add_library(modyn-storage-library)

set(MODYN_STORAGE_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

##### modyn-storage-library #####
add_subdirectory(src/)

### Main binary ###
add_executable(modyn-storage src/main.cpp)
target_link_libraries(modyn-storage PRIVATE modyn modyn-storage-library argparse spdlog)
