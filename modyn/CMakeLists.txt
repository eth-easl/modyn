
##### Common Files for Modyn Library #####
add_subdirectory("common-cpp")

##### MODYN STORAGE BINARY #####
# TODO(MaxiBoether): add while merging storage PR
#add_executable(modyn-storage storage/src/main.cpp)
#target_link_libraries(modyn-storage PRIVATE modyn)

##### PLAYGROUND #####
if (${MODYN_BUILD_PLAYGROUND})
  message(STATUS "Playground is included in this build.")
  add_subdirectory(playground)
endif ()

##### TESTING #####
if (${MODYN_BUILD_TESTS})
  message(STATUS "Tests are included in this build.")
  enable_testing()
  add_subdirectory(tests)
endif ()
