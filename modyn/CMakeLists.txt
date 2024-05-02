
##### Common Files for Modyn Library #####
add_subdirectory("common")

##### Local Storage Backend #####
add_subdirectory("selector")

##### MODYN STORAGE BINARY #####
if (${MODYN_BUILD_STORAGE})
  message(STATUS "Storage is included in this build.")
  add_subdirectory(storage)
endif ()

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
