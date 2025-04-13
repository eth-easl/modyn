##### Common Files for Modyn Library #####
add_subdirectory("common")

##### Local Storage Backend #####
if (IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/selector")
  message(STATUS "Selector LocalStorageBackend is included in this build.")
  # When building the storage container, the selector subdirectory does not exist and is not required.
  add_subdirectory("selector")
endif()

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
