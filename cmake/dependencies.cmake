include(FetchContent)

# Configure path to modules (for find_package)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/modules/")

################### spdlog ####################
message(STATUS "Making spdlog available.")
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.12.0
)
FetchContent_MakeAvailable(spdlog)

################### fmt ####################
message(STATUS "Making fmt available.")
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 10.1.1
)
FetchContent_MakeAvailable(fmt)

################### argparse ####################
message(STATUS "Making argparse available.")
FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
  GIT_TAG v2.9
)
FetchContent_MakeAvailable(argparse)

################### googletest ####################
message(STATUS "Making googletest available.")

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

if (${MODYN_BUILD_STORAGE})
  message(STATUS "Including storage dependencies.")
  include(${MODYN_CMAKE_DIR}/storage_dependencies.cmake)
endif ()
