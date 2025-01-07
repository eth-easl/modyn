include(FetchContent)

# Configure path to modules (for find_package)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/modules/")

################### fmt ####################
message(STATUS "Making fmt available.")
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 11.0.2
)
FetchContent_MakeAvailable(fmt)

################### spdlog ####################
message(STATUS "Making spdlog available.")
set(SPDLOG_FMT_EXTERNAL ON) # Otherwise, we run into linking errors since the fmt version used by spdlog does not match.
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.15.0
)
FetchContent_MakeAvailable(spdlog)

################### argparse ####################
message(STATUS "Making argparse available.")
FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
  GIT_TAG v3.1
)
FetchContent_MakeAvailable(argparse)

################### googletest ####################
message(STATUS "Making googletest available.")

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)
FetchContent_MakeAvailable(googletest)

if (${MODYN_BUILD_STORAGE})
  message(STATUS "Including storage dependencies.")
  include(${MODYN_CMAKE_DIR}/storage_dependencies.cmake)
endif ()

################### yaml-cpp ####################
# Technically, yaml-cpp is currently only required by storage
# But we have a test util function requiring this.

message(STATUS "Making yaml-cpp available.")

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG yaml-cpp-0.7.0
)
FetchContent_MakeAvailable(yaml-cpp)

target_compile_options(yaml-cpp INTERFACE -Wno-shadow -Wno-pedantic -Wno-deprecated-declarations)
