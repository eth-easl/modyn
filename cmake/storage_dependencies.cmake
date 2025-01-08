include(FetchContent)
list(APPEND CMAKE_PREFIX_PATH /opt/homebrew/opt/libpq) # for macOS builds

# Configure path to modules (for find_package)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/modules/")

# Use original download path
message(STATUS "CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}.")
message(STATUS "FETCHCONTENT_BASE_DIR = ${FETCHCONTENT_BASE_DIR}.")

################### libpq++ ####################
find_package(PostgreSQL REQUIRED) # This needs to be installed on the system - cannot do a lightweight CMake install

################### rapidcsv ####################
message(STATUS "Making rapidcsv available.")

FetchContent_Declare(
  rapidcsv
  GIT_REPOSITORY https://github.com/d99kris/rapidcsv.git
  GIT_TAG v8.84
)
FetchContent_MakeAvailable(rapidcsv)

################### soci ####################
message(STATUS "Making soci available.")

FetchContent_Declare(
  soci
  GIT_REPOSITORY https://github.com/SOCI/soci.git
  GIT_TAG v4.0.3
)
set(SOCI_TESTS OFF CACHE BOOL "soci configuration")
set(SOCI_CXX11 ON CACHE BOOL "soci configuration")
set(SOCI_STATIC ON CACHE BOOL "soci configuration")
set(SOCI_SHARED OFF CACHE BOOL "soci configuration")
set(SOCI_EMPTY OFF CACHE BOOL "soci configuration")
set(SOCI_HAVE_BOOST OFF CACHE BOOL "configuration" FORCE)

FetchContent_GetProperties(soci)
if(NOT soci_POPULATED)
    FetchContent_Populate(soci)
    add_subdirectory(${soci_SOURCE_DIR} _deps)
endif()

# Function to help us fix compiler warnings for all soci targets
function(get_all_targets src_dir var)
    set(targets)
    get_all_targets_recursive(targets ${src_dir})
    set(${var} ${targets} PARENT_SCOPE)
endfunction()

macro(get_all_targets_recursive targets dir)
    get_property(subdirectories DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
    foreach(subdir ${subdirectories})
        get_all_targets_recursive(${targets} ${subdir})
    endforeach()

    get_property(current_targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
    list(APPEND ${targets} ${current_targets})
endmacro()

get_all_targets(${soci_SOURCE_DIR} all_soci_targets)
foreach(_soci_target IN LISTS all_soci_targets)
    target_compile_options(${_soci_target} INTERFACE -Wno-shadow -Wno-zero-as-null-pointer-constant -Wno-pedantic -Wno-undef)
endforeach()


################### gRPC ####################
set(MODYN_USES_LOCAL_GRPC false)
if(MODYN_TRY_LOCAL_GRPC)
  set(protobuf_MODULE_COMPATIBLE true)
  find_package(Protobuf CONFIG)
  find_package(gRPC CONFIG)

  if (gRPC_FOUND)
    message(STATUS "Found gRPC version ${gRPC_VERSION} locally (gRPC_FOUND = ${gRPC_FOUND})!")
    if (NOT TARGET gRPC::grpc_cpp_plugin)
      message(STATUS "gRPC::grpc_cpp_plugin is not a target, despite finding CMake. Building from source.")
      set(MODYN_TRY_LOCAL_GRPC OFF)
    else()
      if (Protobuf_FOUND)
        message(STATUS "Found protobuf!")
        include_directories(${PROTOBUF_INCLUDE_DIRS})
        set(MODYN_USES_LOCAL_GRPC true)
        if (NOT TARGET grpc_cpp_plugin)
          message(STATUS "Since grpc_cpp_plugin was not defined as a target, we define it manually.")
          add_executable(grpc_cpp_plugin ALIAS gRPC::grpc_cpp_plugin)
        endif()
      else()
        message(FATAL "Did not find Protobuf, please run cmake in a clean build directory with -DMODYN_TRY_LOCAL_GRPC=Off or install protobuf on your system.")
      endif()
    endif()
  else()
    message(STATUS "Did not find gRPC locally, building from source.")
  endif()
endif()

if((NOT MODYN_TRY_LOCAL_GRPC) OR (NOT gRPC_FOUND))
  message(STATUS "Making gRPC available (this may take a while).")
  set(gRPC_PROTOBUF_PROVIDER "module" CACHE BOOL "" FORCE)
  set(ABSL_ENABLE_INSTALL ON)  # https://github.com/protocolbuffers/protobuf/issues/12185
  FetchContent_Declare(
    gRPC
    GIT_REPOSITORY https://github.com/grpc/grpc
    GIT_TAG        v1.64.3 # When updating this, make sure to also update the modynbase dockerfile
    GIT_SHALLOW TRUE
  )
  set(gRPC_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(gRPC_BUILD_CSHARP_EXT OFF CACHE BOOL "" FORCE)
  set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)

  set(FETCHCONTENT_QUIET OFF)
  FetchContent_MakeAvailable(gRPC)
  set(FETCHCONTENT_QUIET ON)
endif()

file(DOWNLOAD
https://raw.githubusercontent.com/protocolbuffers/protobuf/v29.2/cmake/protobuf-generate.cmake
${CMAKE_CURRENT_BINARY_DIR}/protobuf-generate.cmake)
include(${CMAKE_CURRENT_BINARY_DIR}/protobuf-generate.cmake)

if(NOT COMMAND protobuf_generate)
  message(FATAL_ERROR "protobuf_generate not available. Potentially there is an error with your local CMake installation. If set, try using -DMODYN_TRY_LOCAL_GRPC=Off.")
else()
  message(STATUS "Found protobuf_generate")
endif()

message(STATUS "Processed gRPC.")
