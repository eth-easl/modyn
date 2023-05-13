include(FetchContent)
list(APPEND CMAKE_PREFIX_PATH /opt/homebrew/opt/libpq) # for macOS builds

# Configure path to modules (for find_package)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/modules/")

################### spdlog ####################
message(STATUS "Making spdlog available.")
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.11.0
)
FetchContent_MakeAvailable(spdlog)

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
  GIT_TAG v1.13.0
)
FetchContent_MakeAvailable(googletest)

################### libpq++ ####################
find_package(PostgreSQL REQUIRED) # This needs to be installed on the system - cannot do a lightweight CMake install

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
set(SOCI_SHARED ON CACHE BOOL "soci configuration")
set(SOCI_EMPTY OFF CACHE BOOL "soci configuration")
set(SOCI_HAVE_BOOST OFF CACHE BOOL "configuration" FORCE)

FetchContent_GetProperties(soci)
if(NOT soci_POPULATED)
    FetchContent_Populate(soci)
    add_subdirectory(${soci_SOURCE_DIR})
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

macro(remove_flag_from_target _target _flag)
    get_target_property(_target_cxx_flags ${_target} COMPILE_OPTIONS)
    if(_target_cxx_flags)
        list(REMOVE_ITEM _target_cxx_flags ${_flag})
        set_target_properties(${_target} PROPERTIES COMPILE_OPTIONS "${_target_cxx_flags}")
    endif()
endmacro()

get_all_targets(${soci_SOURCE_DIR} all_soci_targets)
foreach(_soci_target IN LISTS all_soci_targets)
    target_compile_options(${_soci_target} INTERFACE -Wno-zero-as-null-pointer-constant -Wno-pedantic -Wno-undef)
endforeach()


################### yaml-cpp ####################
message(STATUS "Making yaml-cpp available.")

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG yaml-cpp-0.7.0
)
FetchContent_MakeAvailable(yaml-cpp)

target_compile_options(yaml-cpp INTERFACE -Wno-shadow -Wno-pedantic)

################### grpc ####################
#message(STATUS "Making grpc available.")

#FetchContent_Declare(
#  grpc
#  GIT_REPOSITORY https://github.com/grpc/grpc.git
#  GIT_TAG v1.54.1
#  GIT_SHALLOW    TRUE
#  GIT_PROGRESS   TRUE
#)
#FetchContent_MakeAvailable(grpc)