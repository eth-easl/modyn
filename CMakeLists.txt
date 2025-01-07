cmake_minimum_required(VERSION 3.22)
project(modyn)

message(STATUS "Hello, this is Modyn.")

set(CMAKE_CXX_STANDARD 20)
set(MODYN_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

##### BUILD TYPES #####
set(ALLOWED_BUILD_TYPES Asan Tsan Debug Release RelWithDebInfo MinSizeRel)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "${ALLOWED_BUILD_TYPES}")

if(CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE IN_LIST ALLOWED_BUILD_TYPES)
  message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}. Allowed types: ${ALLOWED_BUILD_TYPES}")
endif()

# We don't sanitize vptr because this requires RTTI which is a problem for soci
set(CMAKE_C_FLAGS_ASAN "-Og -g -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined -fno-sanitize=vptr" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_ASAN "-Og -g -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined -fno-sanitize=vptr" CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_ASAN "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} -fsanitize=address -fsanitize=undefined -fno-sanitize=vptr" CACHE STRING "" FORCE)

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -D_GLIBCXX_ASSERTIONS" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g3 -DNDEBUG" CACHE STRING "" FORCE)

# Additional UBSan could be nice here, but causes false positives: https://github.com/google/sanitizers/issues/1106
set(CMAKE_C_FLAGS_TSAN "${CMAKE_C_FLAGS_RELWITHDEBINFO} -fsanitize=thread" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_TSAN "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fsanitize=thread" CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_TSAN "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} -fsanitize=thread" CACHE STRING "" FORCE)

##### PUBLIC OPTIONS #####
option(MODYN_BUILD_PLAYGROUND "Set ON to build playground" ON)
option(MODYN_BUILD_TESTS "Set ON to build tests" ON)
option(MODYN_BUILD_STORAGE "Set ON to build storage components" OFF)
option(MODYN_TRY_LOCAL_GRPC "Set ON to try using local gRPC installation instead of building from source" ON)
option(MODYN_TEST_COVERAGE "Set ON to add test coverage" OFF)

#### INTERNAL OPTIONS ####
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(MODYN_COMPILE_DEFINITIONS "")
set(MODYN_COMPILE_OPTIONS "-Wall" "-Wextra" "-Werror" "-Wpedantic" "-Wextra-semi" "-Wnon-virtual-dtor" "-Wunused"
  "-Wzero-as-null-pointer-constant"
)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # Additional checks not supported by GCC -- some are supported on GCC, but not ignorable / not filterable
  list(APPEND MODYN_COMPILE_OPTIONS "-Wconditional-uninitialized" "-Wmissing-prototypes" "-Wundef"
    "-Wunused-exception-parameter" "-Wunused-member-function")
  list(APPEND MODYN_COMPILE_OPTIONS "-Wno-gnu-zero-variadic-macro-arguments") # legal in c++20

  # We have a lot of shadowed ctor args with simple structs that should usually be aggregate-initialized, but clang
  # doesn't support aggregate-initialization with emplace_back yet
  list(APPEND MODYN_COMPILE_OPTIONS "-Wshadow-all" "-Wno-shadow-field-in-constructor")

  # Without this, we get compilation errors for the chrono header. We need to set this globally
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -std=c++20")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")

elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  list(APPEND MODYN_COMPILE_OPTIONS "-Wno-missing-requires") # causes issues since gcc 12 and grpc 1.69
endif()



if (${MODYN_TEST_COVERAGE})
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Running with coverage flags")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
  else ()
    message(FATAL_ERROR "Test Coverage is enabled, but can't be performed with selected compiler. Please use Clang instead.")
  endif ()
endif ()


# AppleClang needs the SDKROOT setup in order to be used directly. This is not the case for clang++ or g++.
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang")
  message(STATUS "Detected AppleClang compiler. Setting up AppleClang SDKROOT.")
  message(STATUS "  SDKROOT: ${CMAKE_OSX_SYSROOT}")
  set(MODYN_COMPILER_ENV "SDKROOT=${CMAKE_OSX_SYSROOT}")
endif ()

#### CUSTOM COMPILER ####
# Unlike CXX/CC and so on, these variables are currently not checked by CMake, but needed to be checked manually
# If using a compiler built from scratch (e.g., on exotic systems), we need this to use LTO and other features.

if (DEFINED ENV{NM})
    set(CMAKE_NM $ENV{NM})
    message(STATUS "Found NM environment variable, set CMAKE_NM to ${CMAKE_NM}")
endif()

if (DEFINED ENV{AR})
    set(CMAKE_AR $ENV{AR})
    message(STATUS "Found AR environment variable, set CMAKE_AR to ${CMAKE_AR}")
endif()

if (DEFINED ENV{RANLIB})
    set(CMAKE_RANLIB $ENV{RANLIB})
    message(STATUS "Found RANLIB environment variable, set CMAKE_RANLIB to ${CMAKE_RANLIB}")
endif()

### Make MODYN lib available as target in next steps ###
add_library(modyn)

### Obtain (Native) Compile Flags and Definitions ###
include(${MODYN_CMAKE_DIR}/system_info.cmake)

##### DEPENDENCIES #####
include(${MODYN_CMAKE_DIR}/dependencies.cmake)

##### MODYN LIB #####
add_subdirectory(modyn/)
