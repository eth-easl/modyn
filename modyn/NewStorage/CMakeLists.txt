cmake_minimum_required(VERSION 3.20)
project(modyn-storage)

set(CMAKE_CXX_STANDARD 20)
set(MODYNSTORAGE_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

##### BUILD TYPES #####
set(ALLOWED_BUILD_TYPES Asan Tsan Debug Release RelWithDebInfo MinSizeRel)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "${ALLOWED_BUILD_TYPES}")

if(CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE IN_LIST ALLOWED_BUILD_TYPES)
  message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}. Allowed types: ${ALLOWED_BUILD_TYPES}")
endif()

set(CMAKE_C_FLAGS_ASAN "-Og -g -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_ASAN "-Og -g -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined" CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_ASAN "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} -fsanitize=address -fsanitize=undefined" CACHE STRING "" FORCE)

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -D_GLIBCXX_ASSERTIONS" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g3 -DNDEBUG" CACHE STRING "" FORCE)

# Additional UBSan could be nice here, but causes false positives: https://github.com/google/sanitizers/issues/1106
set(CMAKE_C_FLAGS_TSAN "${CMAKE_C_FLAGS_RELWITHDEBINFO} -fsanitize=thread" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_TSAN "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fsanitize=thread" CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_TSAN "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} -fsanitize=thread" CACHE STRING "" FORCE)

##### PUBLIC OPTIONS #####
option(DARWIN_BUILD_PLAYGROUND "Set ON to build playground" ON)
option(MODYNSTORAGE_BUILD_TESTS "Set ON to build tests" ON)
option(MODYNSTORAGE_TEST_COVERAGE "Set ON to add test coverage" OFF)

#### INTERNAL OPTIONS ####
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(MODYNSTORAGE_COMPILE_DEFINITIONS "")
set(MODYNSTORAGE_COMPILE_OPTIONS "-Wall" "-Wextra" "-Werror" "-Wpedantic" "-Wextra-semi" "-Wnon-virtual-dtor" "-Wunused"
  "-Wzero-as-null-pointer-constant"
)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # Additional checks not supported by GCC -- some are supported on GCC, but not ignorable / not filterable
  list(APPEND MODYNSTORAGE_COMPILE_OPTIONS "-Wdocumentation" "-Wconditional-uninitialized" "-Wmissing-prototypes" "-Wundef"
    "-Wunused-exception-parameter" "-Wunused-member-function")
  list(APPEND MODYNSTORAGE_COMPILE_OPTIONS "-Wno-gnu-zero-variadic-macro-arguments") # legal in c++20

  # We have a lot of shadowed ctor args with simple structs that should usually be aggregate-initialized, but clang
  # doesn't support aggregate-initialization with emplace_back yet
  list(APPEND MODYNSTORAGE_COMPILE_OPTIONS "-Wshadow-all" "-Wno-shadow-field-in-constructor")
endif()

if (${MODYNSTORAGE_TEST_COVERAGE})
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Running with coverage flags")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
  else ()
    message(FATAL_ERROR "Test Coverage is enabled, but can't be performed with selected compiler. Please use Clang instead.")
  endif ()
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

### Make MODYNSTORAGE lib available as target in next steps ###
add_library(modynstorage)

### Obtain (Native) Compile Flags and Definitions ###
include(${MODYNSTORAGE_CMAKE_DIR}/system_info.cmake)

##### DEPENDENCIES #####
include(${MODYNSTORAGE_CMAKE_DIR}/dependencies.cmake)

##### MODYNSTORAGE LIB #####
add_subdirectory(src/)

### Main binary ###
add_executable(modyn-new-storage src/main.cpp)
target_link_libraries(modyn-new-storage PRIVATE modynstorage)

##### PLAYGROUND #####
if (${MODYNSTORAGE_BUILD_PLAYGROUND})
  message(STATUS "Playground is included in this build.")
  add_executable(playground playground.cpp)
  target_compile_options(playground PRIVATE ${MODYNSTORAGE_COMPILE_OPTIONS})
  target_link_libraries(playground PRIVATE modynstorage)
endif ()

##### TESTING #####
if (${MODYNSTORAGE_BUILD_TESTS})
  message(STATUS "Tests are included in this build.")
  enable_testing()
  add_subdirectory(test)
endif ()