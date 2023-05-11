set(MODYNSTORAGE_SOURCES

    Storage.cpp
    internal/database/StorageDatabaseConnection.cpp
    internal/file_watcher/FileWatchdog.cpp
    internal/file_watcher/FileWatcher.cpp
    internal/file_wrapper/BinaryFileWrapper.cpp
    internal/file_wrapper/SingleSampleFileWrapper.cpp
    internal/filesystem_wrapper/LocalFileSystemWrapper.cpp
)

# Explicitly set all header files so that IDEs will recognize them as part of the project
# TODO Add include directory
set(MODYNSTORAGE_HEADERS

    ../include/Storage.hpp
    ../include/internal/database/StorageDatabaseConnection.hpp
    ../include/internal/file_watcher/FileWatchdog.hpp
    ../include/internal/file_watcher/FileWatcher.hpp
    ../include/internal/file_wrapper/AbstractFileWrapper.hpp
    ../include/internal/file_wrapper/BinaryFileWrapper.hpp
    ../include/internal/file_wrapper/SingleSampleFileWrapper.hpp
    ../include/internal/filesystem_wrapper/AbstractFileSystemWrapper.hpp
    ../include/internal/filesystem_wrapper/LocalFileSystemWrapper.hpp
    ../include/internal/utils/utils.hpp
    )

target_sources(modynstorage PRIVATE ${MODYNSTORAGE_HEADERS} ${MODYNSTORAGE_SOURCES})
target_include_directories(modynstorage PUBLIC ../include)
target_compile_options(modynstorage PRIVATE ${MODYNSTORAGE_COMPILE_OPTIONS})

target_link_libraries(modynstorage PRIVATE spdlog argparse yaml-cpp soci_core soci_postgresql soci_sqlite3)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
target_compile_definitions(modynstorage PRIVATE MODYNSTORAGE_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\")

# AppleClang needs the SDKROOT setup in order to be used directly. This is not the case for clang++ or g++.
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang")
  message(STATUS "Detected AppleClang compiler. Setting up AppleClang SDKROOT.")
  message(STATUS "  SDKROOT: ${CMAKE_OSX_SYSROOT}")
  set(MODYNSTORAGE_COMPILER_ENV "SDKROOT=${CMAKE_OSX_SYSROOT}")
endif ()

target_compile_definitions(modynstorage PRIVATE "MODYNSTORAGE_CMAKE_COMPILER=\"${MODYNSTORAGE_COMPILER_ENV} ${CMAKE_CXX_COMPILER}\"")

target_compile_definitions(modynstorage PUBLIC ${MODYNSTORAGE_COMPILE_DEFINITIONS})
message(STATUS "System-specific compile definitions: ${MODYNSTORAGE_COMPILE_DEFINITIONS}")

# This adds a `INCLUDE_DIRECTORIES` definition containing all include directories, separate by comma.
# The definition is set to PRIVATE, so it will not be exposed if the target is itself a dependency.
set(INCLUDE_EXPR "$<TARGET_PROPERTY:modynstorage,INCLUDE_DIRECTORIES>")
set(INCLUDE_FILTER "$<FILTER:${INCLUDE_EXPR},INCLUDE,.*/.*>")
set(INCLUDE_JOINED "$<JOIN:${INCLUDE_FILTER},,>")
target_compile_definitions(modynstorage PRIVATE "INCLUDE_DIRECTORIES=\"${INCLUDE_JOINED}\"")