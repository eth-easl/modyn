# modyn has a custom FAIL macro. Use GTEST_FAIL to refer to the google macro
add_definitions(-DGTEST_DONT_DEFINE_FAIL)

##################################################
# TEST UTILITIES
##################################################
set(
  MODYN_TEST_UTILS_SOURCES

  utils/test_utils.cpp
  utils/test_utils.hpp
)

#################################################t
# UNIT TESTS
##################################################
set(
  MODYN_TEST_SOURCES

  common/cpp/utils/test_utils.cpp
  common/example_extension/test_example_extension.cpp
  common/trigger_sample/test_trigger_sample_storage.cpp
  selector/internal/storage_backend/local/test_local_storage_backend.cpp
)

#################################################t
# STORAGE TESTS
##################################################
if (${MODYN_BUILD_STORAGE})
  message(STATUS "Storage is included in this test build.")
  list(
    APPEND MODYN_TEST_UTILS_SOURCES
  
    storage/storage_test_utils.cpp
    storage/storage_test_utils.hpp
  )

  list(
    APPEND MODYN_TEST_SOURCES
  
    storage/internal/file_watcher/file_watcher_test.cpp
    storage/internal/file_watcher/file_watcher_watchdog_test.cpp
    storage/internal/database/storage_database_connection_test.cpp
    storage/internal/database/cursor_handler_test.cpp
    storage/internal/file_wrapper/single_sample_file_wrapper_test.cpp
    storage/internal/file_wrapper/mock_file_wrapper.hpp
    storage/internal/file_wrapper/binary_file_wrapper_test.cpp
    storage/internal/file_wrapper/csv_file_wrapper_test.cpp
    storage/internal/file_wrapper/file_wrapper_utils_test.cpp
    storage/internal/filesystem_wrapper/local_filesystem_wrapper_test.cpp
    storage/internal/filesystem_wrapper/mock_filesystem_wrapper.hpp
    storage/internal/filesystem_wrapper/filesystem_wrapper_utils_test.cpp
    storage/internal/grpc/storage_service_impl_test.cpp
  )

endif ()

add_library(modyn-test-utils-objs OBJECT ${MODYN_TEST_UTILS_SOURCES})
target_include_directories(modyn-test-utils-objs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/utils)
target_link_libraries(modyn-test-utils-objs PUBLIC gtest gmock spdlog fmt yaml-cpp modyn example_extension trigger_sample_storage local_storage_backend)

if (${MODYN_BUILD_STORAGE})
  target_link_libraries(modyn-test-utils-objs PUBLIC modyn-storage-library)
  target_include_directories(modyn-test-utils-objs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/storage)
endif ()

add_library(modyn-test-objs OBJECT ${MODYN_TEST_SOURCES})
target_link_libraries(modyn-test-objs PRIVATE modyn-test-utils-objs)

add_executable(modyn-test cpp_test_main.cpp)
target_link_libraries(modyn-test PRIVATE modyn-test-objs modyn-test-utils-objs)
add_test(modyn-test modyn-test)

##################################################################
# TARGET CONTAINING ALL TEST FILES (FOR CLANG-TIDY UNITY BUILD)
##################################################################
add_executable(modyn-all-test-sources-for-tidy EXCLUDE_FROM_ALL
  cpp_test_main.cpp ${MODYN_TEST_UTILS_SOURCES} ${MODYN_TEST_SOURCES})

# just for the include directories
target_link_libraries(modyn-all-test-sources-for-tidy PRIVATE modyn-test-objs modyn-test-utils-objs modyn)


