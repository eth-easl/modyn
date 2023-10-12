add_compile_options(${MODYNSTORAGE_COMPILE_OPTIONS})

# modyn has a custom FAIL macro. Use GTEST_FAIL to refer to the google macro
add_definitions(-DGTEST_DONT_DEFINE_FAIL)

##################################################
# TEST UTILITIES
##################################################
set(
  MODYNSTORAGE_TEST_UTILS_SOURCES

  test_utils.cpp
  test_utils.hpp
)

add_library(modynstorage-test-utils-objs OBJECT ${MODYNSTORAGE_TEST_UTILS_SOURCES})
target_include_directories(modynstorage-test-utils-objs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(modynstorage-test-utils-objs PUBLIC gtest gmock spdlog fmt modynstorage)

#################################################t
# UNIT TESTS
##################################################
set(
  MODYNSTORAGE_TEST_SOURCES

  unit/storage_test.cpp
  unit/internal/file_watcher/file_watcher_test.cpp
  unit/internal/file_watcher/file_watcher_watchdog_test.cpp
  unit/internal/database/storage_database_connection_test.cpp
  unit/internal/file_wrapper/single_sample_file_wrapper_test.cpp
  unit/internal/file_wrapper/mock_file_wrapper.hpp
  unit/internal/file_wrapper/binary_file_wrapper_test.cpp
  unit/internal/file_wrapper/csv_file_wrapper_test.cpp
  unit/internal/file_wrapper/file_wrapper_utils_test.cpp
  unit/internal/filesystem_wrapper/local_filesystem_wrapper_test.cpp
  unit/internal/filesystem_wrapper/mock_filesystem_wrapper.hpp
  unit/internal/filesystem_wrapper/filesystem_wrapper_utils_test.cpp
  unit/internal/grpc/storage_service_impl_test.cpp
)

add_library(modynstorage-test-objs OBJECT ${MODYNSTORAGE_TEST_SOURCES})
target_link_libraries(modynstorage-test-objs PRIVATE modynstorage-test-utils-objs)

add_executable(modynstorage-test storage_test.cpp)
target_link_libraries(modynstorage-test PRIVATE modynstorage-test-objs modynstorage-test-utils-objs)
add_test(modynstorage-test modynstorage-test)

##################################################################
# TARGET CONTAINING ALL TEST FILES (FOR CLANG-TIDY UNITY BUILD)
##################################################################
add_executable(modynstorage-all-test-sources-for-tidy EXCLUDE_FROM_ALL
  storage_test.cpp ${MODYNSTORAGE_TEST_UTILS_SOURCES} ${MODYNSTORAGE_TEST_SOURCES})

# just for the include directories
target_link_libraries(modynstorage-all-test-sources-for-tidy PRIVATE
modynstorage-test-objs modynstorage-test-utils-objs modynstorage)
