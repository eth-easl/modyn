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

add_library(modyn-test-utils-objs OBJECT ${MODYN_TEST_UTILS_SOURCES})
target_include_directories(modyn-test-utils-objs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/utils)
target_link_libraries(modyn-test-utils-objs PUBLIC gtest gmock spdlog fmt modyn)

#################################################t
# UNIT TESTS
##################################################
set(
  MODYN_TEST_SOURCES

  common/cpp/utils/test_utils.cpp
)

# TODO(MaxiBoether): When merging into storage, only add tests for storage when MODYN_BUILD_STORAGE is enabled

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


