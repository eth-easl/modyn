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

add_subdirectory("common")

# TODO(MaxiBoether): When merging storage, only add subdirectory for storage when MODYN_BUILD_STORAGE is enabled