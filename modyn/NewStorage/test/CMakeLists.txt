add_compile_options(${MODYNSTORAGE_COMPILE_OPTIONS})

# modyn has a custom FAIL macro. Use GTEST_FAIL to refer to the google macro
add_definitions(-DGTEST_DONT_DEFINE_FAIL)

##################################################
# TEST UTILITIES
##################################################
set(
  MODYNSTORAGE_TEST_UTILS_SOURCES

  TestUtils.cpp
  TestUtils.hpp
)

add_library(modynstorage-test-utils-objs OBJECT ${MODYNSTORAGE_TEST_UTILS_SOURCES})
target_include_directories(modynstorage-test-utils-objs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(modynstorage-test-utils-objs PUBLIC gtest gmock spdlog modynstorage)

#################################################t
# UNIT TESTS
##################################################
set(
  MODYNSTORAGE_TEST_SOURCES

  Storage-test.cpp # TODO populate
)


add_library(modynstorage-test-objs OBJECT ${MODYNSTORAGE_TEST_SOURCES})
target_link_libraries(modynstorage-test-objs PRIVATE modynstorage-test-utils-objs)

add_executable(modynstorage-test main.cpp)
target_link_libraries(modynstorage-test PRIVATE modynstorage-test-objs modynstorage-test-utils-objs)
add_test(modynstorage-test modynstorage-test)