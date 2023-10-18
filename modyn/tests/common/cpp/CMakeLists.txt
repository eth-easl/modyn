#################################################t
# UNIT TESTS
##################################################
set(
  MODYN_COMMON_TEST_SOURCES

  utils/test_utils.cpp
)

add_library(modyn-common-test-objs OBJECT ${MODYN_COMMON_TEST_SOURCES})
target_link_libraries(modyn-common-test-objs PRIVATE modyn-test-utils-objs)

add_executable(modynstorage-test storage_test.cpp)
target_link_libraries(modynstorage-test PRIVATE modynstorage-test-objs modynstorage-test-utils-objs)
add_test(modynstorage-test modynstorage-test)

##################################################################
# TARGET CONTAINING ALL TEST FILES (FOR CLANG-TIDY UNITY BUILD)
##################################################################
add_executable(modynstorage-all-test-sources-for-tidy EXCLUDE_FROM_ALL
  storage_test.cpp ${MODYNSTORAGE_TEST_UTILS_SOURCES} ${MODYNSTORAGE_TEST_SOURCES})

# just for the include directories
target_link_libraries(modynstorage-all-test-sources-for-tidy PRIVATE modynstorage-test-objs modynstorage-test-utils-objs modynstorage)
