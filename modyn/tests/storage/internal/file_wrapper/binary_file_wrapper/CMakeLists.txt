# 'Google_test' is the subproject name
project(Google_tests)

include(FetchContent)
FetchContent_Declare(
  googletest
  # Specify the commit you depend on and update it regularly.
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Now simply link against gtest or gtest_main as needed. Eg
add_executable(test_binary_file_wrapper test_binary_file_wrapper.cpp)
target_link_libraries(test_binary_file_wrapper gtest_main)
add_test(NAME test_binary_file_wrapper COMMAND test_binary_file_wrapper)