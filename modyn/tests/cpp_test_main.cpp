#include <gtest/gtest.h>

#include <filesystem>
#include <iostream>

#include "test_utils.hpp"

// Global fixtures run before unit tests are executed
class GlobalTestEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    // create temp parent directory
    std::filesystem::create_directory(modyn::test::TestUtils::get_tmp_testdir());
  }

  void TearDown() override {}
};

int main(int argc, char** argv) {
  GlobalTestEnvironment env{};
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(&env);
  return RUN_ALL_TESTS();
}
