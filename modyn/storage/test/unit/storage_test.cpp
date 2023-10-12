#include "storage.hpp"

#include <gtest/gtest.h>

#include "test_utils.hpp"

using namespace storage::test;
using namespace storage;

class StorageTest : public ::testing::Test {
 protected:
  void SetUp() override { TestUtils::create_dummy_yaml(); }

  void TearDown() override { TestUtils::delete_dummy_yaml(); }
};

TEST_F(StorageTest, TestStorage) {
  const std::string config_file = "config.yaml";
  Storage storage(config_file);
  storage.run();
}
