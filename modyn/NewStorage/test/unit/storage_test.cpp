#include "Storage.hpp"
#include "test_utils.hpp"
#include <gtest/gtest.h>

using namespace storage;

class StorageTest : public ::testing::Test {
protected:
  void SetUp() override { TestUtils::create_dummy_yaml(); }

  void TearDown() override { TestUtils::delete_dummy_yaml(); }
};

TEST_F(StorageTest, TestStorage) {
  std::string config_file = "config.yaml";
  storage::Storage storage(config_file);
  storage.run();
}
