
#include <gtest/gtest.h>

#include <array>
#include <filesystem>

#include "local_storage_backend.hpp"
#include "test_utils.hpp"

using namespace modyn::common::local_storage_backend;

class LocalStorageBackendTest : public ::testing::Test {
 protected:
  std::string tmp_dir_;

  LocalStorageBackendTest() : tmp_dir_{modyn::test::TestUtils::get_tmp_testdir("local_storage_backend_test")} {}

  void SetUp() override {
    // Create temporary directory
    std::filesystem::create_directory(tmp_dir_);
  }

  void TearDown() override {
    // Remove temporary directory
    std::filesystem::remove_all(tmp_dir_);
  }
};

TEST_F(LocalStorageBackendTest, TestWriteReadSingle) {
  // Hardcoded data to write
  const std::string file = tmp_dir_ + "/test.npy";
  const std::string data_write("0123456789012345678901234567890");
  std::string data_read(32, '\0');

  modyn::common::local_storage_backend::write_file(file.c_str(), data_write.c_str(), 0u, 2u);
  modyn::common::local_storage_backend::parse_file(file.c_str(), data_read.c_str(), 0u, 2u, 0u);

  // Check whether read data equals written data
  ASSERT_STREQ(data_read.data(), data_write.data());
}

TEST_F(LocalStorageBackendTest, TestWriteReadAllMulti) {
  // Hardcoded data to write
  const char* files[] = {"test_a.npy", "test_b.npy"};
  const std::string data_write("0123456789012345678901234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTU");
  std::string data_read(80, '\0');
  const int64_t data_lengths[] = {3u, 2u};
  const int64_t data_offsets[] = {0u, 0u};

  modyn::common::local_storage_backend::write_files_impl(files, data_write.c_str(), data_lengths, 2u);
  modyn::common::local_storage_backend::parse_files_impl(files, data_read.c_str(), data_lengths, data_offsets, 2u);

  // Check whether read data equals written data
  ASSERT_STREQ(data_read.data(), data_write.data());
}
