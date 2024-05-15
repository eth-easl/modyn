
#include <gtest/gtest.h>

#include <array>
#include <filesystem>

#include "local_storage_backend.hpp"
#include "test_utils.hpp"

using namespace modyn::selector::local_storage_backend;

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
  std::vector<char> data_read(32, '\0');

  modyn::selector::local_storage_backend::write_file(file.c_str(), data_write.c_str(), 0u, 4u);
  modyn::selector::local_storage_backend::parse_file(file.c_str(), data_read.data(), 0u, 4u, 0u);

  // Check whether read data equals written data
  ASSERT_STREQ(data_read.data(), data_write.data());
}

TEST_F(LocalStorageBackendTest, TestWriteReadAllMulti) {
  // Hardcoded data to write
  std::array<const char*, 2> files = {"test_a.npy", "test_b.npy"};
  const std::string data_write("0123456789012345678901234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTU");
  std::vector<char> data_read(80, '\0');
  std::array<int64_t, 2> data_lengths = {6u, 4u};
  std::array<int64_t, 2> data_offsets = {0u, 0u};

  modyn::selector::local_storage_backend::write_files_impl(files.data(), data_write.c_str(), data_lengths.data(), 2u);
  modyn::selector::local_storage_backend::parse_files_impl(files.data(), data_read.data(), data_lengths.data(),
                                                           data_offsets.data(), 2u);

  // Check whether read data equals written data
  ASSERT_STREQ(data_read.data(), data_write.data());
}
