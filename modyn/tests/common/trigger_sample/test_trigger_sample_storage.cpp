
#include <gtest/gtest.h>

#include <filesystem>

#include "trigger_sample_storage.hpp"

using namespace modyn::common::trigger_sample_storage;

class TriggerSampleTest : public ::testing::Test {
 protected:
  std::string tmp_dir_;

  TriggerSampleTest() : tmp_dir_{std::filesystem::temp_directory_path().string() + "/trigger_sample_test"} {}

  void SetUp() override {
    // Create temporary directory
    std::filesystem::create_directory(tmp_dir_);
  }

  void TearDown() override {
    // Remove temporary directory
    std::filesystem::remove_all(tmp_dir_);
  }
};

TEST_F(TriggerSampleTest, TestWriteReadSingle) {
  // Hardcoded data to write
  const char header[] =
      "\x93NUMPY\x01\x00v\x00{'descr': [('f0', '<i8'), ('f1', '<f8')], 'fortran_order': False, 'shape': (2,), }        "
      "                           \n";
  std::string file = tmp_dir_ + "/test.npy";
  const char data_write[] = "0123456789012345678901234567890";

  modyn::common::trigger_sample_storage::write_file(file.c_str(), &data_write, 0u, 2u, header, 128u);

  // Check whether header wrote correctly
  ASSERT_EQ(modyn::common::trigger_sample_storage::get_num_samples_in_file_impl(file.c_str()), 2u);

  int64_t size[] = {0};
  char* data_read = static_cast<char*>(modyn::common::trigger_sample_storage::parse_file_impl(file.c_str(), size));

  ASSERT_EQ(*size, 2u);

  // Check whether read data equals written data
  ASSERT_STREQ(data_read, data_write);

  modyn::common::trigger_sample_storage::release_data_impl(data_read);
}

TEST_F(TriggerSampleTest, TestWriteReadAllMulti) {
  // Hardcoded data to write
  const char header_a[] =
      "\x93NUMPY\x01\x00v\x00{'descr': [('f0', '<i8'), ('f1', '<f8')], 'fortran_order': False, 'shape': (2,), }        "
      "                           \n";
  std::string file_a = tmp_dir_ + "/test_a.npy";
  const char data_write_a[] = "0123456789012345678901234567890";

  const char header_b[] =
      "\x93NUMPY\x01\x00v\x00{'descr': [('f0', '<i8'), ('f1', '<f8')], 'fortran_order': False, 'shape': (3,), }        "
      "                           \n";
  std::string file_b = tmp_dir_ + "/test_b.npy";
  const char data_write_b[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTU";

  modyn::common::trigger_sample_storage::write_file(file_a.c_str(), &data_write_a, 0u, 2u, header_a, 128u);
  modyn::common::trigger_sample_storage::write_file(file_b.c_str(), &data_write_b, 0u, 3u, header_b, 128u);

  // Check whether header wrote correctly
  ASSERT_EQ(modyn::common::trigger_sample_storage::get_num_samples_in_file_impl(file_a.c_str()), 2u);
  ASSERT_EQ(modyn::common::trigger_sample_storage::get_num_samples_in_file_impl(file_b.c_str()), 3u);

  const char pattern[] = "test";
  int64_t size[] = {0};
  char* data_read =
      static_cast<char*>(modyn::common::trigger_sample_storage::get_all_samples_impl(tmp_dir_.c_str(), size, pattern));

  // Check if both files were added up
  ASSERT_EQ(*size, 5u);

  modyn::common::trigger_sample_storage::release_data_impl(data_read);
}

TEST_F(TriggerSampleTest, TestWriteReadWorkerMulti) {
  // Hardcoded data to write
  const char header_a[] =
      "\x93NUMPY\x01\x00v\x00{'descr': [('f0', '<i8'), ('f1', '<f8')], 'fortran_order': False, 'shape': (2,), }        "
      "                           \n";
  std::string file_a = tmp_dir_ + "/test_a.npy";
  const char data_write_a[] = "0123456789012345678901234567890";

  const char header_b[] =
      "\x93NUMPY\x01\x00v\x00{'descr': [('f0', '<i8'), ('f1', '<f8')], 'fortran_order': False, 'shape': (3,), }        "
      "                           \n";
  std::string file_b = tmp_dir_ + "/test_b.npy";
  const char data_write_b[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTU";

  modyn::common::trigger_sample_storage::write_file(file_a.c_str(), &data_write_a, 0u, 2u, header_a, 128u);
  modyn::common::trigger_sample_storage::write_file(file_b.c_str(), &data_write_b, 0u, 3u, header_b, 128u);

  // Check whether header wrote correctly
  ASSERT_EQ(modyn::common::trigger_sample_storage::get_num_samples_in_file_impl(file_a.c_str()), 2u);
  ASSERT_EQ(modyn::common::trigger_sample_storage::get_num_samples_in_file_impl(file_b.c_str()), 3u);

  const char pattern[] = "test";
  int64_t size[] = {0};
  char* data_read = static_cast<char*>(
      modyn::common::trigger_sample_storage::get_worker_samples_impl(tmp_dir_.c_str(), size, pattern, 1, 3));

  // Check if both files were added up
  ASSERT_EQ(*size, 3u);

  modyn::common::trigger_sample_storage::release_data_impl(data_read);
}
