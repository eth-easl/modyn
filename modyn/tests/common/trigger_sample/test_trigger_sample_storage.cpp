
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
  const std::string header(
      "\x93NUMPY\x01\x00v\x00{'descr': [('f0', '<i8'), ('f1', '<f8')], 'fortran_order': False, 'shape': (2,), }        "
      "                           \n",
      128);
  const std::string file = tmp_dir_ + "/test.npy";
  const std::string data_write("0123456789012345678901234567890", 32);

  modyn::common::trigger_sample_storage::write_file(file.c_str(), data_write.c_str(), 0u, 2u, header.c_str(), 128u);

  // Check whether header wrote correctly
  ASSERT_EQ(modyn::common::trigger_sample_storage::get_num_samples_in_file_impl(file.c_str()), 2u);

  std::array<int64_t, 1> size = {0};
  char* data_read =
      static_cast<char*>(modyn::common::trigger_sample_storage::parse_file_impl(file.c_str(), size.data()));

  ASSERT_EQ(size[0], 2u);

  // Check whether read data equals written data
  ASSERT_STREQ(data_read, data_write.data());

  modyn::common::trigger_sample_storage::release_data_impl(data_read);
}

TEST_F(TriggerSampleTest, TestWriteReadAllMulti) {
  // Hardcoded data to write
  const std::string header_a(
      "\x93NUMPY\x01\x00v\x00{'descr': [('f0', '<i8'), ('f1', '<f8')], 'fortran_order': False, 'shape': (2,), }        "
      "                           \n",
      128);
  const std::string file_a = tmp_dir_ + "/test_a.npy";
  const std::string data_write_a("0123456789012345678901234567890", 32);

  const std::string header_b(
      "\x93NUMPY\x01\x00v\x00{'descr': [('f0', '<i8'), ('f1', '<f8')], 'fortran_order': False, 'shape': (3,), }        "
      "                           \n",
      128);
  const std::string file_b = tmp_dir_ + "/test_b.npy";
  const std::string data_write_b("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTU", 48);

  modyn::common::trigger_sample_storage::write_file(file_a.c_str(), data_write_a.c_str(), 0u, 2u, header_a.c_str(),
                                                    128u);
  modyn::common::trigger_sample_storage::write_file(file_b.c_str(), data_write_b.c_str(), 0u, 3u, header_b.c_str(),
                                                    128u);

  // Check whether header wrote correctly
  ASSERT_EQ(modyn::common::trigger_sample_storage::get_num_samples_in_file_impl(file_a.c_str()), 2u);
  ASSERT_EQ(modyn::common::trigger_sample_storage::get_num_samples_in_file_impl(file_b.c_str()), 3u);

  const std::string pattern = "test";
  std::array<int64_t, 1> size = {0};
  char* data_read = static_cast<char*>(
      modyn::common::trigger_sample_storage::get_all_samples_impl(tmp_dir_.c_str(), size.data(), pattern.c_str()));

  // Check if both files were added up
  ASSERT_EQ(size[0], 5u);

  modyn::common::trigger_sample_storage::release_data_impl(data_read);
}

TEST_F(TriggerSampleTest, TestWriteReadWorkerMulti) {
  // Hardcoded data to write
  const std::string header_a(
      "\x93NUMPY\x01\x00v\x00{'descr': [('f0', '<i8'), ('f1', '<f8')], 'fortran_order': False, 'shape': (2,), }        "
      "                           \n",
      128);
  const std::string file_a = tmp_dir_ + "/test_a.npy";
  const std::string data_write_a("0123456789012345678901234567890", 32);

  const std::string header_b(
      "\x93NUMPY\x01\x00v\x00{'descr': [('f0', '<i8'), ('f1', '<f8')], 'fortran_order': False, 'shape': (3,), }        "
      "                           \n",
      128);
  const std::string file_b = tmp_dir_ + "/test_b.npy";
  const std::string data_write_b("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTU", 48);

  modyn::common::trigger_sample_storage::write_file(file_a.c_str(), data_write_a.c_str(), 0u, 2u, header_a.c_str(),
                                                    128u);
  modyn::common::trigger_sample_storage::write_file(file_b.c_str(), data_write_b.c_str(), 0u, 3u, header_b.c_str(),
                                                    128u);

  // Check whether header wrote correctly
  ASSERT_EQ(modyn::common::trigger_sample_storage::get_num_samples_in_file_impl(file_a.c_str()), 2u);
  ASSERT_EQ(modyn::common::trigger_sample_storage::get_num_samples_in_file_impl(file_b.c_str()), 3u);

  const std::string pattern = "test";
  std::array<int64_t, 1> size = {0};
  char* data_read = static_cast<char*>(modyn::common::trigger_sample_storage::get_worker_samples_impl(
      tmp_dir_.c_str(), size.data(), pattern.c_str(), 1, 3));

  // Check if both files were added up
  ASSERT_EQ(size[0], 3u);

  modyn::common::trigger_sample_storage::release_data_impl(data_read);
}
