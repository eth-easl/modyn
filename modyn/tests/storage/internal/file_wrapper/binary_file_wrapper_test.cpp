#include "internal/file_wrapper/binary_file_wrapper.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <fstream>

#include "internal/filesystem_wrapper/mock_filesystem_wrapper.hpp"
#include "storage_test_utils.hpp"
#include "test_utils.hpp"

using namespace modyn::storage;

class BinaryFileWrapperTest : public ::testing::Test {
 protected:
  std::string file_name_;
  YAML::Node config_;
  std::shared_ptr<MockFilesystemWrapper> filesystem_wrapper_;
  std::string tmp_dir_ = std::filesystem::temp_directory_path().string() + "/binary_file_wrapper_test";

  BinaryFileWrapperTest()
      : config_{StorageTestUtils::get_dummy_file_wrapper_config()},
        filesystem_wrapper_{std::make_shared<MockFilesystemWrapper>()} {
    file_name_ = tmp_dir_ + "/test.bin";
  }

  void SetUp() override {
    std::filesystem::create_directory(tmp_dir_);

    std::ofstream file(file_name_, std::ios::binary);
    const std::vector<std::pair<uint32_t, uint16_t>> data = {{42, 12}, {43, 13}, {44, 14}, {45, 15}};
    for (const auto& [payload, label] : data) {
      payload_to_file(file, payload, label);
    }
    file.close();
  }

  static void payload_to_file(std::ofstream& file, uint16_t payload, uint16_t label) {
    file.write(reinterpret_cast<const char*>(&payload), sizeof(uint16_t));
    file.write(reinterpret_cast<const char*>(&label), sizeof(uint16_t));
  }

  void TearDown() override { std::filesystem::remove_all(file_name_); }
};

TEST_F(BinaryFileWrapperTest, TestGetNumberOfSamples) {
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::Return(stream_ptr));
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillOnce(testing::Return(16));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);
  ASSERT_EQ(file_wrapper.get_number_of_samples(), 4);

  stream_ptr->close();
}

TEST_F(BinaryFileWrapperTest, TestValidateFileExtension) {
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::Return(stream_ptr));
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillOnce(testing::Return(16));
  ASSERT_NO_THROW(const BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_););
}

TEST_F(BinaryFileWrapperTest, TestValidateRequestIndices) {
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillOnce(testing::Return(16));
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillRepeatedly(testing::Return(stream_ptr));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);
  std::vector<unsigned char> sample = file_wrapper.get_sample(0);

  ASSERT_EQ(sample.size(), 2);
  ASSERT_EQ((sample)[0], 12);

  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillOnce(testing::Return(16));
  BinaryFileWrapper file_wrapper2(file_name_, config_, filesystem_wrapper_);
  ASSERT_THROW(file_wrapper2.get_sample(8), modyn::utils::ModynException);
}

TEST_F(BinaryFileWrapperTest, TestGetLabel) {
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillOnce(testing::Return(16));
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::Return(stream_ptr));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);
  ASSERT_EQ(file_wrapper.get_label(0), 42);
  ASSERT_EQ(file_wrapper.get_label(1), 43);
  ASSERT_EQ(file_wrapper.get_label(2), 44);
  ASSERT_EQ(file_wrapper.get_label(3), 45);
}

TEST_F(BinaryFileWrapperTest, TestGetAllLabels) {
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillOnce(testing::Return(16));
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::Return(stream_ptr));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);
  std::vector<int64_t> labels = file_wrapper.get_all_labels();
  ASSERT_EQ(labels.size(), 4);
  ASSERT_EQ((labels)[0], 42);
  ASSERT_EQ((labels)[1], 43);
  ASSERT_EQ((labels)[2], 44);
  ASSERT_EQ((labels)[3], 45);
}

TEST_F(BinaryFileWrapperTest, TestGetSample) {
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillRepeatedly(testing::Return(16));
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillRepeatedly(testing::Return(stream_ptr));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);
  std::vector<unsigned char> sample = file_wrapper.get_sample(0);
  ASSERT_EQ(sample.size(), 2);
  ASSERT_EQ((sample)[0], 12);

  sample = file_wrapper.get_sample(1);
  ASSERT_EQ(sample.size(), 2);
  ASSERT_EQ((sample)[0], 13);

  sample = file_wrapper.get_sample(2);
  ASSERT_EQ(sample.size(), 2);
  ASSERT_EQ((sample)[0], 14);

  sample = file_wrapper.get_sample(3);
  ASSERT_EQ(sample.size(), 2);
  ASSERT_EQ((sample)[0], 15);
}

TEST_F(BinaryFileWrapperTest, TestGetSamples) {
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillRepeatedly(testing::Return(16));
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillRepeatedly(testing::Return(stream_ptr));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);
  std::vector<std::vector<unsigned char>> samples = file_wrapper.get_samples(0, 3);
  ASSERT_EQ(samples.size(), 4);
  ASSERT_EQ((samples)[0][0], 12);
  ASSERT_EQ((samples)[1][0], 13);
  ASSERT_EQ((samples)[2][0], 14);
  ASSERT_EQ((samples)[3][0], 15);

  samples = file_wrapper.get_samples(1, 3);
  ASSERT_EQ(samples.size(), 3);
  ASSERT_EQ((samples)[0][0], 13);
  ASSERT_EQ((samples)[1][0], 14);
  ASSERT_EQ((samples)[2][0], 15);

  samples = file_wrapper.get_samples(2, 3);
  ASSERT_EQ(samples.size(), 2);
  ASSERT_EQ((samples)[0][0], 14);
  ASSERT_EQ((samples)[1][0], 15);

  samples = file_wrapper.get_samples(3, 3);
  ASSERT_EQ(samples.size(), 1);
  ASSERT_EQ((samples)[0][0], 15);

  ASSERT_THROW(file_wrapper.get_samples(4, 3), modyn::utils::ModynException);

  samples = file_wrapper.get_samples(1, 2);
  ASSERT_EQ(samples.size(), 2);
  ASSERT_EQ((samples)[0][0], 13);
  ASSERT_EQ((samples)[1][0], 14);
}

TEST_F(BinaryFileWrapperTest, TestGetSamplesFromIndices) {
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillRepeatedly(testing::Return(16));
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillRepeatedly(testing::Return(stream_ptr));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);
  std::vector<uint64_t> label_indices{0, 1, 2, 3};
  std::vector<std::vector<unsigned char>> samples = file_wrapper.get_samples_from_indices(label_indices);
  ASSERT_EQ(samples.size(), 4);
  ASSERT_EQ((samples)[0][0], 12);
  ASSERT_EQ((samples)[1][0], 13);
  ASSERT_EQ((samples)[2][0], 14);
  ASSERT_EQ((samples)[3][0], 15);

  label_indices = {1, 2, 3};
  samples = file_wrapper.get_samples_from_indices(label_indices);
  ASSERT_EQ(samples.size(), 3);
  ASSERT_EQ((samples)[0][0], 13);
  ASSERT_EQ((samples)[1][0], 14);
  ASSERT_EQ((samples)[2][0], 15);

  label_indices = {2};
  samples = file_wrapper.get_samples_from_indices(label_indices);
  ASSERT_EQ(samples.size(), 1);
  ASSERT_EQ((samples)[0][0], 14);

  label_indices = {1, 3};
  samples = file_wrapper.get_samples_from_indices(label_indices);
  ASSERT_EQ(samples.size(), 2);
  ASSERT_EQ((samples)[0][0], 13);
  ASSERT_EQ((samples)[1][0], 15);

  label_indices = {3, 1, 3};
  samples = file_wrapper.get_samples_from_indices(label_indices);
  ASSERT_EQ(samples.size(), 3);
  ASSERT_EQ((samples)[0][0], 15);
  ASSERT_EQ((samples)[1][0], 13);
  ASSERT_EQ((samples)[2][0], 15);
}

TEST_F(BinaryFileWrapperTest, TestDeleteSamples) {
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::Return(stream_ptr));
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillOnce(testing::Return(16));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);

  const std::vector<uint64_t> label_indices{0, 1, 2, 3};

  ASSERT_NO_THROW(file_wrapper.delete_samples(label_indices));
}