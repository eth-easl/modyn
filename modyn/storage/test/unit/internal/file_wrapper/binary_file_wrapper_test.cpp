#include "internal/file_wrapper/binary_file_wrapper.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <fstream>

#include "test_utils.hpp"
#include "unit/internal/filesystem_wrapper/mock_filesystem_wrapper.hpp"

using namespace storage::file_wrapper;
using namespace storage::test;

class BinaryFileWrapperTest : public ::testing::Test {
 protected:
  std::string file_name_;
  YAML::Node config_;
  std::shared_ptr<MockFilesystemWrapper> filesystem_wrapper_;

  BinaryFileWrapperTest()
      : file_name_{"tmp/test.bin"},
        config_{TestUtils::get_dummy_file_wrapper_config()},
        filesystem_wrapper_{std::make_shared<MockFilesystemWrapper>()} {}

  void SetUp() override {
    std::filesystem::create_directory("tmp");

    std::ofstream file(file_name_);
    file << "12345678";
    file.close();
  }

  void TearDown() override { std::filesystem::remove_all(file_name_); }
};

TEST_F(BinaryFileWrapperTest, TestGetNumberOfSamples) {
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillOnce(testing::Return(8));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);
  ASSERT_EQ(file_wrapper.get_number_of_samples(), 4);
}

TEST_F(BinaryFileWrapperTest, TestValidateFileExtension) {
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillOnce(testing::Return(8));
  ASSERT_NO_THROW(const BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_););
}

TEST_F(BinaryFileWrapperTest, TestValidateRequestIndices) {
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillOnce(testing::Return(8));
  std::unique_ptr<std::ifstream> stream = std::make_unique<std::ifstream>();
  stream->open(file_name_, std::ios::binary);
  std::ifstream& reference = *stream;
  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::ReturnRef(reference));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);
  std::vector<unsigned char> sample = file_wrapper.get_sample(0);

  ASSERT_EQ(sample.size(), 1);
  ASSERT_EQ((sample)[0], '2');

  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillOnce(testing::Return(8));
  BinaryFileWrapper file_wrapper2(file_name_, config_, filesystem_wrapper_);
  ASSERT_THROW(file_wrapper2.get_sample(8), storage::utils::ModynException);
}

TEST_F(BinaryFileWrapperTest, TestGetLabel) {
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillOnce(testing::Return(8));
  std::unique_ptr<std::ifstream> stream = std::make_unique<std::ifstream>();
  stream->open(file_name_, std::ios::binary);
  std::ifstream& reference = *stream;
  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillRepeatedly(testing::ReturnRef(reference));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);
  ASSERT_EQ(file_wrapper.get_label(0), 1);
  ASSERT_EQ(file_wrapper.get_label(1), 3);
  ASSERT_EQ(file_wrapper.get_label(2), 5);
  ASSERT_EQ(file_wrapper.get_label(3), 7);
}

TEST_F(BinaryFileWrapperTest, TestGetAllLabels) {
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillOnce(testing::Return(8));
  std::unique_ptr<std::ifstream> stream = std::make_unique<std::ifstream>();
  stream->open(file_name_, std::ios::binary);
  std::ifstream& reference = *stream;
  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::ReturnRef(reference));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);
  std::vector<int64_t> labels = file_wrapper.get_all_labels();
  ASSERT_EQ(labels.size(), 4);
  ASSERT_EQ((labels)[0], 1);
  ASSERT_EQ((labels)[1], 3);
  ASSERT_EQ((labels)[2], 5);
  ASSERT_EQ((labels)[3], 7);
}

TEST_F(BinaryFileWrapperTest, TestGetSample) {
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillRepeatedly(testing::Return(8));
  std::unique_ptr<std::ifstream> stream = std::make_unique<std::ifstream>();
  stream->open(file_name_, std::ios::binary);
  std::ifstream& reference = *stream;
  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillRepeatedly(testing::ReturnRef(reference));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);
  std::vector<unsigned char> sample = file_wrapper.get_sample(0);
  ASSERT_EQ(sample.size(), 1);
  ASSERT_EQ((sample)[0], '2');

  sample = file_wrapper.get_sample(1);
  ASSERT_EQ(sample.size(), 1);
  ASSERT_EQ((sample)[0], '4');

  sample = file_wrapper.get_sample(2);
  ASSERT_EQ(sample.size(), 1);
  ASSERT_EQ((sample)[0], '6');

  sample = file_wrapper.get_sample(3);
  ASSERT_EQ(sample.size(), 1);
  ASSERT_EQ((sample)[0], '8');
}

TEST_F(BinaryFileWrapperTest, TestGetSamples) {
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillRepeatedly(testing::Return(8));
  std::unique_ptr<std::ifstream> stream = std::make_unique<std::ifstream>();
  stream->open(file_name_, std::ios::binary);
  std::ifstream& reference = *stream;
  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillRepeatedly(testing::ReturnRef(reference));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);
  std::vector<std::vector<unsigned char>> samples = file_wrapper.get_samples(0, 3);
  ASSERT_EQ(samples.size(), 4);
  ASSERT_EQ((samples)[0][0], '2');
  ASSERT_EQ((samples)[1][0], '4');
  ASSERT_EQ((samples)[2][0], '6');
  ASSERT_EQ((samples)[3][0], '8');

  samples = file_wrapper.get_samples(1, 3);
  ASSERT_EQ(samples.size(), 3);
  ASSERT_EQ((samples)[0][0], '4');
  ASSERT_EQ((samples)[1][0], '6');
  ASSERT_EQ((samples)[2][0], '8');

  samples = file_wrapper.get_samples(2, 3);
  ASSERT_EQ(samples.size(), 2);
  ASSERT_EQ((samples)[0][0], '6');
  ASSERT_EQ((samples)[1][0], '8');

  samples = file_wrapper.get_samples(3, 3);
  ASSERT_EQ(samples.size(), 1);
  ASSERT_EQ((samples)[0][0], '8');

  ASSERT_THROW(file_wrapper.get_samples(4, 3), storage::utils::ModynException);

  samples = file_wrapper.get_samples(1, 2);
  ASSERT_EQ(samples.size(), 2);
  ASSERT_EQ((samples)[0][0], '4');
  ASSERT_EQ((samples)[1][0], '6');
}

TEST_F(BinaryFileWrapperTest, TestGetSamplesFromIndices) {
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillRepeatedly(testing::Return(8));
  std::unique_ptr<std::ifstream> stream = std::make_unique<std::ifstream>();
  stream->open(file_name_, std::ios::binary);
  std::ifstream& reference = *stream;
  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillRepeatedly(testing::ReturnRef(reference));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);
  std::vector<int64_t> label_indices{0, 1, 2, 3};
  std::vector<std::vector<unsigned char>> samples = file_wrapper.get_samples_from_indices(label_indices);
  ASSERT_EQ(samples.size(), 4);
  ASSERT_EQ((samples)[0][0], '2');
  ASSERT_EQ((samples)[1][0], '4');
  ASSERT_EQ((samples)[2][0], '6');
  ASSERT_EQ((samples)[3][0], '8');

  label_indices = {1, 2, 3};
  samples = file_wrapper.get_samples_from_indices(label_indices);
  ASSERT_EQ(samples.size(), 3);
  ASSERT_EQ((samples)[0][0], '4');
  ASSERT_EQ((samples)[1][0], '6');
  ASSERT_EQ((samples)[2][0], '8');

  label_indices = {2};
  samples = file_wrapper.get_samples_from_indices(label_indices);
  ASSERT_EQ(samples.size(), 1);
  ASSERT_EQ((samples)[0][0], '6');

  label_indices = {1, 3};
  samples = file_wrapper.get_samples_from_indices(label_indices);
  ASSERT_EQ(samples.size(), 2);
  ASSERT_EQ((samples)[0][0], '4');
  ASSERT_EQ((samples)[1][0], '8');
}

TEST_F(BinaryFileWrapperTest, TestDeleteSamples) {
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillOnce(testing::Return(8));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);

  const std::vector<int64_t> label_indices{0, 1, 2, 3};

  ASSERT_NO_THROW(file_wrapper.delete_samples(label_indices));
}