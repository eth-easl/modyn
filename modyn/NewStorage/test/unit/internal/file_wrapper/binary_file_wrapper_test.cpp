#include "internal/file_wrapper/binary_file_wrapper.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <fstream>

#include "test_utils.hpp"
#include "unit/internal/filesystem_wrapper/mock_filesystem_wrapper.hpp"

using namespace storage;

TEST(BinaryFileWrapperTest, TestGetNumberOfSamples) {
  const std::string file_name = "test.bin";
  const YAML::Node config = TestUtils::get_dummy_file_wrapper_config();
  MockFilesystemWrapper filesystem_wrapper;
  EXPECT_CALL(filesystem_wrapper, get_file_size(testing::_)).WillOnce(testing::Return(8));
  storage::BinaryFileWrapper file_wrapper = storage::BinaryFileWrapper(file_name, config, &filesystem_wrapper);
  ASSERT_EQ(file_wrapper.get_number_of_samples(), 4);
}

TEST(BinaryFileWrapperTest, TestValidateFileExtension) {
  std::string file_name = "test.bin";
  const YAML::Node config = TestUtils::get_dummy_file_wrapper_config();
  MockFilesystemWrapper filesystem_wrapper;
  EXPECT_CALL(filesystem_wrapper, get_file_size(testing::_)).WillOnce(testing::Return(8));
  ASSERT_NO_THROW(const storage::BinaryFileWrapper file_wrapper =
                      storage::BinaryFileWrapper(file_name, config, &filesystem_wrapper));

  file_name = "test.txt";
  ASSERT_THROW(
      const storage::BinaryFileWrapper file_wrapper2 = storage::BinaryFileWrapper(file_name, config, &filesystem_wrapper),
      std::invalid_argument);
}

TEST(BinaryFileWrapperTest, TestValidateRequestIndices) {
  const std::string file_name = "test.bin";
  const YAML::Node config = TestUtils::get_dummy_file_wrapper_config();
  MockFilesystemWrapper filesystem_wrapper;
  EXPECT_CALL(filesystem_wrapper, get_file_size(testing::_)).WillOnce(testing::Return(8));
  EXPECT_CALL(filesystem_wrapper, get(testing::_))
      .WillOnce(testing::Return(new std::vector<unsigned char>{'1', '2', '3', '4', '5', '6', '7', '8'}));
  storage::BinaryFileWrapper file_wrapper(file_name, config, &filesystem_wrapper);
  ASSERT_NO_THROW(file_wrapper.get_sample(0));

  EXPECT_CALL(filesystem_wrapper, get_file_size(testing::_)).WillOnce(testing::Return(8));
  storage::BinaryFileWrapper file_wrapper2(file_name, config, &filesystem_wrapper);
  ASSERT_THROW(file_wrapper2.get_sample(8), std::out_of_range);
}

TEST(BinaryFileWrapperTest, TestGetLabel) {
  const std::string file_name = "test.bin";
  const YAML::Node config = TestUtils::get_dummy_file_wrapper_config();
  MockFilesystemWrapper filesystem_wrapper;
  auto* bytes = new std::vector<unsigned char>{1, 2, 3, 4, 5, 6, 7, 8};
  EXPECT_CALL(filesystem_wrapper, get_file_size(testing::_)).WillOnce(testing::Return(8));
  EXPECT_CALL(filesystem_wrapper, get(testing::_)).WillRepeatedly(testing::Return(bytes));
  storage::BinaryFileWrapper file_wrapper(file_name, config, &filesystem_wrapper);
  ASSERT_EQ(file_wrapper.get_label(0), 1);
  ASSERT_EQ(file_wrapper.get_label(1), 3);
  ASSERT_EQ(file_wrapper.get_label(2), 5);
  ASSERT_EQ(file_wrapper.get_label(3), 7);
}

TEST(BinaryFileWrapperTest, TestGetAllLabels) {
  const std::string file_name = "test.bin";
  const YAML::Node config = TestUtils::get_dummy_file_wrapper_config();
  MockFilesystemWrapper filesystem_wrapper;
  auto* bytes = new std::vector<unsigned char>{1, 2, 3, 4, 5, 6, 7, 8};
  EXPECT_CALL(filesystem_wrapper, get_file_size(testing::_)).WillOnce(testing::Return(8));
  EXPECT_CALL(filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(bytes));
  storage::BinaryFileWrapper file_wrapper(file_name, config, &filesystem_wrapper);
  std::vector<int>* labels = file_wrapper.get_all_labels();
  ASSERT_EQ(labels->size(), 4);
  ASSERT_EQ((*labels)[0], 1);
  ASSERT_EQ((*labels)[1], 3);
  ASSERT_EQ((*labels)[2], 5);
  ASSERT_EQ((*labels)[3], 7);
}

TEST(BinaryFileWrapperTest, TestGetSample) {
  const std::string file_name = "test.bin";
  const YAML::Node config = TestUtils::get_dummy_file_wrapper_config();
  MockFilesystemWrapper filesystem_wrapper;
  auto* bytes = new std::vector<unsigned char>{1, 2, 3, 4, 5, 6, 7, 8};
  EXPECT_CALL(filesystem_wrapper, get_file_size(testing::_)).WillOnce(testing::Return(8));
  EXPECT_CALL(filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(bytes));
  storage::BinaryFileWrapper file_wrapper(file_name, config, &filesystem_wrapper);
  std::vector<unsigned char>* sample = file_wrapper.get_sample(0);
  ASSERT_EQ(sample->size(), 1);
  ASSERT_EQ((*sample)[0], 2);
}

TEST(BinaryFileWrapperTest, TestGetAllSamples) {
  const std::string file_name = "test.bin";
  const YAML::Node config = TestUtils::get_dummy_file_wrapper_config();
  MockFilesystemWrapper filesystem_wrapper;
  auto* bytes = new std::vector<unsigned char>{1, 2, 3, 4, 5, 6, 7, 8};
  EXPECT_CALL(filesystem_wrapper, get_file_size(testing::_)).WillOnce(testing::Return(8));
  EXPECT_CALL(filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(bytes));
  storage::BinaryFileWrapper file_wrapper(file_name, config, &filesystem_wrapper);
  std::vector<std::vector<unsigned char>>* samples = file_wrapper.get_samples(0, 2);
  ASSERT_EQ(samples->size(), 2);
  ASSERT_EQ((*samples)[0][0], 2);
  ASSERT_EQ((*samples)[1][0], 4);
}

TEST(BinaryFileWrapperTest, TestGetSamplesFromIndices) {
  const std::string file_name = "test.bin";
  const YAML::Node config = TestUtils::get_dummy_file_wrapper_config();
  MockFilesystemWrapper filesystem_wrapper;
  auto* bytes = new std::vector<unsigned char>{1, 2, 3, 4, 5, 6, 7, 8};
  EXPECT_CALL(filesystem_wrapper, get_file_size(testing::_)).WillOnce(testing::Return(8));
  EXPECT_CALL(filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(bytes));
  storage::BinaryFileWrapper file_wrapper(file_name, config, &filesystem_wrapper);
  auto* indices = new std::vector<int64_t>{0, 1, 2};
  std::vector<std::vector<unsigned char>>* samples = file_wrapper.get_samples_from_indices(indices);
  ASSERT_EQ(samples->size(), 3);
  ASSERT_EQ((*samples)[0][0], 2);
  ASSERT_EQ((*samples)[1][0], 4);
  ASSERT_EQ((*samples)[2][0], 6);
}
