#pragma once

#include "internal/file_wrapper/csv_file_wrapper.hpp"

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <fstream>

#include "gmock/gmock.h"
#include "test_utils.hpp"
#include "internal/filesystem_wrapper/mock_filesystem_wrapper.hpp"

namespace storage {

class CsvFileWrapperTest : public ::testing::Test {
 protected:
  std::string file_name_;
  YAML::Node config_;
  std::shared_ptr<MockFilesystemWrapper> filesystem_wrapper_;
  CsvFileWrapper file_wrapper_;

  void SetUp() override {
    file_name_ = "test.csv";
    config_ = TestUtils::get_dummy_file_wrapper_config();
    filesystem_wrapper_ = std::make_shared<MockFilesystemWrapper>();
    file_wrapper_ = CsvFileWrapper(file_name_, config_, filesystem_wrapper_);
  }
};

TEST_F(CsvFileWrapperTest, TestGetNumberOfSamples) {
  const std::vector<std::string> csv_data = {
      "1,John,Doe,25\n",
      "2,Jane,Smith,30\n",
      "3,Michael,Johnson,35\n",
  };
  const std::string expected_file_content = TestUtils::join(csv_data);
  const std::vector<unsigned char> bytes(expected_file_content.begin(), expected_file_content.end());
  EXPECT_CALL(*filesystem_wrapper_, get(file_name_)).WillOnce(Return(bytes));

  const int64_t expected_number_of_samples = 3;
  const int64_t actual_number_of_samples = file_wrapper_.get_number_of_samples();

  ASSERT_EQ(actual_number_of_samples, expected_number_of_samples);
}

TEST_F(CsvFileWrapperTest, TestGetLabel) {
  const std::vector<std::string> csv_data = {
      "1,John,Doe,25\n",
      "2,Jane,Smith,30\n",
      "3,Michael,Johnson,35\n",
  };
  const std::string expected_file_content = TestUtils::join(csv_data);
  const std::vector<unsigned char> bytes(expected_file_content.begin(), expected_file_content.end());
  EXPECT_CALL(*filesystem_wrapper_, get(file_name_)).WillOnce(Return(bytes));

  const int64_t index = 1;
  const int64_t expected_label = 2;
  const int64_t actual_label = file_wrapper_.get_label(index);

  ASSERT_EQ(actual_label, expected_label);
}

TEST_F(CsvFileWrapperTest, TestGetAllLabels) {
  const std::vector<std::string> csv_data = {
      "1,John,Doe,25\n",
      "2,Jane,Smith,30\n",
      "3,Michael,Johnson,35\n",
  };
  const std::string expected_file_content = TestUtils::join(csv_data);
  const std::vector<unsigned char> bytes(expected_file_content.begin(), expected_file_content.end());
  EXPECT_CALL(*filesystem_wrapper_, get(file_name_)).WillOnce(Return(bytes));

  const std::vector<int64_t> expected_labels = {1, 2, 3};
  const std::vector<int64_t> actual_labels = file_wrapper_.get_all_labels();

  ASSERT_EQ(actual_labels, expected_labels);
}

TEST_F(CsvFileWrapperTest, TestGetSamples) {
  const std::vector<std::string> csv_data = {
      "1,John,Doe,25\n",
      "2,Jane,Smith,30\n",
      "3,Michael,Johnson,35\n",
  };
  const std::string expected_file_content = TestUtils::join(csv_data);
  const std::vector<unsigned char> bytes(expected_file_content.begin(), expected_file_content.end());
  EXPECT_CALL(*filesystem_wrapper_, get(file_name_)).WillOnce(Return(bytes));

  const int64_t start = 1;
  const int64_t end = 3;
  const std::vector<std::vector<unsigned char>> expected_samples = {
      {'2', ',', 'J', 'a', 'n', 'e', ',', 'S', 'm', 'i', 't', 'h', ',', '3', '0', '\n'},
      {'3', ',', 'M', 'i', 'c', 'h', 'a', 'e', 'l', ',', 'J', 'o', 'h', 'n', 's', 'o', 'n', ',', '3', '5', '\n'},
  };
  const std::vector<std::vector<unsigned char>> actual_samples = file_wrapper_.get_samples(start, end);

  ASSERT_EQ(actual_samples, expected_samples);
}

TEST_F(CsvFileWrapperTest, TestGetSample) {
  const std::vector<std::string> csv_data = {
      "1,John,Doe,25\n",
      "2,Jane,Smith,30\n",
      "3,Michael,Johnson,35\n",
  };
  const std::string expected_file_content = TestUtils::join(csv_data);
  const std::vector<unsigned char> bytes(expected_file_content.begin(), expected_file_content.end());
  EXPECT_CALL(*filesystem_wrapper_, get(file_name_)).WillOnce(Return(bytes));

  const int64_t index = 1;
  const std::vector<unsigned char> expected_sample = {'2', ',', 'J', 'a', 'n', 'e', ',', 'S',
                                                      'm', 'i', 't', 'h', ',', '3', '0', '\n'};
  const std::vector<unsigned char> actual_sample = file_wrapper_.get_sample(index);

  ASSERT_EQ(actual_sample, expected_sample);
}

TEST_F(CsvFileWrapperTest, TestGetSamplesFromIndices) {
  const std::vector<std::string> csv_data = {
      "1,John,Doe,25\n",
      "2,Jane,Smith,30\n",
      "3,Michael,Johnson,35\n",
  };
  const std::string expected_file_content = TestUtils::join(csv_data);
  const std::vector<unsigned char> bytes(expected_file_content.begin(), expected_file_content.end());
  EXPECT_CALL(*filesystem_wrapper_, get(file_name_)).WillOnce(Return(bytes));

  const std::vector<int64_t> indices = {0, 2};
  const std::vector<std::vector<unsigned char>> expected_samples = {
      {'1', ',', 'J', 'o', 'h', 'n', ',', 'D', 'o', 'e', ',', '2', '5', '\n'},
      {'3', ',', 'M', 'i', 'c', 'h', 'a', 'e', 'l', ',', 'J', 'o', 'h', 'n', 's', 'o', 'n', ',', '3', '5', '\n'},
  };
  const std::vector<std::vector<unsigned char>> actual_samples = file_wrapper_.get_samples_from_indices(indices);

  ASSERT_EQ(actual_samples, expected_samples);
}

TEST_F(CsvFileWrapperTest, TestDeleteSamples) {
  const std::vector<int64_t> indices = {0, 1};
  EXPECT_CALL(*filesystem_wrapper_, remove(file_name_)).Times(indices.size());

  file_wrapper_.delete_samples(indices);
}

}  // namespace storage
