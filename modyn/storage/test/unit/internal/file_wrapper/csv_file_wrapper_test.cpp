#include "internal/file_wrapper/csv_file_wrapper.hpp"

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>

#include "gmock/gmock.h"
#include "test_utils.hpp"
#include "unit/internal/filesystem_wrapper/mock_filesystem_wrapper.hpp"

using namespace storage::file_wrapper;
using namespace storage::test;

class CsvFileWrapperTest : public ::testing::Test {
 protected:
  std::string file_name_;
  YAML::Node config_;
  std::shared_ptr<MockFilesystemWrapper> filesystem_wrapper_;

  CsvFileWrapperTest()
      : file_name_{"tmp/test.csv"},
        config_{TestUtils::get_dummy_file_wrapper_config()},
        filesystem_wrapper_{std::make_shared<MockFilesystemWrapper>()} {}

  void SetUp() override {
    std::filesystem::create_directory("tmp");

    std::ofstream file(file_name_);
    file << "id,first_name,last_name,age\n";
    file << "1,John,Doe,25\n";
    file << "2,Jane,Smith,30\n";
    file << "3,Michael,Johnson,35\n";
    file.close();
  }

  // void TearDown() override { std::filesystem::remove_all(file_name_); }
};

TEST_F(CsvFileWrapperTest, TestGetNumberOfSamples) {
  CsvFileWrapper file_wrapper{file_name_, config_, filesystem_wrapper_};
  const std::vector<std::string> csv_data = {
      "1,John,Doe,25\n",
      "2,Jane,Smith,30\n",
      "3,Michael,Johnson,35\n",
  };
  const std::string expected_file_content = TestUtils::join(csv_data);
  const std::vector<unsigned char> bytes(expected_file_content.begin(), expected_file_content.end());
  EXPECT_CALL(*filesystem_wrapper_, exists(testing::_)).WillOnce(testing::Return(true));
  std::ifstream file;
  file.open(file_name_, std::ios::binary);
  //EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::Return(file));  TODO: fix this

  const int64_t expected_number_of_samples = 3;
  const int64_t actual_number_of_samples = file_wrapper.get_number_of_samples();

  ASSERT_EQ(actual_number_of_samples, expected_number_of_samples);
}

TEST_F(CsvFileWrapperTest, TestGetLabel) {
  CsvFileWrapper file_wrapper{file_name_, config_, filesystem_wrapper_};
  const std::vector<std::string> csv_data = {
      "1,John,Doe,25\n",
      "2,Jane,Smith,30\n",
      "3,Michael,Johnson,35\n",
  };
  const std::string expected_file_content = TestUtils::join(csv_data);
  const std::vector<unsigned char> bytes(expected_file_content.begin(), expected_file_content.end());
  EXPECT_CALL(*filesystem_wrapper_, get(testing::_)).WillOnce(testing::Return(bytes));

  const int64_t index = 1;
  const int64_t expected_label = 2;
  const int64_t actual_label = file_wrapper.get_label(index);

  ASSERT_EQ(actual_label, expected_label);
}

TEST_F(CsvFileWrapperTest, TestGetAllLabels) {
  CsvFileWrapper file_wrapper{file_name_, config_, filesystem_wrapper_};
  const std::vector<std::string> csv_data = {
      "1,John,Doe,25\n",
      "2,Jane,Smith,30\n",
      "3,Michael,Johnson,35\n",
  };
  const std::string expected_file_content = TestUtils::join(csv_data);
  const std::vector<unsigned char> bytes(expected_file_content.begin(), expected_file_content.end());
  EXPECT_CALL(*filesystem_wrapper_, get(testing::_)).WillOnce(testing::Return(bytes));

  const std::vector<int64_t> expected_labels = {1, 2, 3};
  const std::vector<int64_t> actual_labels = file_wrapper.get_all_labels();

  ASSERT_EQ(actual_labels, expected_labels);
}

TEST_F(CsvFileWrapperTest, TestGetSamples) {
  CsvFileWrapper file_wrapper{file_name_, config_, filesystem_wrapper_};
  const std::vector<std::string> csv_data = {
      "1,John,Doe,25\n",
      "2,Jane,Smith,30\n",
      "3,Michael,Johnson,35\n",
  };
  const std::string expected_file_content = TestUtils::join(csv_data);
  const std::vector<unsigned char> bytes(expected_file_content.begin(), expected_file_content.end());
  EXPECT_CALL(*filesystem_wrapper_, get(testing::_)).WillOnce(testing::Return(bytes));

  const int64_t start = 1;
  const int64_t end = 3;
  const std::vector<std::vector<unsigned char>> expected_samples = {
      {'2', ',', 'J', 'a', 'n', 'e', ',', 'S', 'm', 'i', 't', 'h', ',', '3', '0'},
      {'3', ',', 'M', 'i', 'c', 'h', 'a', 'e', 'l', ',', 'J', 'o', 'h', 'n', 's', 'o', 'n', ',', '3', '5'},
  };
  const std::vector<std::vector<unsigned char>> actual_samples = file_wrapper.get_samples(start, end);

  ASSERT_EQ(actual_samples, expected_samples);
}

TEST_F(CsvFileWrapperTest, TestGetSample) {
  CsvFileWrapper file_wrapper{file_name_, config_, filesystem_wrapper_};
  const std::vector<std::string> csv_data = {
      "1,John,Doe,25\n",
      "2,Jane,Smith,30\n",
      "3,Michael,Johnson,35\n",
  };
  const std::string expected_file_content = TestUtils::join(csv_data);
  const std::vector<unsigned char> bytes(expected_file_content.begin(), expected_file_content.end());
  EXPECT_CALL(*filesystem_wrapper_, get(testing::_)).WillOnce(testing::Return(bytes));

  const int64_t index = 1;
  const std::vector<unsigned char> expected_sample = {'2', ',', 'J', 'a', 'n', 'e', ',', 'S',
                                                      'm', 'i', 't', 'h', ',', '3', '0'};
  const std::vector<unsigned char> actual_sample = file_wrapper.get_sample(index);

  ASSERT_EQ(actual_sample, expected_sample);
}

TEST_F(CsvFileWrapperTest, TestGetSamplesFromIndices) {
  CsvFileWrapper file_wrapper{file_name_, config_, filesystem_wrapper_};
  const std::vector<std::string> csv_data = {
      "1,John,Doe,25\n",
      "2,Jane,Smith,30\n",
      "3,Michael,Johnson,35\n",
  };
  const std::string expected_file_content = TestUtils::join(csv_data);
  const std::vector<unsigned char> bytes(expected_file_content.begin(), expected_file_content.end());
  EXPECT_CALL(*filesystem_wrapper_, get(testing::_)).WillOnce(testing::Return(bytes));

  const std::vector<int64_t> indices = {0, 2};
  const std::vector<std::vector<unsigned char>> expected_samples = {
      {'1', ',', 'J', 'o', 'h', 'n', ',', 'D', 'o', 'e', ',', '2', '5'},
      {'3', ',', 'M', 'i', 'c', 'h', 'a', 'e', 'l', ',', 'J', 'o', 'h', 'n', 's', 'o', 'n', ',', '3', '5'},
  };
  const std::vector<std::vector<unsigned char>> actual_samples = file_wrapper.get_samples_from_indices(indices);

  ASSERT_EQ(actual_samples, expected_samples);
}

TEST_F(CsvFileWrapperTest, TestDeleteSamples) {
  CsvFileWrapper file_wrapper{file_name_, config_, filesystem_wrapper_};
  const std::vector<int64_t> indices = {0, 1};

  ASSERT_THROW(file_wrapper.delete_samples(indices), std::runtime_error);
}
