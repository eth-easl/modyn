#include "internal/file_wrapper/csv_file_wrapper.hpp"

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>

#include "gmock/gmock.h"
#include "internal/filesystem_wrapper/mock_filesystem_wrapper.hpp"
#include "modyn/utils/utils.hpp"
#include "storage_test_utils.hpp"
#include "test_utils.hpp"

using namespace modyn::storage;

class CsvFileWrapperTest : public ::testing::Test {
 protected:
  std::string file_name_;
  YAML::Node config_;
  std::shared_ptr<MockFilesystemWrapper> filesystem_wrapper_;
  std::string tmp_dir_ = modyn::test::TestUtils::get_tmp_testdir("csv_file_wrapper_test");

  CsvFileWrapperTest()
      : config_{StorageTestUtils::get_dummy_file_wrapper_config()},
        filesystem_wrapper_{std::make_shared<MockFilesystemWrapper>()} {
    file_name_ = tmp_dir_ + "/test.csv";
  }

  void SetUp() override {
    std::filesystem::create_directory(tmp_dir_);

    std::ofstream file(file_name_);
    file << "id,first_name,last_name,age\n";
    file << "1,John,Doe,25\n";
    file << "2,Jane,Smith,30\n";
    file << "3,Michael,Johnson,35\n";
    file.close();
    ASSERT_TRUE(std::filesystem::exists(file_name_));
  }

  void TearDown() override { std::filesystem::remove_all(file_name_); }
};

TEST_F(CsvFileWrapperTest, TestGetNumberOfSamples) {
  EXPECT_CALL(*filesystem_wrapper_, exists(testing::_)).WillOnce(testing::Return(true));
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);
  ASSERT_TRUE(stream_ptr->is_open());

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::Return(stream_ptr));
  CsvFileWrapper file_wrapper{file_name_, config_, filesystem_wrapper_};

  const uint64_t expected_number_of_samples = 3;
  const uint64_t actual_number_of_samples = file_wrapper.get_number_of_samples();

  ASSERT_EQ(actual_number_of_samples, expected_number_of_samples);
}

TEST_F(CsvFileWrapperTest, TestGetLabel) {
  EXPECT_CALL(*filesystem_wrapper_, exists(testing::_)).WillOnce(testing::Return(true));
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);
  ASSERT_TRUE(stream_ptr->is_open());

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::Return(stream_ptr));
  CsvFileWrapper file_wrapper{file_name_, config_, filesystem_wrapper_};

  const int64_t index = 1;
  const int64_t expected_label = 2;
  const int64_t actual_label = file_wrapper.get_label(index);

  ASSERT_EQ(actual_label, expected_label);

  const int64_t invalid_index = 3;
  ASSERT_THROW(file_wrapper.get_label(invalid_index), modyn::utils::ModynException);

  const int64_t negative_index = -1;
  ASSERT_THROW(file_wrapper.get_label(negative_index), modyn::utils::ModynException);
}

TEST_F(CsvFileWrapperTest, TestGetAllLabels) {
  EXPECT_CALL(*filesystem_wrapper_, exists(testing::_)).WillOnce(testing::Return(true));
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);
  ASSERT_TRUE(stream_ptr->is_open());

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::Return(stream_ptr));
  CsvFileWrapper file_wrapper{file_name_, config_, filesystem_wrapper_};

  const std::vector<int64_t> expected_labels = {1, 2, 3};
  const std::vector<int64_t> actual_labels = file_wrapper.get_all_labels();

  ASSERT_EQ(actual_labels, expected_labels);
}

TEST_F(CsvFileWrapperTest, TestGetSamples) {
  EXPECT_CALL(*filesystem_wrapper_, exists(testing::_)).WillOnce(testing::Return(true));
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);
  ASSERT_TRUE(stream_ptr->is_open());

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::Return(stream_ptr));
  CsvFileWrapper file_wrapper{file_name_, config_, filesystem_wrapper_};

  const int64_t start = 1;
  const int64_t end = 3;
  const std::vector<std::vector<unsigned char>> expected_samples = {
      {'J', 'a', 'n', 'e', ',', 'S', 'm', 'i', 't', 'h', ',', '3', '0'},
      {'M', 'i', 'c', 'h', 'a', 'e', 'l', ',', 'J', 'o', 'h', 'n', 's', 'o', 'n', ',', '3', '5'},
  };
  const std::vector<std::vector<unsigned char>> actual_samples = file_wrapper.get_samples(start, end);

  ASSERT_EQ(actual_samples, expected_samples);
}

TEST_F(CsvFileWrapperTest, TestGetSample) {
  EXPECT_CALL(*filesystem_wrapper_, exists(testing::_)).WillOnce(testing::Return(true));
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);
  ASSERT_TRUE(stream_ptr->is_open());

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::Return(stream_ptr));
  CsvFileWrapper file_wrapper{file_name_, config_, filesystem_wrapper_};

  const int64_t index = 1;
  const std::vector<unsigned char> expected_sample = {'J', 'a', 'n', 'e', ',', 'S', 'm', 'i', 't', 'h', ',', '3', '0'};
  const std::vector<unsigned char> actual_sample = file_wrapper.get_sample(index);

  ASSERT_EQ(actual_sample, expected_sample);
}

TEST_F(CsvFileWrapperTest, TestGetSamplesFromIndices) {
  EXPECT_CALL(*filesystem_wrapper_, exists(testing::_)).WillOnce(testing::Return(true));
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);
  ASSERT_TRUE(stream_ptr->is_open());

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::Return(stream_ptr));
  CsvFileWrapper file_wrapper{file_name_, config_, filesystem_wrapper_};

  const std::vector<uint64_t> indices = {0, 2};
  const std::vector<std::vector<unsigned char>> expected_samples = {
      {'J', 'o', 'h', 'n', ',', 'D', 'o', 'e', ',', '2', '5'},
      {'M', 'i', 'c', 'h', 'a', 'e', 'l', ',', 'J', 'o', 'h', 'n', 's', 'o', 'n', ',', '3', '5'},
  };
  const std::vector<std::vector<unsigned char>> actual_samples = file_wrapper.get_samples_from_indices(indices);

  ASSERT_EQ(actual_samples, expected_samples);
}

TEST_F(CsvFileWrapperTest, TestDeleteSamples) {
  EXPECT_CALL(*filesystem_wrapper_, exists(testing::_)).WillOnce(testing::Return(true));
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);
  ASSERT_TRUE(stream_ptr->is_open());

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::Return(stream_ptr));
  CsvFileWrapper file_wrapper{file_name_, config_, filesystem_wrapper_};

  const std::vector<uint64_t> indices = {0, 1};

  file_wrapper.delete_samples(indices);

  const std::vector<std::vector<unsigned char>> expected_samples = {
      {'M', 'i', 'c', 'h', 'a', 'e', 'l', ',', 'J', 'o', 'h', 'n', 's', 'o', 'n', ',', '3', '5'},
  };

  std::ifstream file2(file_name_, std::ios::binary);
  file2.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  file2.ignore(2);
  std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(file2), {});
  file2.close();
  buffer.pop_back();

  ASSERT_EQ(buffer, expected_samples[0]);
}
TEST_F(CsvFileWrapperTest, TestGetSamplesFromIndicesWithoutLabels) {
  // Create a test CSV file without labels
  std::ofstream file_without_labels(file_name_);
  file_without_labels << "first_name,last_name\n";
  file_without_labels << "John,Doe\n";
  file_without_labels << "Jane,Smith\n";
  file_without_labels << "Michael,Johnson\n";
  file_without_labels.close();
  ASSERT_TRUE(std::filesystem::exists(file_name_));
  config_["has_labels"] = false;
  EXPECT_CALL(*filesystem_wrapper_, exists(testing::_)).WillOnce(testing::Return(true));
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);
  ASSERT_TRUE(stream_ptr->is_open());

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::Return(stream_ptr));
  CsvFileWrapper file_wrapper{file_name_, config_, filesystem_wrapper_};

  // Test get_samples_from_indices without labels
  const std::vector<uint64_t> indices = {0, 2};
  const std::vector<std::vector<unsigned char>> expected_samples = {
      {'J', 'o', 'h', 'n', ',', 'D', 'o', 'e'},
      {'M', 'i', 'c', 'h', 'a', 'e', 'l', ',', 'J', 'o', 'h', 'n', 's', 'o', 'n'},
  };

  const std::vector<std::vector<unsigned char>> actual_samples = file_wrapper.get_samples_from_indices(indices);

  ASSERT_EQ(actual_samples, expected_samples);
}

TEST_F(CsvFileWrapperTest, TestGetSamplesWithoutLabels) {
  // Create a test CSV file without labels
  std::ofstream file_without_labels(file_name_);
  file_without_labels << "first_name,last_name\n";
  file_without_labels << "John,Doe\n";
  file_without_labels << "Jane,Smith\n";
  file_without_labels << "Michael,Johnson\n";
  file_without_labels.close();
  ASSERT_TRUE(std::filesystem::exists(file_name_));
  config_["has_labels"] = false;
  EXPECT_CALL(*filesystem_wrapper_, exists(testing::_)).WillOnce(testing::Return(true));
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);
  ASSERT_TRUE(stream_ptr->is_open());

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::Return(stream_ptr));
  CsvFileWrapper file_wrapper{file_name_, config_, filesystem_wrapper_};

  // Test get_samples without labels
  const uint64_t start = 0;
  const uint64_t end = 3;
  const std::vector<std::vector<unsigned char>> expected_samples = {
      {'J', 'o', 'h', 'n', ',', 'D', 'o', 'e'},
      {'J', 'a', 'n', 'e', ',', 'S', 'm', 'i', 't', 'h'},
      {'M', 'i', 'c', 'h', 'a', 'e', 'l', ',', 'J', 'o', 'h', 'n', 's', 'o', 'n'},
  };

  const std::vector<std::vector<unsigned char>> actual_samples = file_wrapper.get_samples(start, end);

  ASSERT_EQ(actual_samples, expected_samples);
}


TEST_F(CsvFileWrapperTest, TestGetTargetCategorical) {
  // Create a CSV with a categorical target in the third column
  std::ofstream file_with_cat_target(file_name_);
  file_with_cat_target << "id,feature,target\n";
  file_with_cat_target << "0,3.14,dog\n";
  file_with_cat_target << "1,1.23,cat\n";
  file_with_cat_target.close();

  // Indicate we have a target and which column it is in
  config_["has_targets"] = true;
  config_["target_index"] = 2;  // third column (0-based index)

  EXPECT_CALL(*filesystem_wrapper_, exists(::testing::_)).WillOnce(::testing::Return(true));
  auto stream_ptr = std::make_shared<std::ifstream>(file_name_, std::ios::binary);
  ASSERT_TRUE(stream_ptr->is_open());

  EXPECT_CALL(*filesystem_wrapper_, get_stream(::testing::_)).WillOnce(::testing::Return(stream_ptr));
  CsvFileWrapper file_wrapper{file_name_, config_, filesystem_wrapper_};

  // Check single target retrieval
  auto single_target = file_wrapper.get_target(0);
  ASSERT_EQ(std::vector<unsigned char>({'d', 'o', 'g'}), single_target);

  // Check multiple target retrieval
  auto multiple_targets = file_wrapper.get_targets(0, 2);
  ASSERT_EQ(2UL, multiple_targets.size());
  ASSERT_EQ(std::vector<unsigned char>({'d', 'o', 'g'}), multiple_targets[0]);
  ASSERT_EQ(std::vector<unsigned char>({'c', 'a', 't'}), multiple_targets[1]);
}

TEST_F(CsvFileWrapperTest, TestGetTargetNumeric) {

  std::ofstream file_with_target(file_name_);
  file_with_target << "id,feature,target\n";
  file_with_target << "0,3.14,42\n";
  file_with_target << "1,1.23,100\n";
  file_with_target.close();

  config_["has_targets"] = true;
  config_["target_index"] = 2;  // third column

  EXPECT_CALL(*filesystem_wrapper_, exists(::testing::_)).WillOnce(::testing::Return(true));
  auto stream_ptr = std::make_shared<std::ifstream>(file_name_, std::ios::binary);
  ASSERT_TRUE(stream_ptr->is_open());

  EXPECT_CALL(*filesystem_wrapper_, get_stream(::testing::_)).WillOnce(::testing::Return(stream_ptr));
  CsvFileWrapper file_wrapper{file_name_, config_, filesystem_wrapper_};

  // Check a single target
  auto t0 = file_wrapper.get_target(0);
  ASSERT_EQ(std::vector<unsigned char>({'4', '2'}), t0);

  // Check multiple targets
  auto targets = file_wrapper.get_targets(0, 2);
  ASSERT_EQ(2UL, targets.size());
  ASSERT_EQ(std::vector<unsigned char>({'4', '2'}), targets[0]);
  ASSERT_EQ(std::vector<unsigned char>({'1', '0', '0'}), targets[1]);
}

TEST_F(CsvFileWrapperTest, TestGetTargetsFromIndices) {
  // Create a CSV with a target column
  std::ofstream file_targets_indices(file_name_);
  file_targets_indices << "id,desc,target\n";
  file_targets_indices << "0,alpha,A\n";
  file_targets_indices << "1,beta,B\n";
  file_targets_indices << "2,gamma,C\n";
  file_targets_indices.close();

  config_["has_targets"] = true;
  config_["target_index"] = 2;  

  EXPECT_CALL(*filesystem_wrapper_, exists(::testing::_)).WillOnce(::testing::Return(true));
  auto stream_ptr = std::make_shared<std::ifstream>(file_name_, std::ios::binary);
  ASSERT_TRUE(stream_ptr->is_open());

  EXPECT_CALL(*filesystem_wrapper_, get_stream(::testing::_)).WillOnce(::testing::Return(stream_ptr));
  CsvFileWrapper file_wrapper{file_name_, config_, filesystem_wrapper_};

  const std::vector<uint64_t> indices = {2, 0};
  auto results = file_wrapper.get_targets_from_indices(indices);
  ASSERT_EQ(2UL, results.size());
  ASSERT_EQ(std::vector<unsigned char>({'C'}), results[0]);
  ASSERT_EQ(std::vector<unsigned char>({'A'}), results[1]);
}
