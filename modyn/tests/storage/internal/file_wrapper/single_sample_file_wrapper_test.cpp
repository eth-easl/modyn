#include "internal/file_wrapper/single_sample_file_wrapper.hpp"

#include <fstream>

#include "internal/filesystem_wrapper/mock_filesystem_wrapper.hpp"
#include "storage_test_utils.hpp"
#include "test_utils.hpp"

using namespace modyn::storage;

TEST(SingleSampleFileWrapperTest, TestGetNumberOfSamples) {
  const std::string file_name = "test.txt";
  const YAML::Node config = StorageTestUtils::get_dummy_file_wrapper_config();
  const std::shared_ptr<MockFilesystemWrapper> filesystem_wrapper = std::make_shared<MockFilesystemWrapper>();
  ::SingleSampleFileWrapper file_wrapper = ::SingleSampleFileWrapper(file_name, config, filesystem_wrapper);
  ASSERT_EQ(file_wrapper.get_number_of_samples(), 1);
}

TEST(SingleSampleFileWrapperTest, TestGetLabel) {
  const std::string file_name = "test.txt";
  const YAML::Node config = StorageTestUtils::get_dummy_file_wrapper_config();
  const std::shared_ptr<MockFilesystemWrapper> filesystem_wrapper = std::make_shared<MockFilesystemWrapper>();
  const std::vector<unsigned char> bytes = {'1', '2', '3', '4', '5', '6', '7', '8'};
  EXPECT_CALL(*filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(bytes));
  EXPECT_CALL(*filesystem_wrapper, exists(testing::_)).WillOnce(testing::Return(true));
  ::SingleSampleFileWrapper file_wrapper = ::SingleSampleFileWrapper(file_name, config, filesystem_wrapper);
  ASSERT_EQ(file_wrapper.get_label(0), 12345678);
}

TEST(SingleSampleFileWrapperTest, TestGetAllLabels) {
  const std::string file_name = "test.txt";
  const YAML::Node config = StorageTestUtils::get_dummy_file_wrapper_config();
  const std::vector<unsigned char> bytes = {'1', '2', '3', '4', '5', '6', '7', '8'};
  const std::shared_ptr<MockFilesystemWrapper> filesystem_wrapper = std::make_shared<MockFilesystemWrapper>();
  EXPECT_CALL(*filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(bytes));
  EXPECT_CALL(*filesystem_wrapper, exists(testing::_)).WillOnce(testing::Return(true));
  ::SingleSampleFileWrapper file_wrapper = ::SingleSampleFileWrapper(file_name, config, filesystem_wrapper);
  const std::vector<int64_t> labels = file_wrapper.get_all_labels();
  ASSERT_EQ(labels.size(), 1);
  ASSERT_EQ((labels)[0], 12345678);
}

TEST(SingleSampleFileWrapperTest, TestGetSamples) {
  const std::string file_name = "test.txt";
  const YAML::Node config = StorageTestUtils::get_dummy_file_wrapper_config();
  const std::vector<unsigned char> bytes = {'1', '2', '3', '4', '5', '6', '7', '8'};
  const std::shared_ptr<MockFilesystemWrapper> filesystem_wrapper = std::make_shared<MockFilesystemWrapper>();
  EXPECT_CALL(*filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(bytes));
  ::SingleSampleFileWrapper file_wrapper = ::SingleSampleFileWrapper(file_name, config, filesystem_wrapper);
  const std::vector<std::vector<unsigned char>> samples = file_wrapper.get_samples(0, 1);
  ASSERT_EQ(samples.size(), 1);
  ASSERT_EQ(samples[0].size(), 8);
  ASSERT_EQ((samples)[0][0], '1');
  ASSERT_EQ((samples)[0][1], '2');
  ASSERT_EQ((samples)[0][2], '3');
  ASSERT_EQ((samples)[0][3], '4');
  ASSERT_EQ((samples)[0][4], '5');
  ASSERT_EQ((samples)[0][5], '6');
  ASSERT_EQ((samples)[0][6], '7');
  ASSERT_EQ((samples)[0][7], '8');
}

TEST(SingleSampleFileWrapperTest, TestGetSample) {
  const std::string file_name = "test.txt";
  const YAML::Node config = StorageTestUtils::get_dummy_file_wrapper_config();
  const std::vector<unsigned char> bytes = {'1', '2', '3', '4', '5', '6', '7', '8'};
  const std::shared_ptr<MockFilesystemWrapper> filesystem_wrapper = std::make_shared<MockFilesystemWrapper>();
  EXPECT_CALL(*filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(bytes));
  ::SingleSampleFileWrapper file_wrapper = ::SingleSampleFileWrapper(file_name, config, filesystem_wrapper);
  const std::vector<unsigned char> samples = file_wrapper.get_sample(0);
  ASSERT_EQ(samples.size(), 8);
  ASSERT_EQ((samples)[0], '1');
  ASSERT_EQ((samples)[1], '2');
  ASSERT_EQ((samples)[2], '3');
  ASSERT_EQ((samples)[3], '4');
  ASSERT_EQ((samples)[4], '5');
  ASSERT_EQ((samples)[5], '6');
  ASSERT_EQ((samples)[6], '7');
  ASSERT_EQ((samples)[7], '8');
}

TEST(SingleSampleFileWrapperTest, TestGetSamplesFromIndices) {
  const std::string file_name = "test.txt";
  const YAML::Node config = StorageTestUtils::get_dummy_file_wrapper_config();
  const std::vector<unsigned char> bytes = {'1', '2', '3', '4', '5', '6', '7', '8'};
  const std::shared_ptr<MockFilesystemWrapper> filesystem_wrapper = std::make_shared<MockFilesystemWrapper>();
  EXPECT_CALL(*filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(bytes));
  ::SingleSampleFileWrapper file_wrapper = ::SingleSampleFileWrapper(file_name, config, filesystem_wrapper);
  const std::vector<uint64_t> indices = {0};
  const std::vector<std::vector<unsigned char>> samples = file_wrapper.get_samples_from_indices(indices);
  ASSERT_EQ(samples.size(), 1);
  ASSERT_EQ(samples[0].size(), 8);
  ASSERT_EQ((samples)[0][0], '1');
  ASSERT_EQ((samples)[0][1], '2');
  ASSERT_EQ((samples)[0][2], '3');
  ASSERT_EQ((samples)[0][3], '4');
  ASSERT_EQ((samples)[0][4], '5');
  ASSERT_EQ((samples)[0][5], '6');
  ASSERT_EQ((samples)[0][6], '7');
  ASSERT_EQ((samples)[0][7], '8');
}

TEST(SingleSampleFileWrapperTest, TestDeleteSamples) {
  const std::shared_ptr<MockFilesystemWrapper> filesystem_wrapper = std::make_shared<MockFilesystemWrapper>();

  const std::string file_name = "test.txt";
  const YAML::Node config = StorageTestUtils::get_dummy_file_wrapper_config();

  ::SingleSampleFileWrapper file_wrapper = ::SingleSampleFileWrapper(file_name, config, filesystem_wrapper);

  const std::vector<uint64_t> indices = {0};
  file_wrapper.delete_samples(indices);
}
TEST(SingleSampleFileWrapperTest, TestGetSamplesFromIndicesWithoutLabels) {
  const std::string file_name = "test.txt";
  const YAML::Node config = StorageTestUtils::get_dummy_file_wrapper_config();
  const std::vector<unsigned char> bytes = {'1', '2', '3', '4', '5', '6', '7', '8'};
  const std::shared_ptr<MockFilesystemWrapper> filesystem_wrapper = std::make_shared<MockFilesystemWrapper>();
  EXPECT_CALL(*filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(bytes));

  ::SingleSampleFileWrapper file_wrapper = ::SingleSampleFileWrapper(file_name, config, filesystem_wrapper);
  const std::vector<uint64_t> indices = {0};
  const std::vector<std::vector<unsigned char>> samples = file_wrapper.get_samples_from_indices(indices);

  ASSERT_EQ(samples.size(), 1);
  ASSERT_EQ(samples[0].size(), 8);
  ASSERT_EQ((samples)[0][0], '1');
  ASSERT_EQ((samples)[0][1], '2');
  ASSERT_EQ((samples)[0][2], '3');
  ASSERT_EQ((samples)[0][3], '4');
  ASSERT_EQ((samples)[0][4], '5');
  ASSERT_EQ((samples)[0][5], '6');
  ASSERT_EQ((samples)[0][6], '7');
  ASSERT_EQ((samples)[0][7], '8');
}

TEST(SingleSampleFileWrapperTest, TestGetTargets) {
  YAML::Node config = StorageTestUtils::get_dummy_file_wrapper_config();
  config["target_file_extension"] = ".target";
  const std::string file_name = "sample_file.txt";
  const std::shared_ptr<MockFilesystemWrapper> filesystem_wrapper = std::make_shared<MockFilesystemWrapper>();

  // Only expect the target file to be checked and retrieved
  EXPECT_CALL(*filesystem_wrapper, exists(testing::_)).WillOnce(testing::Return(true));  // target file exists
  EXPECT_CALL(*filesystem_wrapper, get(testing::_))
      .WillOnce(testing::Return(std::vector<unsigned char>{'1', '0', '0'}));  // target file content

  SingleSampleFileWrapper file_wrapper(file_name, config, filesystem_wrapper);
  const auto targets = file_wrapper.get_targets(0, 1);
  ASSERT_EQ(targets.size(), 1);
  ASSERT_EQ(targets[0], (std::vector<unsigned char>{'1', '0', '0'}));
}
TEST(SingleSampleFileWrapperTest, TestGetTargetsFromIndices) {
  YAML::Node config = StorageTestUtils::get_dummy_file_wrapper_config();
  config["target_file_extension"] = ".target";
  const std::string file_name = "sample_file.txt";
  const std::shared_ptr<MockFilesystemWrapper> filesystem_wrapper = std::make_shared<MockFilesystemWrapper>();

  EXPECT_CALL(*filesystem_wrapper, exists(testing::_)).WillOnce(testing::Return(true));
  EXPECT_CALL(*filesystem_wrapper, get(testing::_))
      .WillOnce(testing::Return(std::vector<unsigned char>{'1', '0', '0'}));

  SingleSampleFileWrapper file_wrapper(file_name, config, filesystem_wrapper);
  const std::vector<uint64_t> indices = {0};
  const auto targets = file_wrapper.get_targets_from_indices(indices);
  ASSERT_EQ(targets.size(), 1);
  ASSERT_EQ(targets[0], (std::vector<unsigned char>{'1', '0', '0'}));
}
