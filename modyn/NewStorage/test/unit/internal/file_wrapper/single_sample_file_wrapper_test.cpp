#include "internal/file_wrapper/single_sample_file_wrapper.hpp"

#include <fstream>

#include "test_utils.hpp"
#include "unit/internal/filesystem_wrapper/mock_filesystem_wrapper.hpp"

using namespace storage;

TEST(SingleSampleFileWrapperTest, TestGetNumberOfSamples) {
  const std::string file_name = "test.txt";
  const YAML::Node config = TestUtils::get_dummy_file_wrapper_config();
  const MockFilesystemWrapper filesystem_wrapper;
  storage::SingleSampleFileWrapper file_wrapper =
      storage::SingleSampleFileWrapper(file_name, config, std::make_shared<MockFilesystemWrapper>(filesystem_wrapper));
  ASSERT_EQ(file_wrapper.get_number_of_samples(), 1);
}

TEST(SingleSampleFileWrapperTest, TestGetLabel) {
  const std::string file_name = "test.txt";
  const YAML::Node config = TestUtils::get_dummy_file_wrapper_config();
  MockFilesystemWrapper filesystem_wrapper;
  const std::vector<unsigned char> bytes = {'1', '2', '3', '4', '5', '6', '7', '8'};
  EXPECT_CALL(filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(bytes));
  storage::SingleSampleFileWrapper file_wrapper =
      storage::SingleSampleFileWrapper(file_name, config, std::make_shared<MockFilesystemWrapper>(filesystem_wrapper));
  ASSERT_EQ(file_wrapper.get_label(0), 12345678);
}

TEST(SingleSampleFileWrapperTest, TestGetAllLabels) {
  const std::string file_name = "test.txt";
  const YAML::Node config = TestUtils::get_dummy_file_wrapper_config();
  const std::vector<unsigned char> bytes = {'1', '2', '3', '4', '5', '6', '7', '8'};
  MockFilesystemWrapper filesystem_wrapper;
  EXPECT_CALL(filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(bytes));
  storage::SingleSampleFileWrapper file_wrapper =
      storage::SingleSampleFileWrapper(file_name, config, std::make_shared<MockFilesystemWrapper>(filesystem_wrapper));
  const std::vector<int64_t> labels = file_wrapper.get_all_labels();
  ASSERT_EQ(labels.size(), 1);
  ASSERT_EQ((labels)[0], 12345678);
}

TEST(SingleSampleFileWrapperTest, TestGetSamples) {
  const std::string file_name = "test.txt";
  const YAML::Node config = TestUtils::get_dummy_file_wrapper_config();
  const std::vector<unsigned char> bytes = {'1', '2', '3', '4', '5', '6', '7', '8'};
  MockFilesystemWrapper filesystem_wrapper;
  EXPECT_CALL(filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(bytes));
  storage::SingleSampleFileWrapper file_wrapper =
      storage::SingleSampleFileWrapper(file_name, config, std::make_shared<MockFilesystemWrapper>(filesystem_wrapper));
  const std::vector<std::vector<unsigned char>> samples = file_wrapper.get_samples(0, 1);
  ASSERT_EQ(samples.size(), 1);
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
  const YAML::Node config = TestUtils::get_dummy_file_wrapper_config();
  const std::vector<unsigned char> bytes = {'1', '2', '3', '4', '5', '6', '7', '8'};
  MockFilesystemWrapper filesystem_wrapper;
  EXPECT_CALL(filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(bytes));
  storage::SingleSampleFileWrapper file_wrapper =
      storage::SingleSampleFileWrapper(file_name, config, std::make_shared<MockFilesystemWrapper>(filesystem_wrapper));
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
  const YAML::Node config = TestUtils::get_dummy_file_wrapper_config();
  const std::vector<unsigned char> bytes = {'1', '2', '3', '4', '5', '6', '7', '8'};
  MockFilesystemWrapper filesystem_wrapper;
  EXPECT_CALL(filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(bytes));
  storage::SingleSampleFileWrapper file_wrapper =
      storage::SingleSampleFileWrapper(file_name, config, std::make_shared<MockFilesystemWrapper>(filesystem_wrapper));
  const std::vector<int64_t> indices = {0};
  const std::vector<std::vector<unsigned char>> samples = file_wrapper.get_samples_from_indices(indices);
  ASSERT_EQ(samples.size(), 1);
  ASSERT_EQ((samples)[0][0], '1');
  ASSERT_EQ((samples)[0][1], '2');
  ASSERT_EQ((samples)[0][2], '3');
  ASSERT_EQ((samples)[0][3], '4');
  ASSERT_EQ((samples)[0][4], '5');
  ASSERT_EQ((samples)[0][5], '6');
  ASSERT_EQ((samples)[0][6], '7');
  ASSERT_EQ((samples)[0][7], '8');
}