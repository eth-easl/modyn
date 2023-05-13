#include "internal/file_wrapper/single_sample_file_wrapper.hpp"

#include <fstream>

#include "test_utils.hpp"
#include "unit/internal/filesystem_wrapper/mock_filesystem_wrapper.hpp"

using namespace storage;

TEST(SingleSampleFileWrapperTest, TestGetNumberOfSamples) {
  std::string file_name = "test.txt";
  YAML::Node config = TestUtils::get_dummy_file_wrapper_config();
  MockFilesystemWrapper filesystem_wrapper;
  storage::SingleSampleFileWrapper file_wrapper(file_name, config, &filesystem_wrapper);
  ASSERT_EQ(file_wrapper.get_number_of_samples(), 1);
}

TEST(SingleSampleFileWrapperTest, TestGetLabel) {
  std::string file_name = "test.txt";
  YAML::Node config = TestUtils::get_dummy_file_wrapper_config();
  MockFilesystemWrapper filesystem_wrapper;
  std::vector<unsigned char>* bytes = new std::vector<unsigned char>{'1', '2', '3', '4', '5', '6', '7', '8'};
  EXPECT_CALL(filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(bytes));
  storage::SingleSampleFileWrapper file_wrapper(file_name, config, &filesystem_wrapper);
  ASSERT_EQ(file_wrapper.get_label(0), 12345678);
}

TEST(SingleSampleFileWrapperTest, TestGetAllLabels) {
  std::string file_name = "test.txt";
  YAML::Node config = TestUtils::get_dummy_file_wrapper_config();
  MockFilesystemWrapper filesystem_wrapper;
  std::vector<unsigned char>* bytes = new std::vector<unsigned char>{'1', '2', '3', '4', '5', '6', '7', '8'};
  EXPECT_CALL(filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(bytes));
  storage::SingleSampleFileWrapper file_wrapper(file_name, config, &filesystem_wrapper);
  std::vector<int>* labels = file_wrapper.get_all_labels();
  ASSERT_EQ(labels->size(), 1);
  ASSERT_EQ((*labels)[0], 12345678);
}

TEST(SingleSampleFileWrapperTest, TestGetSamples) {
  std::string file_name = "test.txt";
  YAML::Node config = TestUtils::get_dummy_file_wrapper_config();
  MockFilesystemWrapper filesystem_wrapper;
  std::vector<unsigned char>* bytes = new std::vector<unsigned char>{'1', '2', '3', '4', '5', '6', '7', '8'};
  EXPECT_CALL(filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(bytes));
  storage::SingleSampleFileWrapper file_wrapper(file_name, config, &filesystem_wrapper);
  std::vector<std::vector<unsigned char>>* samples = file_wrapper.get_samples(0, 1);
  ASSERT_EQ(samples->size(), 1);
  ASSERT_EQ((*samples)[0][0], '1');
  ASSERT_EQ((*samples)[0][1], '2');
  ASSERT_EQ((*samples)[0][2], '3');
  ASSERT_EQ((*samples)[0][3], '4');
  ASSERT_EQ((*samples)[0][4], '5');
  ASSERT_EQ((*samples)[0][5], '6');
  ASSERT_EQ((*samples)[0][6], '7');
  ASSERT_EQ((*samples)[0][7], '8');
}

TEST(SingleSampleFileWrapperTest, TestGetSample) {
  std::string file_name = "test.txt";
  YAML::Node config = TestUtils::get_dummy_file_wrapper_config();
  MockFilesystemWrapper filesystem_wrapper;
  std::vector<unsigned char>* bytes = new std::vector<unsigned char>{'1', '2', '3', '4', '5', '6', '7', '8'};
  EXPECT_CALL(filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(bytes));
  storage::SingleSampleFileWrapper file_wrapper(file_name, config, &filesystem_wrapper);
  std::vector<unsigned char>* sample = file_wrapper.get_sample(0);
  ASSERT_EQ(sample->size(), 8);
  ASSERT_EQ((*sample)[0], '1');
  ASSERT_EQ((*sample)[1], '2');
  ASSERT_EQ((*sample)[2], '3');
  ASSERT_EQ((*sample)[3], '4');
  ASSERT_EQ((*sample)[4], '5');
  ASSERT_EQ((*sample)[5], '6');
  ASSERT_EQ((*sample)[6], '7');
  ASSERT_EQ((*sample)[7], '8');
}

TEST(SingleSampleFileWrapperTest, TestGetSamplesFromIndices) {
  std::string file_name = "test.txt";
  YAML::Node config = TestUtils::get_dummy_file_wrapper_config();
  MockFilesystemWrapper filesystem_wrapper;
  std::vector<unsigned char>* bytes = new std::vector<unsigned char>{'1', '2', '3', '4', '5', '6', '7', '8'};
  EXPECT_CALL(filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(bytes));
  storage::SingleSampleFileWrapper file_wrapper(file_name, config, &filesystem_wrapper);
  std::vector<std::vector<unsigned char>>* samples = file_wrapper.get_samples_from_indices(new std::vector<int>{0});
  ASSERT_EQ(samples->size(), 1);
  ASSERT_EQ((*samples)[0][0], '1');
  ASSERT_EQ((*samples)[0][1], '2');
  ASSERT_EQ((*samples)[0][2], '3');
  ASSERT_EQ((*samples)[0][3], '4');
  ASSERT_EQ((*samples)[0][4], '5');
  ASSERT_EQ((*samples)[0][5], '6');
  ASSERT_EQ((*samples)[0][6], '7');
  ASSERT_EQ((*samples)[0][7], '8');
}