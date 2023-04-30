#include "../../../src/internal/file_wrapper/SingleSampleFileWrapper.h"
#include "MockFilesystemWrapper.cpp"
#include <gtest/gtest.h>
#include "gmock/gmock.h"
#include <fstream>

using namespace storage;

YAML::Node get_dummy_config()
{
    YAML::Node config;
    config["file_extension"] = ".txt";
    config["label_file_extension"] = ".json";
    return config;
}

TEST(SingleSampleFileWrapperTest, TestGetNumberOfSamples)
{
    std::string file_name = "test.txt";
    YAML::Node config = get_dummy_config();
    MockFileSystemWrapper *filesystem_wrapper;
    EXPECT_CALL(*filesystem_wrapper, get_file_size(testing::_)).WillOnce(testing::Return(1));
    storage::SingleSampleFileWrapper file_wrapper(file_name, config, filesystem_wrapper);
    ASSERT_EQ(file_wrapper.get_number_of_samples(), 1);
}

TEST(SingleSampleFileWrapperTest, TestGetLabel)
{
    std::string file_name = "test.txt";
    YAML::Node config = get_dummy_config();
    MockFileSystemWrapper *filesystem_wrapper;
    EXPECT_CALL(*filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(std::vector<unsigned char>{'4'}));
    storage::SingleSampleFileWrapper file_wrapper(file_name, config, filesystem_wrapper);
    ASSERT_EQ(file_wrapper.get_label(0), 4);
}

TEST(SingleSampleFileWrapperTest, TestGetAllLabels)
{
    std::string file_name = "test.txt";
    YAML::Node config = get_dummy_config();
    MockFileSystemWrapper *filesystem_wrapper;
    EXPECT_CALL(*filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(std::vector<unsigned char>{'4'}));
    storage::SingleSampleFileWrapper file_wrapper(file_name, config, filesystem_wrapper);
    std::vector<std::vector<int>> *labels = file_wrapper.get_all_labels();
    ASSERT_EQ(labels->size(), 1);
    ASSERT_EQ((*labels)[0][0], 4);
}

TEST(SingleSampleFileWrapperTest, TestGetSamples)
{
    std::string file_name = "test.txt";
    YAML::Node config = get_dummy_config();
    MockFileSystemWrapper *filesystem_wrapper;
    EXPECT_CALL(*filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(std::vector<unsigned char>{'1'}));
    storage::SingleSampleFileWrapper file_wrapper(file_name, config, filesystem_wrapper);
    std::vector<std::vector<unsigned char>> *samples = file_wrapper.get_samples(0, 1);
    ASSERT_EQ(samples->size(), 1);
    ASSERT_EQ((*samples)[0][0], '1');
}

TEST(SingleSampleFileWrapperTest, TestGetSample)
{
    std::string file_name = "test.txt";
    YAML::Node config = get_dummy_config();
    MockFileSystemWrapper *filesystem_wrapper;
    EXPECT_CALL(*filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(std::vector<unsigned char>{'1'}));
    storage::SingleSampleFileWrapper file_wrapper(file_name, config, filesystem_wrapper);
    std::vector<unsigned char> *sample = file_wrapper.get_sample(0);
    ASSERT_EQ(sample->size(), 1);
    ASSERT_EQ((*sample)[0], '1');
}

TEST(SingleSampleFileWrapperTest, TestGetSamplesFromIndices)
{
    std::string file_name = "test.txt";
    YAML::Node config = get_dummy_config();
    MockFileSystemWrapper *filesystem_wrapper;
    EXPECT_CALL(*filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(std::vector<unsigned char>{'1'}));
    storage::SingleSampleFileWrapper file_wrapper(file_name, config, filesystem_wrapper);
    std::vector<std::vector<unsigned char>> *samples = file_wrapper.get_samples_from_indices(new std::vector<int>{0});
    ASSERT_EQ(samples->size(), 1);
    ASSERT_EQ((*samples)[0][0], '1');
}