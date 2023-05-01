#include "../../../src/internal/file_wrapper/BinaryFileWrapper.hpp"
#include "MockFilesystemWrapper.hpp"
#include <fstream>
#include "../../Utils.hpp"

using namespace storage;

TEST(BinaryFileWrapperTest, TestGetNumberOfSamples)
{
    std::string file_name = "test.bin";
    YAML::Node config = Utils::get_dummy_config();
    MockFileSystemWrapper *filesystem_wrapper;
    EXPECT_CALL(*filesystem_wrapper, get_file_size(testing::_)).WillOnce(testing::Return(1));
    storage::BinaryFileWrapper file_wrapper(file_name, config, filesystem_wrapper);
    ASSERT_EQ(file_wrapper.get_number_of_samples(), 1);
}

TEST(BinaryFileWrapperTest, TestValidateFileExtension)
{
    std::string file_name = "test.bin";
    YAML::Node config = Utils::get_dummy_config();
    MockFileSystemWrapper *filesystem_wrapper;
    EXPECT_CALL(*filesystem_wrapper, get_file_size(testing::_)).WillOnce(testing::Return(1));
    ASSERT_NO_THROW(storage::BinaryFileWrapper file_wrapper(file_name, config, filesystem_wrapper));

    file_name = "test.txt";
    EXPECT_CALL(*filesystem_wrapper, get_file_size(testing::_)).WillOnce(testing::Return(1));
    ASSERT_THROW(storage::BinaryFileWrapper file_wrapper2(file_name, config, filesystem_wrapper), std::invalid_argument);
}

TEST(BinaryFileWrapperTest, TestValidateRequestIndices)
{
    std::string file_name = "test.bin";
    YAML::Node config = Utils::get_dummy_config();
    MockFileSystemWrapper *filesystem_wrapper;
    EXPECT_CALL(*filesystem_wrapper, get_file_size(testing::_)).WillOnce(testing::Return(1));
    storage::BinaryFileWrapper file_wrapper(file_name, config, filesystem_wrapper);
    ASSERT_NO_THROW(file_wrapper.get_sample(0));

    EXPECT_CALL(*filesystem_wrapper, get_file_size(testing::_)).WillOnce(testing::Return(1));
    storage::BinaryFileWrapper file_wrapper2(file_name, config, filesystem_wrapper);
    ASSERT_THROW(file_wrapper2.get_sample(1), std::runtime_error);
}

TEST(BinaryFileWrapperTest, TestGetLabel)
{
    std::string file_name = "test.bin";
    YAML::Node config = Utils::get_dummy_config();
    MockFileSystemWrapper *filesystem_wrapper;
    std::vector<unsigned char> *bytes = new std::vector<unsigned char>{'1', '2', '3', '4', '5', '6', '7', '8'};
    EXPECT_CALL(*filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(bytes));
    storage::BinaryFileWrapper file_wrapper(file_name, config, filesystem_wrapper);
    ASSERT_EQ(file_wrapper.get_label(0), 0x04030201);
}

TEST(BinaryFileWrapperTest, TestGetAllLabels)
{
    std::string file_name = "test.bin";
    YAML::Node config = Utils::get_dummy_config();
    MockFileSystemWrapper *filesystem_wrapper;
    std::vector<unsigned char> *bytes = new std::vector<unsigned char>{'1', '2', '3', '4', '5', '6', '7', '8'};
    EXPECT_CALL(*filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(bytes));
    storage::BinaryFileWrapper file_wrapper(file_name, config, filesystem_wrapper);
    std::vector<std::vector<int>> *labels = file_wrapper.get_all_labels();
    ASSERT_EQ(labels->size(), 1);
    ASSERT_EQ((*labels)[0][0], 0x04030201);
}

TEST(BinaryFileWrapperTest, TestGetSample)
{
    std::string file_name = "test.bin";
    YAML::Node config = Utils::get_dummy_config();
    MockFileSystemWrapper *filesystem_wrapper;
    std::vector<unsigned char> *bytes = new std::vector<unsigned char>{'1', '2', '3', '4', '5', '6', '7', '8'};
    EXPECT_CALL(*filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(bytes));
    storage::BinaryFileWrapper file_wrapper(file_name, config, filesystem_wrapper);
    std::vector<unsigned char> *sample = file_wrapper.get_sample(0);
    ASSERT_EQ(sample->size(), 2);
    ASSERT_EQ((*sample)[0], 0x04030201);
    ASSERT_EQ((*sample)[1], 0x08070605);
}

TEST(BinaryFileWrapperTest, TestGetAllSamples)
{
    std::string file_name = "test.bin";
    YAML::Node config = Utils::get_dummy_config();
    MockFileSystemWrapper *filesystem_wrapper;
    std::vector<unsigned char> *bytes = new std::vector<unsigned char>{'1', '2', '3', '4', '5', '6', '7', '8'};
    EXPECT_CALL(*filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(bytes));
    storage::BinaryFileWrapper file_wrapper(file_name, config, filesystem_wrapper);
    std::vector<std::vector<unsigned char>> *samples = file_wrapper.get_samples(0, 1);
    ASSERT_EQ(samples->size(), 1);
    ASSERT_EQ((*samples)[0][0], 0x04030201);
    ASSERT_EQ((*samples)[0][1], 0x08070605);
}

TEST(BinaryFileWrapperTest, TestGetSamplesFromIndices)
{
    std::string file_name = "test.bin";
    YAML::Node config = Utils::get_dummy_config();
    MockFileSystemWrapper *filesystem_wrapper;
    std::vector<unsigned char> *bytes = new std::vector<unsigned char>{'1', '2', '3', '4', '5', '6', '7', '8'};
    EXPECT_CALL(*filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(bytes));
    storage::BinaryFileWrapper file_wrapper(file_name, config, filesystem_wrapper);
    std::vector<int> *indices = new std::vector<int>{0, 1, 2};
    std::vector<std::vector<unsigned char>> *samples = file_wrapper.get_samples_from_indices(indices);
    ASSERT_EQ(samples->size(), 1);
    ASSERT_EQ((*samples)[0][0], 0x04030201);
    ASSERT_EQ((*samples)[0][1], 0x08070605);
}
