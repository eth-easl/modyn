#include "internal/file_wrapper/binary_file_wrapper.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <fstream>

#include "internal/filesystem_wrapper/mock_filesystem_wrapper.hpp"
#include "storage_test_utils.hpp"
#include "test_utils.hpp"

using namespace modyn::storage;

class ::modyn::storage::BinaryFileWrapperTest : public ::testing::Test {
 protected:
  std::string file_name_, file_name_endian_;
  YAML::Node config_;
  std::shared_ptr<MockFilesystemWrapper> filesystem_wrapper_;
  std::string tmp_dir_ = std::filesystem::temp_directory_path().string() + "/binary_file_wrapper_test";

  BinaryFileWrapperTest()
      : config_{StorageTestUtils::get_dummy_file_wrapper_config()},
        filesystem_wrapper_{std::make_shared<MockFilesystemWrapper>()} {
    file_name_ = tmp_dir_ + "/test.bin";
    file_name_endian_ = tmp_dir_ + "/test_endian.bin";
  }

  void SetUp() override {
    std::filesystem::create_directory(tmp_dir_);

    std::ofstream file(file_name_, std::ios::binary);
    const std::vector<std::pair<uint16_t, uint16_t>> data = {{42, 12}, {43, 13}, {44, 14}, {45, 15}};
    for (const auto& [payload, label] : data) {
      payload_to_file(file, payload, label);
    }
    file.close();

    // tmp test file with 2 byte labels
    std::ofstream file_endian(file_name_endian_, std::ios::binary);
    const std::vector<std::pair<uint16_t, uint16_t>> data_endian = {
        {(1 << 8) + 2, 0}, {(3 << 8) + 4, 1}, {(5 << 8) + 6, 2}, {(7 << 8) + 8, 3}};
    for (const auto& [payload, label] : data_endian) {
      // note: on macos & linux, the architecture's endianess is little
      payload_to_file(file_endian, payload, label);
    }
    file_endian.close();

    ASSERT_TRUE(std::filesystem::exists(file_name_));
    ASSERT_TRUE(std::filesystem::exists(file_endian));
  }

  static void payload_to_file(std::ofstream& file, uint16_t payload, uint16_t label) {
    file.write(reinterpret_cast<const char*>(&payload), sizeof(uint16_t));
    file.write(reinterpret_cast<const char*>(&label), sizeof(uint16_t));
  }

  void TearDown() override {
    std::filesystem::remove_all(file_name_);
    std::filesystem::remove_all(file_name_endian_);
  }

  // tests needs to be in class context to access private members
  void test_int_from_bytes() {
    // length: 1 byte
    const unsigned char bytes1[] = {0b00000001};
    ASSERT_EQ(BinaryFileWrapper::int_from_bytes_little_endian(bytes1, bytes1 + 1), 1);
    ASSERT_EQ(BinaryFileWrapper::int_from_bytes_big_endian(bytes1, bytes1 + 1), 1);

    // length: 2 byte
    const unsigned char bytes2[] = {0b00000001, 0b00000010};
    ASSERT_EQ(BinaryFileWrapper::int_from_bytes_little_endian(bytes2, bytes2 + 2), (2 << 8) + 1);
    ASSERT_EQ(BinaryFileWrapper::int_from_bytes_big_endian(bytes2, bytes2 + 2), (1 << 8) + 2);

    // length: 4 byte
    const unsigned char bytes4[] = {0b00000001, 0b00000010, 0b00000011, 0b00000100};
    ASSERT_EQ(BinaryFileWrapper::int_from_bytes_little_endian(bytes4, bytes4 + 4),
              (4LL << 24) + (3 << 16) + (2 << 8) + 1);
    ASSERT_EQ(BinaryFileWrapper::int_from_bytes_big_endian(bytes4, bytes4 + 4), (1 << 24) + (2 << 16) + (3 << 8) + 4);

    // length: 8 byte
    const unsigned char bytes8[] = {0b00000001, 0, 0, 0, 0, 0, 0, 0b00001000};
    ASSERT_EQ(BinaryFileWrapper::int_from_bytes_little_endian(bytes8, bytes8 + 8), (8LL << 56) + 1);
    ASSERT_EQ(BinaryFileWrapper::int_from_bytes_big_endian(bytes8, bytes8 + 8), (1LL << 56) + 8);
  }
};

TEST_F(BinaryFileWrapperTest, TestGetNumberOfSamples) {
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);
  ASSERT_TRUE(stream_ptr->is_open());

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::Return(stream_ptr));
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillOnce(testing::Return(16));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);
  ASSERT_EQ(file_wrapper.get_number_of_samples(), 4);

  stream_ptr->close();
}

TEST_F(BinaryFileWrapperTest, TestValidateFileExtension) {
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);
  ASSERT_TRUE(stream_ptr->is_open());

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::Return(stream_ptr));
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillOnce(testing::Return(16));
  ASSERT_NO_THROW(const BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_););
}

TEST_F(BinaryFileWrapperTest, TestValidateRequestIndices) {
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillOnce(testing::Return(16));
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);
  ASSERT_TRUE(stream_ptr->is_open());

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillRepeatedly(testing::Return(stream_ptr));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);
  std::vector<unsigned char> sample = file_wrapper.get_sample(0);

  ASSERT_EQ(sample.size(), 2);
  ASSERT_EQ((sample)[0], 12);

  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillOnce(testing::Return(16));
  BinaryFileWrapper file_wrapper2(file_name_, config_, filesystem_wrapper_);
  ASSERT_THROW(file_wrapper2.get_sample(8), modyn::utils::ModynException);
}

TEST_F(BinaryFileWrapperTest, TestIntFromBytes) { test_int_from_bytes(); }

TEST_F(BinaryFileWrapperTest, TestGetLabel) {
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillOnce(testing::Return(16));
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);
  ASSERT_TRUE(stream_ptr->is_open());

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::Return(stream_ptr));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);
  ASSERT_EQ(file_wrapper.get_label(0), 42);
  ASSERT_EQ(file_wrapper.get_label(1), 43);
  ASSERT_EQ(file_wrapper.get_label(2), 44);
  ASSERT_EQ(file_wrapper.get_label(3), 45);
}

TEST_F(BinaryFileWrapperTest, TestGetLabelEndian) {
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).Times(2).WillRepeatedly(testing::Return(16));
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_endian_, std::ios::binary);
  ASSERT_TRUE(stream_ptr->is_open());

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).Times(2).WillRepeatedly(testing::Return(stream_ptr));

  // [LITTLE ENDIAN]
  YAML::Node little_endian_config = StorageTestUtils::get_dummy_file_wrapper_config("little");
  BinaryFileWrapper file_wrapper_little_endian(file_name_endian_, little_endian_config, filesystem_wrapper_);
  ASSERT_EQ(file_wrapper_little_endian.get_label(0), (1 << 8) + 2);
  ASSERT_EQ(file_wrapper_little_endian.get_label(1), (3 << 8) + 4);
  ASSERT_EQ(file_wrapper_little_endian.get_label(2), (5 << 8) + 6);
  ASSERT_EQ(file_wrapper_little_endian.get_label(3), (7 << 8) + 8);

  // [BIG ENDIAN]
  YAML::Node big_endian_config = StorageTestUtils::get_dummy_file_wrapper_config("big");
  BinaryFileWrapper file_wrapper_big_endian(file_name_, big_endian_config, filesystem_wrapper_);
  ASSERT_EQ(file_wrapper_big_endian.get_label(0), (2 << 8) + 1);
  ASSERT_EQ(file_wrapper_big_endian.get_label(1), (4 << 8) + 3);
  ASSERT_EQ(file_wrapper_big_endian.get_label(2), (6 << 8) + 5);
  ASSERT_EQ(file_wrapper_big_endian.get_label(3), (8 << 8) + 7);
}

TEST_F(BinaryFileWrapperTest, TestGetAllLabels) {
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillOnce(testing::Return(16));
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);
  ASSERT_TRUE(stream_ptr->is_open());

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::Return(stream_ptr));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);
  std::vector<int64_t> labels = file_wrapper.get_all_labels();
  ASSERT_EQ(labels.size(), 4);
  ASSERT_EQ((labels)[0], 42);
  ASSERT_EQ((labels)[1], 43);
  ASSERT_EQ((labels)[2], 44);
  ASSERT_EQ((labels)[3], 45);
}

TEST_F(BinaryFileWrapperTest, TestGetAllLabelsEndian) {
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).Times(2).WillRepeatedly(testing::Return(16));
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_endian_, std::ios::binary);
  ASSERT_TRUE(stream_ptr->is_open());

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).Times(2).WillRepeatedly(testing::Return(stream_ptr));

  // [LITTLE ENDIAN]
  YAML::Node little_endian_config = StorageTestUtils::get_dummy_file_wrapper_config("little");
  BinaryFileWrapper file_wrapper_little_endian(file_name_, little_endian_config, filesystem_wrapper_);
  std::vector<int64_t> labels_little = file_wrapper_little_endian.get_all_labels();
  ASSERT_EQ(labels_little.size(), 4);
  ASSERT_EQ((labels_little)[0], (1 << 8) + 2);
  ASSERT_EQ((labels_little)[1], (3 << 8) + 4);
  ASSERT_EQ((labels_little)[2], (5 << 8) + 6);
  ASSERT_EQ((labels_little)[3], (7 << 8) + 8);

  // [BIG ENDIAN]
  YAML::Node big_endian_config = StorageTestUtils::get_dummy_file_wrapper_config("big");
  BinaryFileWrapper file_wrapper_big_endian(file_name_, big_endian_config, filesystem_wrapper_);
  std::vector<int64_t> labels_big = file_wrapper_big_endian.get_all_labels();
  ASSERT_EQ(labels_big.size(), 4);
  ASSERT_EQ((labels_big)[0], (2 << 8) + 1);
  ASSERT_EQ((labels_big)[1], (4 << 8) + 3);
  ASSERT_EQ((labels_big)[2], (6 << 8) + 5);
  ASSERT_EQ((labels_big)[3], (8 << 8) + 7);
}

TEST_F(BinaryFileWrapperTest, TestGetSample) {
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillRepeatedly(testing::Return(16));
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);
  ASSERT_TRUE(stream_ptr->is_open());

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillRepeatedly(testing::Return(stream_ptr));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);
  std::vector<unsigned char> sample = file_wrapper.get_sample(0);
  ASSERT_EQ(sample.size(), 2);
  ASSERT_EQ((sample)[0], 12);

  sample = file_wrapper.get_sample(1);
  ASSERT_EQ(sample.size(), 2);
  ASSERT_EQ((sample)[0], 13);

  sample = file_wrapper.get_sample(2);
  ASSERT_EQ(sample.size(), 2);
  ASSERT_EQ((sample)[0], 14);

  sample = file_wrapper.get_sample(3);
  ASSERT_EQ(sample.size(), 2);
  ASSERT_EQ((sample)[0], 15);
}

TEST_F(BinaryFileWrapperTest, TestGetSamples) {
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillRepeatedly(testing::Return(16));
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);
  ASSERT_TRUE(stream_ptr->is_open());

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillRepeatedly(testing::Return(stream_ptr));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);
  std::vector<std::vector<unsigned char>> samples = file_wrapper.get_samples(0, 3);
  ASSERT_EQ(samples.size(), 4);
  ASSERT_EQ((samples)[0][0], 12);
  ASSERT_EQ((samples)[1][0], 13);
  ASSERT_EQ((samples)[2][0], 14);
  ASSERT_EQ((samples)[3][0], 15);

  samples = file_wrapper.get_samples(1, 3);
  ASSERT_EQ(samples.size(), 3);
  ASSERT_EQ((samples)[0][0], 13);
  ASSERT_EQ((samples)[1][0], 14);
  ASSERT_EQ((samples)[2][0], 15);

  samples = file_wrapper.get_samples(2, 3);
  ASSERT_EQ(samples.size(), 2);
  ASSERT_EQ((samples)[0][0], 14);
  ASSERT_EQ((samples)[1][0], 15);

  samples = file_wrapper.get_samples(3, 3);
  ASSERT_EQ(samples.size(), 1);
  ASSERT_EQ((samples)[0][0], 15);

  ASSERT_THROW(file_wrapper.get_samples(4, 3), modyn::utils::ModynException);

  samples = file_wrapper.get_samples(1, 2);
  ASSERT_EQ(samples.size(), 2);
  ASSERT_EQ((samples)[0][0], 13);
  ASSERT_EQ((samples)[1][0], 14);
}

TEST_F(BinaryFileWrapperTest, TestGetSamplesFromIndices) {
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillRepeatedly(testing::Return(16));
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);
  ASSERT_TRUE(stream_ptr->is_open());

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillRepeatedly(testing::Return(stream_ptr));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);
  std::vector<uint64_t> label_indices{0, 1, 2, 3};
  std::vector<std::vector<unsigned char>> samples = file_wrapper.get_samples_from_indices(label_indices);
  ASSERT_EQ(samples.size(), 4);
  ASSERT_EQ((samples)[0][0], 12);
  ASSERT_EQ((samples)[1][0], 13);
  ASSERT_EQ((samples)[2][0], 14);
  ASSERT_EQ((samples)[3][0], 15);

  label_indices = {1, 2, 3};
  samples = file_wrapper.get_samples_from_indices(label_indices);
  ASSERT_EQ(samples.size(), 3);
  ASSERT_EQ((samples)[0][0], 13);
  ASSERT_EQ((samples)[1][0], 14);
  ASSERT_EQ((samples)[2][0], 15);

  label_indices = {2};
  samples = file_wrapper.get_samples_from_indices(label_indices);
  ASSERT_EQ(samples.size(), 1);
  ASSERT_EQ((samples)[0][0], 14);

  label_indices = {1, 3};
  samples = file_wrapper.get_samples_from_indices(label_indices);
  ASSERT_EQ(samples.size(), 2);
  ASSERT_EQ((samples)[0][0], 13);
  ASSERT_EQ((samples)[1][0], 15);

  label_indices = {3, 1, 3};
  samples = file_wrapper.get_samples_from_indices(label_indices);
  ASSERT_EQ(samples.size(), 3);
  ASSERT_EQ((samples)[0][0], 15);
  ASSERT_EQ((samples)[1][0], 13);
  ASSERT_EQ((samples)[2][0], 15);
}

TEST_F(BinaryFileWrapperTest, TestDeleteSamples) {
  const std::shared_ptr<std::ifstream> stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);
  ASSERT_TRUE(stream_ptr->is_open());

  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::Return(stream_ptr));
  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillOnce(testing::Return(16));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);

  const std::vector<uint64_t> label_indices{0, 1, 2, 3};

  ASSERT_NO_THROW(file_wrapper.delete_samples(label_indices));
}
