#include "internal/file_wrapper/binary_file_wrapper.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>

#include "internal/filesystem_wrapper/mock_filesystem_wrapper.hpp"
#include "storage_test_utils.hpp"
#include "test_utils.hpp"

namespace modyn::storage {

class BinaryFileWrapperTest : public ::testing::Test {
 protected:
  std::string file_name_, file_name_endian_;
  YAML::Node config_;
  std::shared_ptr<MockFilesystemWrapper> filesystem_wrapper_;
  std::string tmp_dir_ = modyn::test::TestUtils::get_tmp_testdir("binary_file_wrapper_test");

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
      payload_to_file_little_endian(file, payload);
      payload_to_file_little_endian(file, label);
    }
    file.close();

    // tmp test file with 2 byte labels
    std::ofstream file_endian(file_name_endian_, std::ios::binary);
    const std::vector<std::pair<uint16_t, uint16_t>> data_endian = {
        {(1u << 8u) + 2u, 0}, {(3u << 8u) + 4u, 1}, {(5u << 8u) + 6u, 2}, {(7u << 8u) + 8u, 3}};
    for (const auto& [payload, label] : data_endian) {
      payload_to_file_little_endian(file_endian, payload);
      payload_to_file_little_endian(file_endian, label);
    }
    file_endian.close();

    ASSERT_TRUE(std::filesystem::exists(file_name_));
    ASSERT_TRUE(std::filesystem::exists(file_name_endian_));
  }

  static void payload_to_file_little_endian(std::ofstream& file, uint16_t data) {
    auto tmp = static_cast<uint8_t>(data & 0x00FFu);  // least significant byte
    file.write(reinterpret_cast<const char*>(&tmp), 1);
    tmp = static_cast<uint8_t>(data >> 8u);  // most significant byte
    file.write(reinterpret_cast<const char*>(&tmp), 1);
  }

  void TearDown() override {
    std::filesystem::remove_all(file_name_);
    std::filesystem::remove_all(file_name_endian_);
  }

  // tests needs to be in class context to access private members
  constexpr static std::array<unsigned char, 1> BYTES1{0b00000001};
  constexpr static std::array<unsigned char, 2> BYTES2{0b00000001, 0b00000010};
  constexpr static std::array<unsigned char, 4> BYTES4{0b00000001, 0b00000010, 0b00000011, 0b00000100};
  constexpr static std::array<unsigned char, 8> BYTES8{0b00000001, 0, 0, 0, 0, 0, 0, 0b00001000};

  static void test_int_from_bytes_little_endian() {
    ASSERT_EQ(BinaryFileWrapper::int_from_bytes_little_endian(BYTES1.data(), BYTES1.data() + 1u), 1);
    ASSERT_EQ(BinaryFileWrapper::int_from_bytes_little_endian(BYTES2.data(), BYTES2.data() + 2u), (2u << 8u) + 1u);
    ASSERT_EQ(BinaryFileWrapper::int_from_bytes_little_endian(BYTES4.data(), BYTES4.data() + 4u),
              (4ull << 24u) + (3u << 16u) + (2u << 8u) + 1u);
    ASSERT_EQ(BinaryFileWrapper::int_from_bytes_little_endian(BYTES8.data(), BYTES8.data() + 8u), (8ull << 56u) + 1u);
  }

  static void test_int_from_bytes_big_endian() {
    ASSERT_EQ(BinaryFileWrapper::int_from_bytes_big_endian(BYTES1.data(), BYTES1.data() + 1u), 1);
    ASSERT_EQ(BinaryFileWrapper::int_from_bytes_big_endian(BYTES2.data(), BYTES2.data() + 2u), (1u << 8u) + 2u);
    ASSERT_EQ(BinaryFileWrapper::int_from_bytes_big_endian(BYTES4.data(), BYTES4.data() + 4u),
              (1u << 24u) + (2u << 16u) + (3u << 8u) + 4u);
    ASSERT_EQ(BinaryFileWrapper::int_from_bytes_big_endian(BYTES8.data(), BYTES8.data() + 8u), (1ull << 56u) + 8u);
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

TEST_F(BinaryFileWrapperTest, TestIntFromBytesLittleEndian) { test_int_from_bytes_little_endian(); }
TEST_F(BinaryFileWrapperTest, TestIntFromBytesBigEndian) { test_int_from_bytes_big_endian(); }

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
  const YAML::Node little_endian_config = StorageTestUtils::get_dummy_file_wrapper_config("little");
  BinaryFileWrapper file_wrapper_little_endian(file_name_endian_, little_endian_config, filesystem_wrapper_);
  ASSERT_EQ(file_wrapper_little_endian.get_label(0), (1u << 8u) + 2u);
  ASSERT_EQ(file_wrapper_little_endian.get_label(1), (3u << 8u) + 4u);
  ASSERT_EQ(file_wrapper_little_endian.get_label(2), (5u << 8u) + 6u);
  ASSERT_EQ(file_wrapper_little_endian.get_label(3), (7u << 8u) + 8u);

  // [BIG ENDIAN]
  const YAML::Node big_endian_config = StorageTestUtils::get_dummy_file_wrapper_config("big");
  BinaryFileWrapper file_wrapper_big_endian(file_name_, big_endian_config, filesystem_wrapper_);
  ASSERT_EQ(file_wrapper_big_endian.get_label(0), (2u << 8u) + 1u);
  ASSERT_EQ(file_wrapper_big_endian.get_label(1), (4u << 8u) + 3u);
  ASSERT_EQ(file_wrapper_big_endian.get_label(2), (6u << 8u) + 5u);
  ASSERT_EQ(file_wrapper_big_endian.get_label(3), (8u << 8u) + 7u);
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
  const YAML::Node little_endian_config = StorageTestUtils::get_dummy_file_wrapper_config("little");
  BinaryFileWrapper file_wrapper_little_endian(file_name_, little_endian_config, filesystem_wrapper_);
  std::vector<int64_t> labels_little = file_wrapper_little_endian.get_all_labels();
  ASSERT_EQ(labels_little.size(), 4);
  ASSERT_EQ((labels_little)[0], (1u << 8u) + 2u);
  ASSERT_EQ((labels_little)[1], (3u << 8u) + 4u);
  ASSERT_EQ((labels_little)[2], (5u << 8u) + 6u);
  ASSERT_EQ((labels_little)[3], (7u << 8u) + 8u);

  // [BIG ENDIAN]
  const YAML::Node big_endian_config = StorageTestUtils::get_dummy_file_wrapper_config("big");
  BinaryFileWrapper file_wrapper_big_endian(file_name_, big_endian_config, filesystem_wrapper_);
  std::vector<int64_t> labels_big = file_wrapper_big_endian.get_all_labels();
  ASSERT_EQ(labels_big.size(), 4);
  ASSERT_EQ((labels_big)[0], (2u << 8u) + 1u);
  ASSERT_EQ((labels_big)[1], (4u << 8u) + 3u);
  ASSERT_EQ((labels_big)[2], (6u << 8u) + 5u);
  ASSERT_EQ((labels_big)[3], (8u << 8u) + 7u);
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

TEST_F(BinaryFileWrapperTest, TestNoLabels) {
  config_["has_labels"] = false;
  config_["label_size"] = 0;
  config_["has_targets"] = true;
  config_["target_size"] = 2;
  config_["sample_size"] = 2;

  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillOnce(testing::Return(16));
  auto stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);
  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::Return(stream_ptr));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);
  auto target = file_wrapper.get_target(0);
  ASSERT_EQ(target.size(), 2);  // we do have targets
  stream_ptr->close();
}

TEST_F(BinaryFileWrapperTest, TestNoTargets) {
  config_["has_labels"] = true;
  config_["label_size"] = 2;
  config_["has_targets"] = false;
  config_["target_size"] = 0;
  config_["sample_size"] = 2;

  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillOnce(testing::Return(16));
  auto stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);
  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::Return(stream_ptr));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);
  auto single_target = file_wrapper.get_target(0);
  ASSERT_TRUE(single_target.empty());
  auto multi_targets = file_wrapper.get_targets(0, 1);
  ASSERT_EQ(multi_targets.size(), 2u);
  ASSERT_TRUE(multi_targets[0].empty());
  ASSERT_TRUE(multi_targets[1].empty());
  stream_ptr->close();
}

TEST_F(BinaryFileWrapperTest, TestNoLabelsNoTargets) {
  // Record layout: [SAMPLE (4 bytes)] => 4 bytes/record => 16 total => 4 samples
  config_["has_labels"] = false;
  config_["label_size"] = 0;
  config_["has_targets"] = false;
  config_["target_size"] = 0;
  config_["sample_size"] = 4;

  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillOnce(testing::Return(16));
  auto stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);
  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::Return(stream_ptr));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);
  auto single_target = file_wrapper.get_target(0);
  ASSERT_TRUE(single_target.empty());  // no targets
  auto multi_targets = file_wrapper.get_targets(0, 1);
  ASSERT_EQ(multi_targets.size(), 2u);
  ASSERT_TRUE(multi_targets[0].empty());
  ASSERT_TRUE(multi_targets[1].empty());
  stream_ptr->close();
}
TEST_F(BinaryFileWrapperTest, TestGetTargets) {
  // Layout: label = 2 bytes, target = 1 byte, sample = 1 byte (total = 4 bytes per record)
  config_["has_labels"] = true;
  config_["label_size"] = 2;
  config_["has_targets"] = true;
  config_["target_size"] = 1;
  config_["sample_size"] = 1;  // sample_size must be non-zero

  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillOnce(testing::Return(16));
  auto stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);
  ASSERT_TRUE(stream_ptr->is_open());
  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::Return(stream_ptr));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);

  auto first_target = file_wrapper.get_target(0);
  ASSERT_EQ(first_target.size(), 1u);
  ASSERT_EQ(first_target[0], 12);

  auto all_targets = file_wrapper.get_targets(0, 3);
  ASSERT_EQ(all_targets.size(), 4u);
  ASSERT_EQ(all_targets[0][0], 12);
  ASSERT_EQ(all_targets[1][0], 13);
  ASSERT_EQ(all_targets[2][0], 14);
  ASSERT_EQ(all_targets[3][0], 15);

  stream_ptr->close();
}

TEST_F(BinaryFileWrapperTest, TestGetTargetsFromIndices) {
  // Use the same layout: 2 bytes label, 1 byte target, 1 byte sample
  config_["has_labels"] = true;
  config_["label_size"] = 2;
  config_["has_targets"] = true;
  config_["target_size"] = 1;
  config_["sample_size"] = 1;

  EXPECT_CALL(*filesystem_wrapper_, get_file_size(testing::_)).WillOnce(testing::Return(16));
  auto stream_ptr = std::make_shared<std::ifstream>();
  stream_ptr->open(file_name_, std::ios::binary);
  ASSERT_TRUE(stream_ptr->is_open());
  EXPECT_CALL(*filesystem_wrapper_, get_stream(testing::_)).WillOnce(testing::Return(stream_ptr));

  BinaryFileWrapper file_wrapper(file_name_, config_, filesystem_wrapper_);

  const std::vector<uint64_t> indices = {1, 3};
  auto targets = file_wrapper.get_targets_from_indices(indices);
  ASSERT_EQ(targets.size(), 2u);
  ASSERT_EQ(targets[0].size(), 1u);
  ASSERT_EQ(targets[1].size(), 1u);
  ASSERT_EQ(targets[0][0], 13);
  ASSERT_EQ(targets[1][0], 15);

  stream_ptr->close();
}

}  // namespace modyn::storage
