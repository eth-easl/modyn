#include "internal/file_wrapper/file_wrapper_utils.hpp"

#include <gtest/gtest.h>

#include "internal/filesystem_wrapper/mock_filesystem_wrapper.hpp"
#include "storage_test_utils.hpp"
#include "test_utils.hpp"

using namespace modyn::storage;

TEST(UtilsTest, TestGetFileWrapper) {
  YAML::Node config = StorageTestUtils::get_dummy_file_wrapper_config();  // NOLINT
  const std::shared_ptr<MockFilesystemWrapper> filesystem_wrapper = std::make_shared<MockFilesystemWrapper>();
  EXPECT_CALL(*filesystem_wrapper, get_file_size(testing::_)).WillOnce(testing::Return(8));
  EXPECT_CALL(*filesystem_wrapper, exists(testing::_)).WillRepeatedly(testing::Return(true));
  std::unique_ptr<FileWrapper> file_wrapper1 =
      get_file_wrapper("Testpath.txt", FileWrapperType::SINGLE_SAMPLE, config, filesystem_wrapper);
  ASSERT_NE(file_wrapper1, nullptr);
  ASSERT_EQ(file_wrapper1->get_type(), FileWrapperType::SINGLE_SAMPLE);

  std::unique_ptr<std::ifstream> binary_stream = std::make_unique<std::ifstream>();
  binary_stream->open("Testpath.bin", std::ios::binary);
  std::ifstream& binary_reference = *binary_stream;
  EXPECT_CALL(*filesystem_wrapper, get_stream(testing::_)).WillOnce(testing::ReturnRef(binary_reference));
  config["file_extension"] = ".bin";
  std::unique_ptr<FileWrapper> file_wrapper2 =
      get_file_wrapper("Testpath.bin", FileWrapperType::BINARY, config, filesystem_wrapper);
  ASSERT_NE(file_wrapper2, nullptr);
  ASSERT_EQ(file_wrapper2->get_type(), FileWrapperType::BINARY);

  std::unique_ptr<std::ifstream> csv_stream = std::make_unique<std::ifstream>();
  csv_stream->open("Testpath.csv", std::ios::binary);
  std::ifstream& csv_reference = *csv_stream;
  EXPECT_CALL(*filesystem_wrapper, get_stream(testing::_)).WillOnce(testing::ReturnRef(csv_reference));
  config["file_extension"] = ".csv";
  std::unique_ptr<FileWrapper> file_wrapper3 =
      get_file_wrapper("Testpath.csv", FileWrapperType::CSV, config, filesystem_wrapper);
  ASSERT_NE(file_wrapper3, nullptr);
  ASSERT_EQ(file_wrapper3->get_type(), FileWrapperType::CSV);
}
