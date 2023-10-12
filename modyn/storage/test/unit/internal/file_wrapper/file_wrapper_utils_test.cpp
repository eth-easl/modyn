#include "internal/file_wrapper/file_wrapper_utils.hpp"

#include <gtest/gtest.h>

#include "test_utils.hpp"
#include "unit/internal/filesystem_wrapper/mock_filesystem_wrapper.hpp"

using namespace storage::file_wrapper;
using namespace storage::test;

TEST(UtilsTest, TestGetFileWrapper) {
  YAML::Node config = TestUtils::get_dummy_file_wrapper_config();  // NOLINT
  const std::shared_ptr<MockFilesystemWrapper> filesystem_wrapper = std::make_shared<MockFilesystemWrapper>();
  EXPECT_CALL(*filesystem_wrapper, get_file_size(testing::_)).WillOnce(testing::Return(8));
  EXPECT_CALL(*filesystem_wrapper, exists(testing::_)).WillRepeatedly(testing::Return(true));
  std::unique_ptr<FileWrapper> file_wrapper1 =
      get_file_wrapper("Testpath.txt", FileWrapperType::SINGLE_SAMPLE, config, filesystem_wrapper);
  ASSERT_NE(file_wrapper1, nullptr);
  ASSERT_EQ(file_wrapper1->get_type(), FileWrapperType::SINGLE_SAMPLE);

  config["file_extension"] = ".bin";
  std::unique_ptr<FileWrapper> file_wrapper2 =
      get_file_wrapper("Testpath.bin", FileWrapperType::BINARY, config, filesystem_wrapper);
  ASSERT_NE(file_wrapper2, nullptr);
  ASSERT_EQ(file_wrapper2->get_type(), FileWrapperType::BINARY);
}
