#include "internal/filesystem_wrapper/filesystem_wrapper_utils.hpp"

#include <gtest/gtest.h>

using namespace storage::filesystem_wrapper;

TEST(UtilsTest, TestGetFilesystemWrapper) {
  const std::shared_ptr<FilesystemWrapper> filesystem_wrapper =
      get_filesystem_wrapper("Testpath", FilesystemWrapperType::LOCAL);
  ASSERT_NE(filesystem_wrapper, nullptr);
  ASSERT_EQ(filesystem_wrapper->get_type(), FilesystemWrapperType::LOCAL);
}