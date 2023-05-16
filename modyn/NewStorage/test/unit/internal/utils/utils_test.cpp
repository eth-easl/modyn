#include "internal/utils/utils.hpp"

#include <gtest/gtest.h>
#include <utime.h>

#include <filesystem>
#include <fstream>

#include "gmock/gmock.h"
#include "test_utils.hpp"
#include "unit/internal/filesystem_wrapper/mock_filesystem_wrapper.hpp"

using namespace storage;

TEST(UtilsTest, TestGetFilesystemWrapper) {
  const std::shared_ptr<AbstractFilesystemWrapper> filesystem_wrapper = Utils::get_filesystem_wrapper("Testpath", "LOCAL");
  ASSERT_NE(filesystem_wrapper, nullptr);
  ASSERT_EQ(filesystem_wrapper->get_name(), "LOCAL");

  ASSERT_THROW(Utils::get_filesystem_wrapper("Testpath", "UNKNOWN"), std::runtime_error);
}

TEST(UtilsTest, TestGetFileWrapper) {
  YAML::Node config = TestUtils::get_dummy_file_wrapper_config();  // NOLINT
  MockFilesystemWrapper filesystem_wrapper;
  EXPECT_CALL(filesystem_wrapper, get_file_size(testing::_)).WillOnce(testing::Return(8));
  std::unique_ptr<AbstractFileWrapper> file_wrapper1 = Utils::get_file_wrapper(
      "Testpath.txt", "SINGLE_SAMPLE", config, std::make_unique<MockFilesystemWrapper>(filesystem_wrapper));
  ASSERT_NE(file_wrapper1, nullptr);
  ASSERT_EQ(file_wrapper1->get_name(), "SINGLE_SAMPLE");

  config["file_extension"] = ".bin";
  std::unique_ptr<AbstractFileWrapper> file_wrapper2 = Utils::get_file_wrapper(
      "Testpath.bin", "BIN", config, std::make_unique<MockFilesystemWrapper>(filesystem_wrapper));
  ASSERT_NE(file_wrapper2, nullptr);
  ASSERT_EQ(file_wrapper2->get_name(), "BIN");

  ASSERT_THROW(Utils::get_file_wrapper("Testpath", "UNKNOWN", config,
                                       std::make_unique<MockFilesystemWrapper>(filesystem_wrapper)),
               std::runtime_error);
}

TEST(UtilsTest, TestJoinStringList) {
  std::vector<std::string> string_list = {"a", "b", "c"};
  ASSERT_EQ(Utils::join_string_list(string_list, ","), "a,b,c");

  string_list = {"a"};
  ASSERT_EQ(Utils::join_string_list(string_list, ","), "a");

  string_list = {};
  ASSERT_EQ(Utils::join_string_list(string_list, ","), "");
}

TEST(UtilsTest, TestGetTmpFilename) {
  const std::string tmp_filename = Utils::get_tmp_filename("Testpath");
  ASSERT_EQ(tmp_filename.substr(0, 8), "Testpath");
  ASSERT_EQ(tmp_filename.substr(tmp_filename.size() - 4, 4), ".tmp");
}