#include "../../../src/internal/utils/Utils.hpp"
#include "../../TestUtils.hpp"
#include "../filesystem_wrapper/MockFilesystemWrapper.hpp"
#include "gmock/gmock.h"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <utime.h>

using namespace storage;

TEST(UtilsTest, TestGetFilesystemWrapper) {
  AbstractFilesystemWrapper *filesystem_wrapper =
      Utils::get_filesystem_wrapper("Testpath", "LOCAL");
  ASSERT_NE(filesystem_wrapper, nullptr);
  ASSERT_EQ(filesystem_wrapper->get_name(), "LOCAL");

  ASSERT_THROW(Utils::get_filesystem_wrapper("Testpath", "UNKNOWN"),
               std::runtime_error);
}

TEST(UtilsTest, TestGetFileWrapper) {
  YAML::Node config = TestUtils::get_dummy_file_wrapper_config();
  MockFilesystemWrapper filesystem_wrapper;
  EXPECT_CALL(filesystem_wrapper, get_file_size(testing::_))
      .WillOnce(testing::Return(8));
  AbstractFileWrapper *file_wrapper1 = Utils::get_file_wrapper(
      "Testpath.txt", "SINGLE_SAMPLE", config, &filesystem_wrapper);
  ASSERT_NE(file_wrapper1, nullptr);
  ASSERT_EQ(file_wrapper1->get_name(), "SINGLE_SAMPLE");

  config["file_extension"] = ".bin";
  AbstractFileWrapper *file_wrapper2 = Utils::get_file_wrapper(
      "Testpath.bin", "BIN", config, &filesystem_wrapper);
  ASSERT_NE(file_wrapper2, nullptr);
  ASSERT_EQ(file_wrapper2->get_name(), "BIN");

  ASSERT_THROW(Utils::get_file_wrapper("Testpath", "UNKNOWN", config,
                                       &filesystem_wrapper),
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
    std::string tmp_filename = Utils::get_tmp_filename("Testpath");
    ASSERT_EQ(tmp_filename.substr(0, 8), "Testpath");
    ASSERT_EQ(tmp_filename.substr(tmp_filename.size() - 4, 4), ".tmp");
}