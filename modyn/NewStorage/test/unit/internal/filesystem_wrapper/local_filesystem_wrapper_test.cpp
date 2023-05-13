#include "internal/filesystem_wrapper/local_filesystem_wrapper.hpp"

#include <gtest/gtest.h>
#include <utime.h>

#include <filesystem>
#include <fstream>

#include "gmock/gmock.h"
#include "test_utils.hpp"

using namespace storage;

const char kPathSeparator =
#ifdef _WIN32
    '\\';
#else
    '/';
#endif

std::string current_dir = std::filesystem::current_path();
std::string test_base_dir = current_dir + kPathSeparator + "test_dir";

class LocalFilesystemWrapperTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string test_dir = current_dir + kPathSeparator + "test_dir";
    std::filesystem::create_directory(test_dir);

    std::string test_dir_2 = test_dir + kPathSeparator + "test_dir_2";
    std::filesystem::create_directory(test_dir_2);

    std::string test_file = test_dir + kPathSeparator + "test_file.txt";
    std::ofstream file(test_file, std::ios::binary);
    file << "12345678";
    file.close();

    time_t zero_time = 0;
    utimbuf ub;
    ub.modtime = zero_time;

    utime(test_file.c_str(), &ub);

    std::string test_file_2 = test_dir_2 + kPathSeparator + "test_file_2.txt";
    std::ofstream file_2(test_file_2, std::ios::binary);
    file_2 << "12345678";
    file_2.close();
  }

  void TearDown() override {
    std::string current_dir = std::filesystem::current_path();

    std::string test_dir = current_dir + kPathSeparator + "test_dir";
    std::filesystem::remove_all(test_dir);
  }
};

TEST_F(LocalFilesystemWrapperTest, TestGet) {
  YAML::Node config = TestUtils::get_dummy_config();
  std::string file_name = test_base_dir + kPathSeparator + "test_file.txt";
  LocalFilesystemWrapper filesystem_wrapper = LocalFilesystemWrapper(file_name);
  std::vector<unsigned char>* bytes = filesystem_wrapper.get(file_name);
  ASSERT_EQ(bytes->size(), 8);
  ASSERT_EQ((*bytes)[0], '1');
  ASSERT_EQ((*bytes)[1], '2');
  ASSERT_EQ((*bytes)[2], '3');
  ASSERT_EQ((*bytes)[3], '4');
  ASSERT_EQ((*bytes)[4], '5');
  ASSERT_EQ((*bytes)[5], '6');
  ASSERT_EQ((*bytes)[6], '7');
  ASSERT_EQ((*bytes)[7], '8');
}

TEST_F(LocalFilesystemWrapperTest, TestExists) {
  YAML::Node config = TestUtils::get_dummy_config();
  std::string file_name = test_base_dir + kPathSeparator + "test_file.txt";
  LocalFilesystemWrapper filesystem_wrapper = LocalFilesystemWrapper(file_name);
  ASSERT_TRUE(filesystem_wrapper.exists(file_name));
  ASSERT_FALSE(filesystem_wrapper.exists(file_name));
}

TEST_F(LocalFilesystemWrapperTest, TestList) {
  YAML::Node config = TestUtils::get_dummy_config();
  LocalFilesystemWrapper filesystem_wrapper = LocalFilesystemWrapper(test_base_dir);
  std::vector<std::string>* files = filesystem_wrapper.list(test_base_dir);
  std::string file_name = test_base_dir + kPathSeparator + "test_file.txt";
  ASSERT_EQ(files->size(), 1);
  ASSERT_EQ((*files)[0], file_name);
}

TEST_F(LocalFilesystemWrapperTest, TestListRecursive) {
  YAML::Node config = TestUtils::get_dummy_config();
  LocalFilesystemWrapper filesystem_wrapper = LocalFilesystemWrapper(test_base_dir);
  std::vector<std::string>* files = filesystem_wrapper.list(test_base_dir, true);
  ASSERT_EQ(files->size(), 2);
  std::string file_name = test_base_dir + kPathSeparator + "test_file.txt";
  ASSERT_EQ((*files)[0], file_name);
  std::string file_name_2 = test_base_dir + kPathSeparator + "test_dir_2/test_file_2.txt";
  ASSERT_EQ((*files)[1], file_name_2);
}

TEST_F(LocalFilesystemWrapperTest, TestIsDirectory) {
  YAML::Node config = TestUtils::get_dummy_config();
  LocalFilesystemWrapper filesystem_wrapper = LocalFilesystemWrapper(test_base_dir);
  ASSERT_TRUE(filesystem_wrapper.is_directory(test_base_dir));
  std::string file_name = test_base_dir + kPathSeparator + "test_file.txt";
  ASSERT_FALSE(filesystem_wrapper.is_directory(file_name));
  ASSERT_FALSE(filesystem_wrapper.is_directory(test_base_dir));
}

TEST_F(LocalFilesystemWrapperTest, TestIsFile) {
  YAML::Node config = TestUtils::get_dummy_config();
  LocalFilesystemWrapper filesystem_wrapper = LocalFilesystemWrapper(test_base_dir);
  ASSERT_FALSE(filesystem_wrapper.is_file(test_base_dir));
  std::string file_name = test_base_dir + kPathSeparator + "test_file.txt";
  ASSERT_TRUE(filesystem_wrapper.is_file(file_name));
  ASSERT_FALSE(filesystem_wrapper.is_file(test_base_dir));
}

TEST_F(LocalFilesystemWrapperTest, TestGetFileSize) {
  YAML::Node config = TestUtils::get_dummy_config();
  LocalFilesystemWrapper filesystem_wrapper = LocalFilesystemWrapper(test_base_dir);
  std::string file_name = test_base_dir + kPathSeparator + "test_file.txt";
  ASSERT_EQ(filesystem_wrapper.get_file_size(file_name), 8);
}

TEST_F(LocalFilesystemWrapperTest, TestGetModifiedTime) {
  YAML::Node config = TestUtils::get_dummy_config();
  LocalFilesystemWrapper filesystem_wrapper = LocalFilesystemWrapper(test_base_dir);
  std::string file_name = test_base_dir + kPathSeparator + "test_file.txt";
  ASSERT_EQ(filesystem_wrapper.get_modified_time(file_name), 0);
}

TEST_F(LocalFilesystemWrapperTest, TestGetCreatedTime) {
  YAML::Node config = TestUtils::get_dummy_config();
  LocalFilesystemWrapper filesystem_wrapper = LocalFilesystemWrapper(test_base_dir);
  std::string file_name = test_base_dir + kPathSeparator + "test_file.txt";
  struct stat file_info;
  time_t creation_time = file_info.st_ctime;
  ASSERT_EQ(filesystem_wrapper.get_created_time(file_name), creation_time);
}

TEST_F(LocalFilesystemWrapperTest, TestJoin) {
  YAML::Node config = TestUtils::get_dummy_config();
  LocalFilesystemWrapper filesystem_wrapper = LocalFilesystemWrapper(test_base_dir);
  std::string file_name = "test_file.txt";
  std::vector<std::string> paths = {test_base_dir, file_name};
  ASSERT_EQ(filesystem_wrapper.join(paths), test_base_dir + kPathSeparator + "" + file_name);
}

TEST_F(LocalFilesystemWrapperTest, TestIsValidPath) {
  YAML::Node config = TestUtils::get_dummy_config();
  LocalFilesystemWrapper filesystem_wrapper = LocalFilesystemWrapper(test_base_dir);
  std::string file_name = test_base_dir + kPathSeparator + "test_file.txt";
  ASSERT_TRUE(filesystem_wrapper.is_valid_path(test_base_dir));
  ASSERT_TRUE(filesystem_wrapper.is_valid_path(file_name));
  ASSERT_FALSE(filesystem_wrapper.is_valid_path(test_base_dir + kPathSeparator + ".." + kPathSeparator));
}