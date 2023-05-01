#include "../../../src/internal/filesystem_wrapper/LocalFileSystemWrapper.hpp"
#include "gmock/gmock.h"
#include <gtest/gtest.h>
#include <fstream>
#include "../../Utils.hpp"
#include <filesystem>
#include <utime.h>

using namespace storage;

const char kPathSeparator =
#ifdef _WIN32
                            '\\';
#else
                            '/';
#endif

void teardown_test_dir() {
    std::string current_dir = std::filesystem::current_path();

    std::string test_dir = current_dir + kPathSeparator + "test_dir";
    std::filesystem::remove_all(test_dir);
}

std::string setup_test_dir() {
    teardown_test_dir();
    std::string current_dir = std::filesystem::current_path();

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
    return test_dir;
}

TEST(LocalFileSystemWrapperTest, TestGet)
{
    std::string test_base_dir = setup_test_dir();
    YAML::Node config = Utils::get_dummy_config();
    std::string file_name = test_base_dir + kPathSeparator + "test_file.txt";
    LocalFileSystemWrapper filesystem_wrapper = LocalFileSystemWrapper(file_name);
    std::vector<unsigned char> *bytes = filesystem_wrapper.get(file_name);
    ASSERT_EQ(bytes->size(), 8);
    ASSERT_EQ((*bytes)[0], '1');
    ASSERT_EQ((*bytes)[1], '2');
    ASSERT_EQ((*bytes)[2], '3');
    ASSERT_EQ((*bytes)[3], '4');
    ASSERT_EQ((*bytes)[4], '5');
    ASSERT_EQ((*bytes)[5], '6');
    ASSERT_EQ((*bytes)[6], '7');
    ASSERT_EQ((*bytes)[7], '8');
    teardown_test_dir();
}

TEST(LocalFileSystemWrapperTest, TestExists)
{
    std::string test_base_dir = setup_test_dir();
    YAML::Node config = Utils::get_dummy_config();
    std::string file_name = test_base_dir + kPathSeparator + "test_file.txt";
    LocalFileSystemWrapper filesystem_wrapper = LocalFileSystemWrapper(file_name);
    ASSERT_TRUE(filesystem_wrapper.exists(file_name));
    teardown_test_dir();
    ASSERT_FALSE(filesystem_wrapper.exists(file_name));
}

TEST(LocalFileSystemWrapperTest, TestList)
{
    std::string test_base_dir = setup_test_dir();
    YAML::Node config = Utils::get_dummy_config();
    LocalFileSystemWrapper filesystem_wrapper = LocalFileSystemWrapper(test_base_dir);
    std::vector<std::string> *files = filesystem_wrapper.list(test_base_dir);
    std::string file_name = test_base_dir + kPathSeparator + "test_file.txt";
    ASSERT_EQ(files->size(), 1);
    ASSERT_EQ((*files)[0], file_name);
}

TEST(LocalFileSystemWrapperTest, TestListRecursive) 
{
    std::string test_base_dir = setup_test_dir();

    YAML::Node config = Utils::get_dummy_config();
    LocalFileSystemWrapper filesystem_wrapper = LocalFileSystemWrapper(test_base_dir);
    std::vector<std::string> *files = filesystem_wrapper.list(test_base_dir, true);
    ASSERT_EQ(files->size(), 2);
    std::string file_name = test_base_dir + kPathSeparator + "test_file.txt";
    ASSERT_EQ((*files)[0], file_name);
    std::string file_name_2 = test_base_dir + kPathSeparator + "test_dir_2/test_file_2.txt";
    ASSERT_EQ((*files)[1], file_name_2);
}

TEST(LocalFileSystemWrapperTest, TestIsDirectory)
{
    std::string test_base_dir = setup_test_dir();
    YAML::Node config = Utils::get_dummy_config();
    LocalFileSystemWrapper filesystem_wrapper = LocalFileSystemWrapper(test_base_dir);
    ASSERT_TRUE(filesystem_wrapper.is_directory(test_base_dir));
    std::string file_name = test_base_dir + kPathSeparator + "test_file.txt";
    ASSERT_FALSE(filesystem_wrapper.is_directory(file_name));
    teardown_test_dir();
    ASSERT_FALSE(filesystem_wrapper.is_directory(test_base_dir));
}

TEST(LocalFileSystemWrapperTest, TestIsFile)
{
    std::string test_base_dir = setup_test_dir();
    YAML::Node config = Utils::get_dummy_config();
    LocalFileSystemWrapper filesystem_wrapper = LocalFileSystemWrapper(test_base_dir);
    ASSERT_FALSE(filesystem_wrapper.is_file(test_base_dir));
    std::string file_name = test_base_dir + kPathSeparator + "test_file.txt";
    ASSERT_TRUE(filesystem_wrapper.is_file(file_name));
    teardown_test_dir();
    ASSERT_FALSE(filesystem_wrapper.is_file(test_base_dir));
}

TEST(LocalFileSystemWrapperTest, TestGetFileSize)
{
    std::string test_base_dir = setup_test_dir();
    YAML::Node config = Utils::get_dummy_config();
    LocalFileSystemWrapper filesystem_wrapper = LocalFileSystemWrapper(test_base_dir);
    std::string file_name = test_base_dir + kPathSeparator + "test_file.txt";
    ASSERT_EQ(filesystem_wrapper.get_file_size(file_name), 8);
    teardown_test_dir();
}

TEST(LocalFileSystemWrapperTest, TestGetModifiedTime)
{
    std::string test_base_dir = setup_test_dir();
    YAML::Node config = Utils::get_dummy_config();
    LocalFileSystemWrapper filesystem_wrapper = LocalFileSystemWrapper(test_base_dir);
    std::string file_name = test_base_dir + kPathSeparator + "test_file.txt";
    ASSERT_EQ(filesystem_wrapper.get_modified_time(file_name), 0);
    teardown_test_dir();
}

TEST(LocalFileSystemWrapperTest, TestGetCreatedTime)
{
    std::string test_base_dir = setup_test_dir();
    YAML::Node config = Utils::get_dummy_config();
    LocalFileSystemWrapper filesystem_wrapper = LocalFileSystemWrapper(test_base_dir);
    std::string file_name = test_base_dir + kPathSeparator + "test_file.txt";
    struct stat file_info;
    int result = stat(file_name.c_str(), &file_info);
    time_t creation_time = file_info.st_ctime;
    ASSERT_EQ(filesystem_wrapper.get_created_time(file_name), creation_time);
    teardown_test_dir();
}

TEST(LocalFileSystemWrapperTest, TestJoin)
{
    std::string test_base_dir = setup_test_dir();
    YAML::Node config = Utils::get_dummy_config();
    LocalFileSystemWrapper filesystem_wrapper = LocalFileSystemWrapper(test_base_dir);
    std::string file_name = "test_file.txt";
    std::vector<std::string> paths = {test_base_dir, file_name};
    ASSERT_EQ(filesystem_wrapper.join(paths), test_base_dir + kPathSeparator + "" + file_name);
    teardown_test_dir();
}

TEST(LocalFileSystemWrapperTest, TestIsValidPath)
{
    std::string test_base_dir = setup_test_dir();
    YAML::Node config = Utils::get_dummy_config();
    LocalFileSystemWrapper filesystem_wrapper = LocalFileSystemWrapper(test_base_dir);
    std::string file_name = test_base_dir + kPathSeparator + "test_file.txt";
    ASSERT_TRUE(filesystem_wrapper.is_valid_path(test_base_dir));
    ASSERT_TRUE(filesystem_wrapper.is_valid_path(file_name));
    ASSERT_FALSE(filesystem_wrapper.is_valid_path(test_base_dir + kPathSeparator + ".." + kPathSeparator));
    teardown_test_dir();
}